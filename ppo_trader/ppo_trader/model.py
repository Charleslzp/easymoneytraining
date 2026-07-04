"""
PPO 模型定义 - 从 train_e2e_ppo.py 抽取
"""
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False

def _mamba_available_on_device(device: torch.device) -> bool:
    return HAS_MAMBA


class States(IntEnum):
    EMPTY = 0
    LONG  = 1
    SHORT = 2

class FlatActions(IntEnum):
    WAIT = 0
    OPEN = 1

class HoldActions(IntEnum):
    CLOSE = 0
    HOLD  = 1


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, use_mamba: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.has_mamba = False
        self.gru = None

        # 如果use_mamba=True且mamba_ssm可用，则使用Mamba（不管是CPU还是GPU）
        if HAS_MAMBA and use_mamba:
            try:
                self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                self.has_mamba = True
            except Exception as e:
                print(f"Warning: Failed to initialize Mamba: {e}")
                self.has_mamba = False

        # 只有在不使用Mamba时才创建GRU作为fallback
        if not self.has_mamba:
            self.gru = nn.GRU(d_model, d_model, batch_first=True)

        self.dropout = nn.Identity() if float(dropout) <= 0.0 else nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        if self.has_mamba:
            x = self.mamba(x)
        elif self.gru is not None:
            x, _ = self.gru(x)
        else:
            raise RuntimeError("Neither Mamba nor GRU is available")
        return self.dropout(x) + residual


class MambaBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout,
                 use_temporal_attention=True, temporal_attn_hidden=64, use_mamba=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, dropout, use_mamba=use_mamba) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.use_temporal_attention = bool(use_temporal_attention)
        if self.use_temporal_attention:
            attn_hidden = max(int(temporal_attn_hidden), 16)
            self.temporal_attn = nn.Sequential(
                nn.Linear(hidden_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
            self.temporal_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        last = x[:, -1, :]
        if not self.use_temporal_attention:
            return last
        scores = self.temporal_attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (weights * x).sum(dim=1)
        g = torch.sigmoid(self.temporal_gate)
        return g * pooled + (1.0 - g) * last


class StudentPolicy(nn.Module):
    """
    run19 模型结构，硬编码参数：
      hidden=256, num_layers=4, dropout=0.1,
      pos_dim=64, input_dim=5,
      use_temporal_attention=True, temporal_attn_hidden=64
    """
    def __init__(self, use_mamba: bool = True):
        super().__init__()
        hidden   = 256
        pos_dim  = 64
        dropout  = 0.1
        fuse_dim = hidden + pos_dim  # use_asset_ids=False

        self.backbone = MambaBackbone(
            input_dim=5, hidden_dim=hidden, num_layers=4, dropout=dropout,
            use_temporal_attention=True, temporal_attn_hidden=64, use_mamba=use_mamba,
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(6, pos_dim), nn.ReLU(), nn.Linear(pos_dim, pos_dim)
        )
        self.open_fuse = nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.close_fuse = nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.open_gate_head  = nn.Linear(hidden, 2)
        self.entry_size_head = nn.Linear(hidden, 1)
        self.entry_log_std   = nn.Parameter(torch.tensor(-0.5))
        self.close_head      = nn.Linear(hidden, 2)
        self.strength_head   = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(hidden, 1)
        )
        self.teacher_repr_head = nn.Linear(hidden, 512)
        self.max_logit_gap = 20.0

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StudentPolicy":
        dev = torch.device(device)
        # 始终尝试使用Mamba（CPU和GPU都支持）
        use_mamba = HAS_MAMBA
        m = cls(use_mamba=use_mamba)
        ckpt = torch.load(path, map_location=device)
        state = ckpt["policy"]
        own   = set(m.state_dict().keys())
        m.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
        m.to(dev)
        m.eval()
        return m

    @torch.no_grad()
    def act_deterministic(
        self,
        market: torch.Tensor,          # (B, T, 5)
        pos_info: torch.Tensor,        # (B, 6)
        position_states: torch.Tensor, # (B,) int
        action_mask: Optional[torch.Tensor] = None,  # (B, 2)
    ) -> torch.Tensor:                 # (B, 2)
        encoded = self.backbone(market)
        pos     = self.pos_proj(pos_info)
        shared  = torch.cat([encoded, pos], dim=-1)
        open_feat  = self.open_fuse(shared)
        close_feat = self.close_fuse(shared)

        open_logits  = self.open_gate_head(open_feat)   # (B, 2)
        close_logits = self.close_head(close_feat)       # (B, 2)
        size_mean    = self.entry_size_head(open_feat).squeeze(-1)  # (B,)

        B = market.shape[0]
        actions = torch.zeros(B, 2, device=market.device)

        empty = position_states == int(States.EMPTY)
        hold  = ~empty

        if empty.any():
            logits = open_logits[empty]
            if action_mask is not None:
                mask = action_mask[empty].to(logits.dtype)
                logits = logits + (mask - 1.0) * 1e9
            gate = logits.argmax(dim=-1)
            actions[empty, 0] = gate.float()
            open_sel = empty.nonzero(as_tuple=True)[0][gate == int(FlatActions.OPEN)]
            if len(open_sel):
                actions[open_sel, 1] = torch.tanh(size_mean[open_sel])

        if hold.any():
            logits = close_logits[hold]
            if action_mask is not None:
                mask = action_mask[hold].to(logits.dtype)
                logits = logits + (mask - 1.0) * 1e9
            gate = logits.argmax(dim=-1)
            actions[hold, 0] = gate.float()

        return actions
