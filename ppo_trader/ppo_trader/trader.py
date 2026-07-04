"""
PPO LiveTrader - 推理封装
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np
import torch

from .model import StudentPolicy, States, FlatActions, HoldActions

SEQ_LEN = 300  # 修改为300以匹配策略配置
ATR_WINDOW = 14
STOP_MULTIPLE = 1.5
VOL_FLOOR = 0.008
MAX_HOLD = 40
MIN_HOLD = 3
MAX_EMPTY = 30
MIN_EMPTY = 3


@dataclass
class Position:
    state: str = "empty"   # empty / long / short
    entry_price: float = 0.0
    size: float = 0.0
    position: float = 0.0
    holding_steps: int = 0
    empty_steps: int = 0
    peak_return: float = 0.0
    peak_step: int = 0
    unrealized_pnl: float = 0.0
    step_count: int = 0


@dataclass
class Signal:
    action: str     # wait / long / short / close / hold
    size: float     # 0~1
    stop_pct: float # 止损比例，如 -0.033（每根K线更新）
    stop_price: float
    raw_size: float


def _resolve_config_path(config_path: Optional[str] = None) -> Optional[Path]:
    candidates = []
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    if config_path:
        configured = Path(config_path).expanduser()
        if configured.is_absolute():
            candidates.append(configured)
        else:
            candidates.extend([
                cwd / configured,
                here / configured,
                here.parent / configured,
                here.parent.parent / configured,
            ])
    candidates.extend([
        cwd / "configs/e2e_ppo_run19_sl15_valrandom.json",
        here.parent.parent / "configs/e2e_ppo_run19_sl15_valrandom.json",
    ])
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_env_settings(config_path: Optional[str] = None) -> dict:
    path = _resolve_config_path(config_path)
    if path is None:
        return {
            "seq_length": SEQ_LEN,
            "max_holding_steps": MAX_HOLD,
            "min_holding_steps": MIN_HOLD,
            "max_empty_steps": MAX_EMPTY,
            "min_empty_steps": MIN_EMPTY,
            "stop_loss_vol_multiple": STOP_MULTIPLE,
            "reward_vol_floor": VOL_FLOOR,
            "position_scale": 1.0,
        }
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    env_cfg = cfg.get("env", {})
    data_cfg = cfg.get("data", {})
    return {
        "seq_length": int(data_cfg.get("seq_length", SEQ_LEN)),
        "max_holding_steps": int(env_cfg.get("max_holding_steps", MAX_HOLD)),
        "min_holding_steps": int(env_cfg.get("min_holding_steps", MIN_HOLD)),
        "max_empty_steps": int(env_cfg.get("max_empty_steps", MAX_EMPTY)),
        "min_empty_steps": int(env_cfg.get("min_empty_steps", MIN_EMPTY)),
        "stop_loss_vol_multiple": float(env_cfg.get("stop_loss_vol_multiple", STOP_MULTIPLE)),
        "reward_vol_floor": float(env_cfg.get("reward_vol_floor", VOL_FLOOR)),
        "position_scale": float(env_cfg.get("position_scale", 1.0)),
    }


def normalize_ohlcv(ohlcv: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """(N,5) → (seq_len,5) 滚动窗口归一化，对齐 test_run19_live_norm.py。"""
    out = np.zeros_like(ohlcv, dtype=np.float32)
    for i in range(4):
        p = ohlcv[:, i]
        ret = np.zeros(len(p), dtype=np.float32)
        ret[1:] = (p[1:] - p[:-1]) / (p[:-1] + 1e-8)
        out[:, i] = ret
    vol = ohlcv[:, 4]
    lv = np.log(vol + 1.0)
    out[:, 4] = (lv - lv.mean()) / (lv.std() + 1e-8)
    return out[-seq_len:]


def compute_atr(ohlcv: np.ndarray, atr_window: int = ATR_WINDOW, vol_floor: float = VOL_FLOOR) -> float:
    """返回 ATR/price 比例，对齐 train_e2e_ppo.py:_atr。"""
    n = min(atr_window + 1, len(ohlcv))
    bars = ohlcv[-n:]
    if len(bars) < 2:
        return vol_floor
    highs, lows, prev_c = bars[1:, 1], bars[1:, 2], bars[:-1, 3]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    cur = float(bars[-1, 3])
    return max(float(np.mean(tr)) / cur, vol_floor) if cur > 0 else vol_floor


def encode_pos_info(pos: Position, max_hold: int = MAX_HOLD, max_empty: int = MAX_EMPTY) -> np.ndarray:
    info = np.zeros(6, dtype=np.float32)
    info[0] = float(pos.position)
    if pos.state == "empty":
        info[4] = min(pos.empty_steps / max(max_empty, 1), 1.0)
        return info
    info[1] = np.tanh(pos.unrealized_pnl * 20.0)
    info[2] = np.tanh(pos.peak_return * 20.0)
    info[3] = np.tanh(max(pos.peak_return - pos.unrealized_pnl, 0.0) * 20.0)
    info[4] = min(pos.holding_steps / max(max_hold, 1), 1.0)
    steps_since_peak = max(pos.step_count - pos.peak_step, 0)
    info[5] = min(steps_since_peak / max(max_hold, 1), 1.0)
    return info


class LiveTrader:
    def __init__(self, checkpoint: str, config: Optional[str] = None, device: str = "cpu", seq_length: Optional[int] = None):
        self.device = torch.device(device)
        self.settings = _load_env_settings(config)
        # 如果显式传入seq_length参数，覆盖配置文件中的值
        if seq_length is not None:
            self.seq_len = int(seq_length)
        else:
            self.seq_len = int(self.settings["seq_length"])
        self.max_hold = int(self.settings["max_holding_steps"])
        self.min_hold = int(self.settings["min_holding_steps"])
        self.max_empty = int(self.settings["max_empty_steps"])
        self.min_empty = int(self.settings["min_empty_steps"])
        self.stop_multiple = float(self.settings["stop_loss_vol_multiple"])
        self.vol_floor = max(float(self.settings["reward_vol_floor"]), 1e-4)
        self.position_scale = max(float(self.settings["position_scale"]), 0.0)
        self.policy = StudentPolicy.load(checkpoint, device=device)
        self.policy.to(self.device)

    def _encode_pos_state(self, pos: Position) -> int:
        return {"empty": int(States.EMPTY), "long": int(States.LONG), "short": int(States.SHORT)}.get(pos.state, 0)

    def _action_mask(self, pos: Position) -> np.ndarray:
        mask = np.ones(2, dtype=np.float32)
        if pos.state == "empty":
            mask[int(FlatActions.OPEN)] = 1.0 if pos.empty_steps >= self.min_empty else 0.0
            return mask
        if pos.holding_steps < self.min_hold:
            mask[int(HoldActions.CLOSE)] = 0.0
        if pos.holding_steps >= self.max_hold:
            mask[int(HoldActions.HOLD)] = 0.0
            mask[int(HoldActions.CLOSE)] = 1.0
        return mask

    def predict(
        self,
        ohlcv_dict: Dict[str, np.ndarray],
        positions: Dict[str, Position],
    ) -> Dict[str, Signal]:
        symbols = list(ohlcv_dict.keys())
        n = len(symbols)

        market_b = np.zeros((n, self.seq_len, 5), dtype=np.float32)
        pos_b = np.zeros((n, 6), dtype=np.float32)
        state_b = np.zeros(n, dtype=np.int64)
        mask_b = np.ones((n, 2), dtype=np.float32)

        for i, sym in enumerate(symbols):
            pos = positions.get(sym, Position())
            market_b[i] = normalize_ohlcv(ohlcv_dict[sym], seq_len=self.seq_len)
            pos_b[i] = encode_pos_info(pos, max_hold=self.max_hold, max_empty=self.max_empty)
            state_b[i] = self._encode_pos_state(pos)
            mask_b[i] = self._action_mask(pos)

        actions = self.policy.act_deterministic(
            torch.from_numpy(market_b).to(self.device),
            torch.from_numpy(pos_b).to(self.device),
            torch.from_numpy(state_b).to(self.device),
            torch.from_numpy(mask_b).to(self.device),
        ).cpu().numpy()

        signals = {}
        for i, sym in enumerate(symbols):
            pos = positions.get(sym, Position())
            ohlcv = ohlcv_dict[sym]
            cur = float(ohlcv[-1, 3])
            atr_pct = compute_atr(ohlcv, atr_window=ATR_WINDOW, vol_floor=self.vol_floor)
            stop_pct = -self.stop_multiple * atr_pct
            gate = int(round(float(actions[i, 0])))
            raw_size = float(actions[i, 1]) if actions.shape[1] > 1 else 0.0

            if pos.state == "empty":
                if gate == int(FlatActions.OPEN):
                    direction = "long" if raw_size >= 0 else "short"
                    stop_price = cur * (1.0 + stop_pct) if direction == "long" else cur * (1.0 - stop_pct)
                    signals[sym] = Signal(direction, abs(raw_size), stop_pct, stop_price, raw_size)
                else:
                    signals[sym] = Signal("wait", 0.0, stop_pct, 0.0, raw_size)
            else:
                if gate == int(HoldActions.CLOSE):
                    signals[sym] = Signal("close", 0.0, stop_pct, 0.0, raw_size)
                else:
                    entry = max(float(pos.entry_price), 1e-8)
                    stop_price = entry * (1.0 + stop_pct) if pos.state == "long" else entry * (1.0 - stop_pct)
                    signals[sym] = Signal("hold", pos.size, stop_pct, stop_price, raw_size)

        return signals
