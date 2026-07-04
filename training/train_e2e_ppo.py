"""
End-to-end PPO training with frozen teacher regularization.

Pipeline:
1) Load pure OHLCV from parquet
2) Build chronological train/val/test splits per symbol
3) Collect rollouts from vectorized trading env
4) Train student Mamba/GRU policy + value with PPO
5) Optionally regularize student policy toward frozen teacher policy
6) Periodic validation + checkpointing

This script is independent from stage1/stage2/stage3 flow.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic enums and logging
# -----------------------------


class FlatActions(IntEnum):
    WAIT = 0
    OPEN = 1


class HoldActions(IntEnum):
    CLOSE = 0
    HOLD = 1


class States(IntEnum):
    EMPTY = 0
    LONG = 1
    SHORT = 2


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# -----------------------------
# Config
# -----------------------------


@dataclass
class DataConfig:
    data_file: str = "data/processed/market_data_10M.parquet"
    symbols: List[str] = field(default_factory=list)  # empty = all symbols
    seq_length: int = 300
    overlap_lookback: int = 0
    normalize_method: str = "returns"  # returns / zscore
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    min_bars: int = 700
    price_basis: str = "adjusted"  # adjusted / raw


@dataclass
class ModelConfig:
    input_dim: int = 5
    hidden_dim: int = 256
    num_layers: int = 4
    value_hidden_dim: int = 256
    value_num_layers: int = 4
    position_dim: int = 64
    asset_embed_dim: int = 16
    use_asset_ids: bool = True
    dropout: float = 0.0
    max_logit_gap: float = 4.0
    use_temporal_attention: bool = True
    value_use_temporal_attention: bool = True
    temporal_attn_hidden: int = 64


@dataclass
class EnvConfig:
    transaction_cost: float = 0.001
    stop_loss: float = -0.04
    take_profit: float = 0.08
    max_holding_steps: int = 49
    max_empty_steps: int = 49
    min_holding_steps: int = 8
    min_empty_steps: int = 7
    entry_threshold: float = 0.20
    position_scale: float = 1.30
    open_fee_penalty: float = 0.0  # deprecated: folded into transaction_cost at open
    potential_scale: float = 0.02
    potential_drawdown_weight: float = 0.60
    reward_vol_window: int = 20
    reward_vol_floor: float = 0.008
    reward_hold_norm_denom: float = 1.5
    stop_loss_vol_multiple: float = 0.0
    empty_first_step_bonus: float = 0.0001
    empty_reward_steps: int = 5
    empty_step_penalty: float = 0.0
    empty_penalty_start: int = 30
    empty_late_penalty: float = 0.0001
    long_only_market_groups: List[str] = field(default_factory=lambda: ["CN_EQ", "CN_A_SH", "CN_A_SZ"])


@dataclass
class PPOConfig:
    max_updates: int = 3000
    trajectory_length: int = 200
    val_trajectory_length: int = 300
    envs_per_batch: int = 96
    val_envs: int = 180
    parallel_value_policy_update: bool = True
    use_bf16_amp: bool = False
    market_bucket_sampling: bool = False
    market_bucket_ratios: Dict[str, float] = field(default_factory=lambda: {
        "us": 0.3,
        "cn": 0.3,
        "crypto": 0.3,
        "other": 0.1,
    })

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.02
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    policy_epochs: int = 3
    value_epochs: int = 5
    mini_batch_size: int = 512
    value_infer_batch_size: int = 512

    normalize_advantages: bool = True
    normalize_returns: bool = False

    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_updates: int = 100
    kl_target: float = 0.02
    log_ratio_clip: float = 6.0
    minibatch_kl_stop_mult: float = 3.0

    # align with core training: light long/short balance regularization on OPEN state
    enable_direction_balance: bool = True
    direction_balance_strength: float = 0.2

    val_interval: int = 20
    save_interval: int = 50
    log_interval: int = 5
    val_temperature: float = 0.3
    val_fixed_seed: int = 12345
    val_fixed_deterministic: bool = True
    val_fixed_envs: int = 0
    val_fixed_trajectory_length: int = 0
    val_fixed_manifest_path: str = ""
    val_fixed_manifest_rebuild: bool = False
    val_fixed_manifest_stride: int = 0
    val_random_enabled: bool = False
    val_random_envs: int = 0
    val_random_trajectory_length: int = 0

    # Gradient accumulation and EMA
    gradient_accumulation_steps: int = 1  # 1=disabled, 2/4=accumulate gradients
    use_policy_ema: bool = False  # Use EMA for policy network
    policy_ema_decay: float = 0.999  # EMA decay rate (0.999-0.9999)


@dataclass
class TeacherConfig:
    enabled: bool = True
    # Kron teacher sources (repo id or local path)
    kron_model_repo: str = "NeoQuasar/Kronos-small"
    kron_tokenizer_repo: str = "NeoQuasar/Kronos-Tokenizer-base"
    kron_model_path: str = ""  # optional local directory/file override
    kron_tokenizer_path: str = ""  # optional local directory/file override
    kron_device: str = "cuda:0"
    kron_max_context: int = 512
    kron_clip: float = 5.0
    kron_logit_scale: float = 6.0
    kron_hold_bias: float = 0.25
    use_offline_cache: bool = True
    cache_dir: str = "data/cache/kron_strength"
    cache_splits: List[str] = field(default_factory=lambda: ["train", "val"])
    cache_force_rebuild: bool = False
    cache_batch_size: int = 256
    distill_mode: str = "action"  # action / representation
    repr_cache_dir: str = "data/cache/kron_repr"
    repr_dim: int = 512
    repr_pool: str = "last"  # last / mean
    repr_coef_start: float = 0.05
    repr_coef_end: float = 0.0
    repr_decay_updates: int = 1000

    # teacher regularization weight
    kl_coef_start: float = 2.0
    kl_coef_end: float = 0.2
    kl_decay_updates: int = 2000
    kl_trust_scale: float = 0.30
    strength_loss_weight: float = 0.20

    # model architecture: frozen teacher(Kron) + trainable student(Mamba) + policy fusion
    fuse_in_policy: bool = True
    teacher_logit_weight: float = 0.35
    teacher_logit_weight_end: float = 0.0
    teacher_logit_decay_updates: int = 0


def _teacher_distill_mode(cfg: TeacherConfig) -> str:
    mode = str(getattr(cfg, "distill_mode", "action") or "action").strip().lower()
    return mode if mode in {"action", "representation"} else "action"


@dataclass
class StudentInitConfig:
    checkpoint: str = ""  # optional pretrained student checkpoint
    strict: bool = False


@dataclass
class EmailConfig:
    enabled: bool = True
    config_path: str = "core/email_config.json"
    smtp_server: str = "smtp.163.com"
    smtp_port: int = 465
    sender_email: str = ""
    sender_password: str = ""
    recipient_email: str = ""


@dataclass
class PhaseConfig:
    # stage1_policy: freeze Mamba backbone, train policy heads
    # stage2_mamba: train only Mamba backbone, freeze policy heads
    # stage3_joint: train all policy params
    mode: str = "stage1_policy"


@dataclass
class RunConfig:
    seed: int = 42
    policy_device: str = "cuda:0"
    value_device: str = "cuda:0"
    output_dir: str = "checkpoints/ppo_e2e"
    resume: str = ""  # optional checkpoint path


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student_init: StudentInitConfig = field(default_factory=StudentInitConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _apply_overrides(cfg_obj, overrides: Dict) -> None:
    for k, v in overrides.items():
        if not hasattr(cfg_obj, k):
            continue
        cur = getattr(cfg_obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _apply_overrides(cur, v)
        else:
            setattr(cfg_obj, k, v)


def load_config(path: Optional[str]) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        _apply_overrides(cfg, overrides)
    ratio_sum = float(cfg.data.train_ratio) + float(cfg.data.val_ratio)
    if ratio_sum > 1.0 + 1e-8:
        raise ValueError(
            f"Invalid data split ratios: train_ratio + val_ratio = {ratio_sum:.6f} > 1.0"
        )
    return cfg


# -----------------------------
# Data pipeline (pure OHLCV)
# -----------------------------


class OHLCVNormalizer:
    def __init__(self, method: str = "returns"):
        self.method = method

    def normalize(self, ohlcv: np.ndarray) -> np.ndarray:
        if self.method == "returns":
            return self._normalize_returns(ohlcv)
        if self.method == "zscore":
            return self._normalize_zscore(ohlcv)
        raise ValueError(f"Unknown normalize method: {self.method}")

    @staticmethod
    def _normalize_returns(ohlcv: np.ndarray) -> np.ndarray:
        t = len(ohlcv)
        out = np.zeros_like(ohlcv, dtype=np.float32)

        for i in range(4):
            p = ohlcv[:, i]
            ret = np.zeros(t, dtype=np.float32)
            ret[1:] = (p[1:] - p[:-1]) / (p[:-1] + 1e-8)
            out[:, i] = ret

        vol = ohlcv[:, 4]
        lv = np.log(vol + 1.0)
        out[:, 4] = (lv - lv.mean()) / (lv.std() + 1e-8)
        return out

    @staticmethod
    def _normalize_zscore(ohlcv: np.ndarray) -> np.ndarray:
        out = np.zeros_like(ohlcv, dtype=np.float32)
        for i in range(5):
            c = ohlcv[:, i]
            if i == 4:
                c = np.log(c + 1.0)
            out[:, i] = (c - c.mean()) / (c.std() + 1e-8)
        return out


@dataclass
class Segment:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


@dataclass
class SymbolData:
    symbol: str
    features: np.ndarray
    raw_ohlcva: np.ndarray
    time_features: np.ndarray
    prices: np.ndarray
    segment_indices: List[Segment]
    market: str = ""
    asset_class: str = ""
    tier: str = "tier3"
    corp_action_cash_dividend_per10: Optional[np.ndarray] = None
    corp_action_share_multiplier: Optional[np.ndarray] = None
    corp_action_rights_ratio_per10: Optional[np.ndarray] = None
    corp_action_rights_event: Optional[np.ndarray] = None


def _effective_overlap_lookback(seq_len: int, overlap_lookback: int, split: str) -> int:
    if split not in {"val", "test"}:
        return 0
    return max(min(int(overlap_lookback), int(seq_len)), 0)


def _decision_start_index(split_start: int, seq_len: int, overlap_lookback: int) -> int:
    warmup = max(int(seq_len) - int(overlap_lookback), 0)
    return max(int(split_start) + warmup, int(seq_len))


def _window_start_index(split_start: int, overlap_lookback: int) -> int:
    return max(int(split_start) - int(overlap_lookback), 0)


def _min_split_width(seq_len: int, max_steps: int, overlap_lookback: int) -> int:
    warmup = max(int(seq_len) - int(overlap_lookback), 0)
    return warmup + int(max_steps) + 10


def _market_group_from_metadata(symbol: str, market: str = "", asset_class: str = "") -> str:
    m = (market or "").strip().lower()
    a = (asset_class or "").strip().lower()

    if a == "crypto" or "crypto" in m or "binance" in m:
        return "CRYPTO"
    if m in {"us", "us_equity"}:
        return "US_EQ"
    if m in {"china", "china_equity"}:
        return "CN_EQ"
    if m == "cn" and a in {"commodity", "futures", "mixed"}:
        return "CN_FUT"
    if m and a == "equity" and "us" in m:
        return "US_EQ"
    if m and a == "equity" and ("china" in m or m.startswith("cn")):
        return "CN_EQ"
    return _infer_market_group(symbol)


def _sampling_market_bucket(symbol: str, market: str = "", asset_class: str = "") -> str:
    group = _market_group_from_metadata(symbol=symbol, market=market, asset_class=asset_class)
    if group in {"CN_EQ", "CN_FUT", "CN_A_SZ", "CN_A_SH"}:
        return "cn"
    if group == "CRYPTO":
        return "crypto"
    if group == "US_EQ":
        return "us"
    return "other"


def _normalize_market_bucket_ratios(raw: Dict[str, float]) -> Dict[str, float]:
    default = {"us": 0.3, "cn": 0.3, "crypto": 0.3, "other": 0.1}
    weights = {k: max(float(raw.get(k, default[k])), 0.0) for k in default}
    total = sum(weights.values())
    if total <= 0.0:
        return default
    return {k: v / total for k, v in weights.items()}


@dataclass
class EvalEpisode:
    symbol: str
    start_idx: int
    bucket: str
    market_group: str
    score: float = 0.0


def _build_asset_ranges(
    symbol_store: Dict[str, SymbolData],
    cfg: Config,
    mode: str,
    max_steps: int,
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, List[str]]]:
    seq_len = int(cfg.data.seq_length)
    asset_ranges: Dict[str, List[Tuple[int, int]]] = {}
    bucket_symbols: Dict[str, List[str]] = {"us": [], "cn": [], "crypto": [], "other": []}
    tier_symbols: Dict[str, List[str]] = {"tier1": [], "tier2": [], "tier3": []}

    for sym, data in symbol_store.items():
        ranges: List[Tuple[int, int]] = []
        for seg in data.segment_indices:
            if mode == "train":
                s, e = seg.train_start, seg.train_end
            elif mode == "val":
                s, e = seg.val_start, seg.val_end
            else:
                s, e = seg.test_start, seg.test_end
            overlap = _effective_overlap_lookback(seq_len, cfg.data.overlap_lookback, mode)
            if e - s > _min_split_width(seq_len, max_steps, overlap):
                ranges.append((_window_start_index(s, overlap), e))
        asset_ranges[sym] = ranges
        if ranges:
            bucket_symbols[_sampling_market_bucket(sym, data.market, data.asset_class)].append(sym)
            t = data.tier if data.tier in tier_symbols else "tier3"
            tier_symbols[t].append(sym)

    return asset_ranges, bucket_symbols, tier_symbols


def _allocate_bucket_counts(
    num_envs: int,
    available_counts: Dict[str, int],
    raw_weights: Dict[str, float],
) -> Dict[str, int]:
    weights = _normalize_market_bucket_ratios(raw_weights)
    available = [bucket for bucket, count in available_counts.items() if count > 0]
    if not available:
        return {}

    filtered = {bucket: weights[bucket] for bucket in available}
    total = sum(filtered.values())
    filtered = {bucket: weight / total for bucket, weight in filtered.items()}

    raw_counts = {bucket: filtered[bucket] * num_envs for bucket in filtered}
    counts = {bucket: int(raw_counts[bucket]) for bucket in filtered}
    remaining = num_envs - sum(counts.values())
    order = sorted(filtered.keys(), key=lambda bucket: raw_counts[bucket] - counts[bucket], reverse=True)
    for idx in range(remaining):
        counts[order[idx % len(order)]] += 1

    overflow = 0
    for bucket, count in list(counts.items()):
        capacity = int(available_counts.get(bucket, 0))
        if count > capacity:
            overflow += count - capacity
            counts[bucket] = capacity

    while overflow > 0:
        expandable = [
            bucket for bucket in counts if int(available_counts.get(bucket, 0)) > int(counts.get(bucket, 0))
        ]
        if not expandable:
            break
        expandable.sort(
            key=lambda bucket: (
                int(available_counts[bucket]) - int(counts[bucket]),
                filtered.get(bucket, 0.0),
            ),
            reverse=True,
        )
        for bucket in expandable:
            if overflow <= 0:
                break
            if int(available_counts[bucket]) <= int(counts[bucket]):
                continue
            counts[bucket] += 1
            overflow -= 1

    return counts


def _episode_activity_score(prices: np.ndarray, current_idx: int, max_steps: int) -> float:
    start = max(int(current_idx) - 1, 0)
    end = min(int(current_idx) + int(max_steps), len(prices) - 1)
    if end - start < 2:
        return 0.0

    window = prices[start:end + 1].astype(np.float64, copy=False)
    rets = np.diff(window) / np.clip(window[:-1], 1e-8, None)
    if rets.size == 0:
        return 0.0

    mean_abs = float(np.mean(np.abs(rets)))
    start_px = max(abs(float(window[0])), 1e-8)
    total_move = abs(float(window[-1]) - float(window[0])) / start_px
    full_range = (float(window.max()) - float(window.min())) / start_px
    return float(0.5 * mean_abs + 0.3 * full_range + 0.2 * total_move)


def _select_manifest_bucket_episodes(
    candidates: List[EvalEpisode],
    count: int,
    rng: random.Random,
) -> List[EvalEpisode]:
    if count <= 0 or not candidates:
        return []

    ordered = sorted(candidates, key=lambda ep: ep.score, reverse=True)
    pool = ordered[:max(count * 4, count)]
    if len(pool) <= count:
        selected = pool[:count]
        rng.shuffle(selected)
        return selected

    selected: List[EvalEpisode] = []
    used_indices = set()
    step = (len(pool) - 1) / max(count - 1, 1)
    for idx in range(count):
        pick = int(round(idx * step))
        while pick in used_indices and pick + 1 < len(pool):
            pick += 1
        if pick in used_indices:
            pick = max(i for i in range(len(pool)) if i not in used_indices)
        used_indices.add(pick)
        selected.append(pool[pick])

    rng.shuffle(selected)
    return selected


def _default_val_manifest_path(cfg: Config, num_envs: int, max_steps: int) -> Path:
    raw_path = str(getattr(cfg.ppo, "val_fixed_manifest_path", "") or "").strip()
    if raw_path:
        return Path(raw_path)
    return Path(cfg.run.output_dir) / f"val_fixed_manifest_{num_envs}x{max_steps}.json"


def _load_eval_manifest(
    path: Path,
    cfg: Config,
    symbol_store: Dict[str, SymbolData],
    num_envs: int,
    max_steps: int,
) -> Optional[List[EvalEpisode]]:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[warn] failed to load val manifest {path}: {exc}")
        return None

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    episodes_raw = payload.get("episodes", []) if isinstance(payload, dict) else []
    if not isinstance(meta, dict) or not isinstance(episodes_raw, list):
        return None

    if int(meta.get("num_envs", -1)) != int(num_envs):
        return None
    if int(meta.get("max_steps", -1)) != int(max_steps):
        return None
    if int(meta.get("seq_length", -1)) != int(cfg.data.seq_length):
        return None
    if str(meta.get("data_file", "")) != str(cfg.data.data_file):
        return None

    episodes: List[EvalEpisode] = []
    for item in episodes_raw:
        if not isinstance(item, dict):
            return None
        sym = str(item.get("symbol", ""))
        if sym not in symbol_store:
            return None
        episodes.append(
            EvalEpisode(
                symbol=sym,
                start_idx=int(item.get("start_idx", 0)),
                bucket=str(item.get("bucket", "other")),
                market_group=str(item.get("market_group", _market_group_from_metadata(sym))),
                score=float(item.get("score", 0.0)),
            )
        )

    return episodes if len(episodes) == int(num_envs) else None


def _write_eval_manifest(
    path: Path,
    cfg: Config,
    num_envs: int,
    max_steps: int,
    episodes: List[EvalEpisode],
) -> None:
    payload = {
        "meta": {
            "data_file": str(cfg.data.data_file),
            "num_envs": int(num_envs),
            "max_steps": int(max_steps),
            "seq_length": int(cfg.data.seq_length),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "episodes": [
            {
                "symbol": ep.symbol,
                "start_idx": int(ep.start_idx),
                "bucket": ep.bucket,
                "market_group": ep.market_group,
                "score": float(ep.score),
            }
            for ep in episodes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_fixed_val_manifest(
    symbol_store: Dict[str, SymbolData],
    cfg: Config,
    num_envs: int,
    max_steps: int,
    fixed_seed: int,
) -> List[EvalEpisode]:
    asset_ranges, _, _ = _build_asset_ranges(symbol_store, cfg, mode="val", max_steps=max_steps)
    stride = int(getattr(cfg.ppo, "val_fixed_manifest_stride", 0) or max(max_steps // 4, 20))
    rng = random.Random(int(fixed_seed))

    candidates_by_bucket: Dict[str, List[EvalEpisode]] = {"us": [], "cn": [], "crypto": [], "other": []}
    for sym, data in symbol_store.items():
        ranges = asset_ranges.get(sym) or []
        if not ranges:
            continue
        bucket = _sampling_market_bucket(sym, data.market, data.asset_class)
        market_group = _market_group_from_metadata(sym, data.market, data.asset_class)
        best_episode: Optional[EvalEpisode] = None
        for r_start, r_end in ranges:
            min_start = int(r_start)
            max_start = int(r_end) - int(cfg.data.seq_length) - int(max_steps) - 10
            if max_start <= min_start:
                continue
            start_points = list(range(min_start, max_start, max(stride, 1)))
            tail_start = max_start - 1
            if not start_points or start_points[-1] != tail_start:
                start_points.append(tail_start)
            for start_idx in start_points:
                score = _episode_activity_score(data.prices, start_idx + int(cfg.data.seq_length), max_steps)
                if best_episode is None or score > best_episode.score:
                    best_episode = EvalEpisode(
                        symbol=sym,
                        start_idx=int(start_idx),
                        bucket=bucket,
                        market_group=market_group,
                        score=float(score),
                    )
        if best_episode is not None:
            candidates_by_bucket[bucket].append(best_episode)

    counts = _allocate_bucket_counts(
        num_envs=num_envs,
        available_counts={bucket: len(cands) for bucket, cands in candidates_by_bucket.items()},
        raw_weights=dict(getattr(cfg.ppo, "market_bucket_ratios", {})),
    )

    episodes: List[EvalEpisode] = []
    for bucket, count in counts.items():
        episodes.extend(_select_manifest_bucket_episodes(candidates_by_bucket.get(bucket, []), count, rng))

    if len(episodes) < num_envs:
        used = {(ep.symbol, ep.start_idx) for ep in episodes}
        leftovers: List[EvalEpisode] = []
        for bucket_candidates in candidates_by_bucket.values():
            for ep in sorted(bucket_candidates, key=lambda item: item.score, reverse=True):
                key = (ep.symbol, ep.start_idx)
                if key in used:
                    continue
                leftovers.append(ep)
                used.add(key)
        episodes.extend(leftovers[: max(num_envs - len(episodes), 0)])

    if len(episodes) < num_envs:
        raise RuntimeError(
            f"Unable to build val fixed manifest: need {num_envs} episodes, got {len(episodes)}"
        )

    episodes = episodes[:num_envs]
    rng.shuffle(episodes)
    return episodes


def build_symbol_store(cfg: Config) -> Dict[str, SymbolData]:
    data_file = Path(cfg.data.data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"data_file not found: {data_file}")

    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    optional_cols: List[str] = []
    try:
        import pyarrow.parquet as pq
        available_cols = set(pq.ParquetFile(data_file).schema.names)
        # 兼容 timestamp 列名（crypto数据用timestamp，其他数据用date）
        if "date" not in available_cols and "timestamp" in available_cols:
            cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        optional_cols = [
            c
            for c in (
                "market",
                "asset_class",
                "tier",
                "corp_action_cash_dividend_per10",
                "corp_action_share_multiplier",
                "corp_action_rights_ratio_per10",
                "corp_action_rights_event",
            )
            if c in available_cols
        ]
    except Exception:
        optional_cols = []
    df = pd.read_parquet(data_file, columns=cols + optional_cols)
    # 统一列名为 date
    if "timestamp" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"timestamp": "date"})

    if cfg.data.symbols:
        symbol_set = set(cfg.data.symbols)
        df = df[df["symbol"].isin(symbol_set)]

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    normalizer = OHLCVNormalizer(cfg.data.normalize_method)

    symbol_store: Dict[str, SymbolData] = {}
    skipped = 0
    for symbol, group in df.groupby("symbol", sort=False):
        g = group.reset_index(drop=True)
        n = len(g)
        if n < cfg.data.min_bars:
            skipped += 1
            continue

        train_end = int(n * cfg.data.train_ratio)
        val_end = int(n * (cfg.data.train_ratio + cfg.data.val_ratio))
        train_end = min(max(train_end, 0), n)
        val_end = min(max(val_end, train_end), n)
        test_ratio = max(1.0 - float(cfg.data.train_ratio) - float(cfg.data.val_ratio), 0.0)

        max_steps = max(cfg.ppo.trajectory_length, cfg.ppo.val_trajectory_length)
        eval_overlap = _effective_overlap_lookback(
            cfg.data.seq_length,
            cfg.data.overlap_lookback,
            "val",
        )
        train_min_need = _min_split_width(cfg.data.seq_length, max_steps, 0)
        eval_min_need = _min_split_width(cfg.data.seq_length, max_steps, eval_overlap)

        if train_end <= train_min_need:
            skipped += 1
            continue
        if cfg.data.val_ratio > 0 and (val_end - train_end) <= eval_min_need:
            skipped += 1
            continue
        if test_ratio > 0 and (n - val_end) <= eval_min_need:
            skipped += 1
            continue

        ohlcv = g[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        amount = (ohlcv[:, 3] * ohlcv[:, 4]).astype(np.float32)
        raw_ohlcva = np.concatenate([ohlcv, amount[:, None]], axis=1)

        dt = pd.to_datetime(g["date"])
        time_features = np.stack(
            [
                dt.dt.minute.values.astype(np.float32),
                dt.dt.hour.values.astype(np.float32),
                dt.dt.weekday.values.astype(np.float32),
                dt.dt.day.values.astype(np.float32),
                dt.dt.month.values.astype(np.float32),
            ],
            axis=1,
        )

        features = normalizer.normalize(ohlcv)
        prices = g["close"].astype(np.float32).values
        market = str(g["market"].iloc[0]) if "market" in g.columns else ""
        asset_class = str(g["asset_class"].iloc[0]) if "asset_class" in g.columns else ""
        tier = str(g["tier"].iloc[0]) if "tier" in g.columns else "tier3"
        corp_action_cash_dividend_per10 = (
            pd.to_numeric(g["corp_action_cash_dividend_per10"], errors="coerce").fillna(0.0).astype(np.float32).values
            if "corp_action_cash_dividend_per10" in g.columns
            else np.zeros(n, dtype=np.float32)
        )
        corp_action_share_multiplier = (
            pd.to_numeric(g["corp_action_share_multiplier"], errors="coerce").fillna(1.0).astype(np.float32).values
            if "corp_action_share_multiplier" in g.columns
            else np.ones(n, dtype=np.float32)
        )
        corp_action_rights_ratio_per10 = (
            pd.to_numeric(g["corp_action_rights_ratio_per10"], errors="coerce").fillna(0.0).astype(np.float32).values
            if "corp_action_rights_ratio_per10" in g.columns
            else np.zeros(n, dtype=np.float32)
        )
        corp_action_rights_event = (
            g["corp_action_rights_event"].eq(True).values
            if "corp_action_rights_event" in g.columns
            else np.zeros(n, dtype=bool)
        )

        seg = Segment(
            train_start=0,
            train_end=train_end,
            val_start=train_end,
            val_end=val_end,
            test_start=val_end,
            test_end=n,
        )

        symbol_store[symbol] = SymbolData(
            symbol=symbol,
            features=features,
            raw_ohlcva=raw_ohlcva,
            time_features=time_features,
            prices=prices,
            segment_indices=[seg],
            market=market,
            asset_class=asset_class,
            tier=tier,
            corp_action_cash_dividend_per10=corp_action_cash_dividend_per10,
            corp_action_share_multiplier=corp_action_share_multiplier,
            corp_action_rights_ratio_per10=corp_action_rights_ratio_per10,
            corp_action_rights_event=corp_action_rights_event,
        )

    if not symbol_store:
        raise RuntimeError("No valid symbols after filtering. Check data/min_bars/split ratios.")

    log(
        f"Loaded symbols={len(symbol_store)}, skipped={skipped}, "
        f"bars_total={sum(len(v.prices) for v in symbol_store.values()):,}"
    )
    return symbol_store


# -----------------------------
# Trading environment
# -----------------------------


@dataclass
class TradeResult:
    direction: int = 0
    asset_id: int = 0
    pnl: float = 0.0
    holding_steps: int = 0
    empty_steps: int = 0
    market_group: str = ""
    exit_reason: str = ""
    stop_source: str = ""
    tier: str = "tier3"
    stop_loss_value: float = 0.0  # 开仓时的ATR止损比例（负数，如-0.03表示3%）


class TradingEnv:
    def __init__(
        self,
        symbol: str,
        features: np.ndarray,
        raw_ohlcva: np.ndarray,
        time_features: np.ndarray,
        prices: np.ndarray,
        asset_id: int,
        market_group: str,
        price_basis: str,
        corp_action_cash_dividend_per10: Optional[np.ndarray],
        corp_action_share_multiplier: Optional[np.ndarray],
        corp_action_rights_ratio_per10: Optional[np.ndarray],
        corp_action_rights_event: Optional[np.ndarray],
        cfg: EnvConfig,
        seq_length: int,
        start_idx: int,
        max_steps: int,
        tier: str = "tier3",
    ):
        self.symbol = symbol
        self.features = features
        self.raw_ohlcva = raw_ohlcva
        self.time_features = time_features
        self.prices = prices
        self.asset_id = asset_id
        self.market_group = market_group
        self.tier = tier
        self.price_basis = str(price_basis or "adjusted").lower()
        self.cfg = cfg
        self.seq_len = seq_length
        self.max_steps = max_steps
        self.total_bars = len(features)
        self.corp_action_cash_dividend_per10 = (
            corp_action_cash_dividend_per10.astype(np.float32, copy=False)
            if corp_action_cash_dividend_per10 is not None
            else np.zeros(self.total_bars, dtype=np.float32)
        )
        self.corp_action_share_multiplier = (
            corp_action_share_multiplier.astype(np.float32, copy=False)
            if corp_action_share_multiplier is not None
            else np.ones(self.total_bars, dtype=np.float32)
        )
        self.corp_action_rights_ratio_per10 = (
            corp_action_rights_ratio_per10.astype(np.float32, copy=False)
            if corp_action_rights_ratio_per10 is not None
            else np.zeros(self.total_bars, dtype=np.float32)
        )
        self.corp_action_rights_event = (
            corp_action_rights_event.astype(bool, copy=False)
            if corp_action_rights_event is not None
            else np.zeros(self.total_bars, dtype=bool)
        )

        self.step_count = 0
        self.position_state = States.EMPTY
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.current_idx = start_idx + seq_length
        self.long_only_market_groups = {
            str(group).strip().upper()
            for group in getattr(cfg, "long_only_market_groups", [])
            if str(group).strip()
        }

        self.empty_steps = 0
        self.holding_steps = 0
        self._entry_empty_steps = 0
        self._entry_stop_loss_value = 0.0

        self.peak_return = 0.0
        self.peak_step = 0
        self.last_unrealized = 0.0

        self.completed_trades: List[TradeResult] = []
        self.position_units = 0.0
        self.position_cash = 0.0
        self.cash_adjustments = 0.0
        self.entry_notional = 0.0
        self.corp_action_idx_applied = self.current_idx

    def get_market_features(self) -> np.ndarray:
        s = max(0, self.current_idx - self.seq_len)
        e = self.current_idx
        return self.features[s:e].copy()

    def get_teacher_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = max(0, self.current_idx - self.seq_len)
        e = self.current_idx
        x = self.raw_ohlcva[s:e].copy()
        x_stamp = self.time_features[s:e].copy()
        y_idx = min(e, self.total_bars - 1)
        y_stamp = self.time_features[y_idx:y_idx + 1].copy()
        return x, x_stamp, y_stamp

    def get_teacher_key(self) -> Tuple[str, int]:
        return self.symbol, int(self.current_idx)

    def get_position_info(self) -> np.ndarray:
        info = np.zeros(6, dtype=np.float32)
        info[0] = float(self.position)
        if self.position_state == States.EMPTY:
            info[4] = min(self.empty_steps / max(self.cfg.max_empty_steps, 1), 1.0)
            return info

        unrealized = float(self.last_unrealized)
        info[1] = np.tanh(unrealized * 20.0)
        info[2] = np.tanh(self.peak_return * 20.0)
        info[3] = np.tanh(max(self.peak_return - unrealized, 0.0) * 20.0)
        info[4] = min(self.holding_steps / max(self.cfg.max_holding_steps, 1), 1.0)
        steps_since_peak = max(self.step_count - self.peak_step, 0)
        info[5] = min(steps_since_peak / max(self.cfg.max_holding_steps, 1), 1.0)
        return info

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(2, dtype=np.float32)
        if self.position_state == States.EMPTY:
            mask[int(FlatActions.WAIT)] = 1.0
            mask[int(FlatActions.OPEN)] = 1.0 if self.empty_steps >= self.cfg.min_empty_steps else 0.0
            return mask

        if self.holding_steps < self.cfg.min_holding_steps:
            mask[int(HoldActions.CLOSE)] = 0.0
        if self.holding_steps >= self.cfg.max_holding_steps:
            mask[int(HoldActions.HOLD)] = 0.0
            mask[int(HoldActions.CLOSE)] = 1.0
        return mask

    def _uses_raw_price_accounting(self) -> bool:
        return self.price_basis == "raw"

    def _reset_position_accounting(self) -> None:
        self.position_units = 0.0
        self.position_cash = 0.0
        self.cash_adjustments = 0.0
        self.entry_notional = 0.0
        self.corp_action_idx_applied = self.current_idx

    def _initialize_position_accounting(self) -> None:
        self.position_units = 1.0
        self.entry_notional = max(float(self.entry_price), 1e-8)
        self.position_cash = self.entry_notional if self.position_state == States.SHORT else 0.0
        self.cash_adjustments = 0.0
        self.corp_action_idx_applied = self.current_idx

    def _apply_pending_corporate_actions(self) -> None:
        if not self._uses_raw_price_accounting() or self.position_state == States.EMPTY:
            return
        if self.current_idx <= self.corp_action_idx_applied:
            return

        end_idx = min(self.current_idx, self.total_bars - 1)
        for idx in range(self.corp_action_idx_applied + 1, end_idx + 1):
            units_before = self.position_units
            cash_dividend_per_share = float(self.corp_action_cash_dividend_per10[idx]) / 10.0
            if cash_dividend_per_share != 0.0 and units_before > 0.0:
                sign = 1.0 if self.position_state == States.LONG else -1.0
                self.cash_adjustments += sign * units_before * cash_dividend_per_share

            share_multiplier = float(self.corp_action_share_multiplier[idx])
            if np.isfinite(share_multiplier) and share_multiplier > 0.0 and abs(share_multiplier - 1.0) > 1e-8:
                self.position_units *= share_multiplier

        self.corp_action_idx_applied = end_idx

    def _raw_trade_return_at_price(self, price: float) -> float:
        if self.position_state == States.EMPTY:
            return 0.0
        if not self._uses_raw_price_accounting():
            if self.position_state == States.LONG:
                return (float(price) - self.entry_price) / max(self.entry_price, 1e-8)
            return (self.entry_price - float(price)) / max(self.entry_price, 1e-8)

        self._apply_pending_corporate_actions()
        cur_price = float(price)
        if self.position_state == States.LONG:
            equity = self.cash_adjustments + self.position_units * cur_price
            return (equity - self.entry_notional) / max(self.entry_notional, 1e-8)

        equity = self.position_cash + self.cash_adjustments - self.position_units * cur_price
        return equity / max(self.entry_notional, 1e-8)

    def _scaled_trade_return_at_price(self, price: float) -> float:
        return abs(float(self.position)) * self._raw_trade_return_at_price(price)

    def _trade_cost(self) -> float:
        return float(self.cfg.transaction_cost) * abs(float(self.position))

    def _round_trip_trade_cost(self) -> float:
        return 2.0 * self._trade_cost()

    def _bar_ohlc(self, bar_idx: int) -> Tuple[float, float, float, float]:
        row = self.raw_ohlcva[bar_idx]
        return float(row[0]), float(row[1]), float(row[2]), float(row[3])

    def _recent_volatility(self) -> float:
        window = max(int(getattr(self.cfg, "reward_vol_window", 20)), 5)
        floor = max(float(getattr(self.cfg, "reward_vol_floor", 0.008)), 1e-4)
        end = max(min(self.current_idx, self.total_bars - 1), 1)
        start = max(1, end - window)
        price_slice = self.prices[start - 1:end]
        if len(price_slice) <= 1:
            return floor
        returns = np.diff(price_slice) / (price_slice[:-1] + 1e-8)
        if returns.size == 0:
            return floor
        return max(float(np.std(returns)), floor)

    def _atr(self, window: int = 14) -> float:
        """计算ATR（Average True Range），以当前价格归一化为比例"""
        end = max(min(self.current_idx, self.total_bars - 1), 1)
        start = max(1, end - window)
        # raw_ohlcva: [open, high, low, close, volume, amount]
        ohlcv = self.raw_ohlcva[start - 1:end + 1]
        if len(ohlcv) < 2:
            return max(float(getattr(self.cfg, "reward_vol_floor", 0.008)), 1e-4)
        highs = ohlcv[1:, 1]
        lows = ohlcv[1:, 2]
        prev_closes = ohlcv[:-1, 3]
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
        atr = float(np.mean(tr))
        cur_price = float(self.prices[end]) if end < len(self.prices) else float(self.prices[-1])
        if cur_price <= 0:
            return max(float(getattr(self.cfg, "reward_vol_floor", 0.008)), 1e-4)
        return max(atr / cur_price, max(float(getattr(self.cfg, "reward_vol_floor", 0.008)), 1e-4))

    def _effective_stop_loss(self) -> float:
        static_stop = float(self.cfg.stop_loss)
        vol_multiple = max(float(getattr(self.cfg, "stop_loss_vol_multiple", 0.0)), 0.0)
        if vol_multiple <= 0.0:
            return static_stop
        atr_stop = -vol_multiple * self._atr()
        if static_stop < 0.0:
            return max(static_stop, atr_stop)
        return atr_stop

    def _stop_loss_details(self) -> Tuple[float, float, float, str]:
        static_stop = float(self.cfg.stop_loss)
        vol_multiple = max(float(getattr(self.cfg, "stop_loss_vol_multiple", 0.0)), 0.0)
        if vol_multiple <= 0.0:
            return static_stop, static_stop, static_stop, "fixed"

        atr_stop = -vol_multiple * self._atr()
        if static_stop < 0.0:
            effective = max(static_stop, atr_stop)
            source = "fixed" if effective <= static_stop + 1e-8 else "atr"
            return static_stop, atr_stop, effective, source

        return static_stop, atr_stop, atr_stop, "atr"

    def _entry_signal_to_position(self, entry_signal: float) -> float:
        signal = float(np.clip(entry_signal, -1.0, 1.0))
        if str(self.market_group).upper() in self.long_only_market_groups:
            signal = max(signal, 0.0)
        return signal * max(float(getattr(self.cfg, "position_scale", 1.0)), 0.0)

    def _parse_flat_action(self, action) -> Tuple[int, float]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return int(FlatActions.WAIT), 0.0

        if arr.size == 1:
            legacy_signal = float(np.clip(arr[0], -1.0, 1.0))
            if not np.isfinite(legacy_signal) or abs(legacy_signal) < float(self.cfg.entry_threshold):
                return int(FlatActions.WAIT), 0.0
            return int(FlatActions.OPEN), legacy_signal

        gate_action = int(
            np.clip(int(round(float(arr[0]))), int(FlatActions.WAIT), int(FlatActions.OPEN))
        )
        size_action = float(np.clip(arr[1], -1.0, 1.0)) if np.isfinite(arr[1]) else 0.0
        if gate_action != int(FlatActions.OPEN):
            return int(FlatActions.WAIT), 0.0
        return gate_action, size_action

    def _parse_holding_action(self, action) -> int:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_action = float(arr[0]) if arr.size else float(HoldActions.HOLD)
        if not np.isfinite(raw_action):
            raw_action = float(HoldActions.HOLD)
        return int(np.clip(int(round(raw_action)), int(HoldActions.CLOSE), int(HoldActions.HOLD)))

    def _hold_reward_scale(self) -> float:
        vol = self._recent_volatility()
        denom = max(float(getattr(self.cfg, "reward_hold_norm_denom", 1.0)), 1e-4)
        floor = max(float(getattr(self.cfg, "reward_vol_floor", 0.008)), 1e-4)
        pos_scale = max(abs(float(self.position)), 1e-4)
        return max(vol * denom * pos_scale, floor)

    def _hold_potential(self, cur_ret: float, peak_ret: float) -> float:
        scale = self._hold_reward_scale()
        profit_term = math.tanh(float(cur_ret) / scale)
        drawdown = max(float(peak_ret) - float(cur_ret), 0.0)
        drawdown_term = math.tanh(drawdown / scale)
        dd_weight = max(float(getattr(self.cfg, "potential_drawdown_weight", 0.60)), 0.0)
        return float(profit_term - dd_weight * drawdown_term)

    def _holding_reward(self, cur_ret: float) -> float:
        prev_ret = float(self.last_unrealized)
        prev_peak = float(self.peak_return)
        prev_phi = self._hold_potential(prev_ret, prev_peak)

        if cur_ret > self.peak_return:
            self.peak_return = cur_ret
            self.peak_step = self.step_count

        cur_phi = self._hold_potential(cur_ret, self.peak_return)
        self.last_unrealized = cur_ret

        potential_scale = max(float(getattr(self.cfg, "potential_scale", 0.0)), 0.0)
        return potential_scale * (cur_phi - prev_phi)

    def _empty_wait_reward(self) -> float:
        voluntary_wait_step = max(int(self.empty_steps) - int(self.cfg.min_empty_steps), 0)
        if voluntary_wait_step <= 0:
            return 0.0

        reward_steps = max(int(getattr(self.cfg, "empty_reward_steps", 5)), 0)
        early_wait_reward = float(getattr(self.cfg, "empty_first_step_bonus", 0.0))
        late_penalty_start = max(int(getattr(self.cfg, "empty_penalty_start", 30)), reward_steps)
        late_penalty = float(getattr(self.cfg, "empty_late_penalty", 0.0))

        if voluntary_wait_step <= reward_steps:
            return float(early_wait_reward)
        if voluntary_wait_step > late_penalty_start:
            return float(-late_penalty)
        return 0.0

    def _mark_peak(self, cur_ret: float) -> None:
        if cur_ret > self.peak_return:
            self.peak_return = cur_ret
            self.peak_step = self.step_count

    def _close_trade_at_price(self, price: float, exit_reason: str = "", stop_source: str = "") -> float:
        scaled_ret = self._scaled_trade_return_at_price(price)
        close_reward = scaled_ret
        full_pnl = scaled_ret - self._round_trip_trade_cost()
        direction = 1 if self.position > 0.0 else -1

        self.completed_trades.append(
            TradeResult(
                direction=direction,
                asset_id=self.asset_id,
                pnl=float(full_pnl),
                holding_steps=int(self.holding_steps),
                empty_steps=int(self._entry_empty_steps),
                market_group=self.market_group,
                exit_reason=str(exit_reason or ""),
                stop_source=str(stop_source or ""),
                tier=self.tier,
                stop_loss_value=float(self._entry_stop_loss_value),
            )
        )

        self.position_state = States.EMPTY
        self.position = 0.0
        self.entry_price = 0.0
        self.holding_steps = 0
        self.empty_steps = 0
        self.last_unrealized = 0.0
        self.peak_return = 0.0
        self._reset_position_accounting()
        return float(close_reward)

    def _stop_or_takeprofit_price(self, bar_idx: int) -> Tuple[Optional[float], str, str]:
        if self.position_state == States.EMPTY:
            return None, "", ""

        bar_open, bar_high, bar_low, _ = self._bar_ohlc(bar_idx)
        _, _, stop_loss_threshold, stop_source = self._stop_loss_details()
        take_profit = float(self.cfg.take_profit)

        if self.position_state == States.LONG:
            if stop_loss_threshold < 0.0:
                stop_px = self.entry_price * (1.0 + stop_loss_threshold)
                if bar_open <= stop_px:
                    return bar_open, "stop_loss", stop_source
                if bar_low <= stop_px:
                    return stop_px, "stop_loss", stop_source
            if take_profit > 0.0:
                take_px = self.entry_price * (1.0 + take_profit)
                if bar_open >= take_px:
                    return bar_open, "take_profit", ""
                if bar_high >= take_px:
                    return take_px, "take_profit", ""
            return None, "", ""

        if stop_loss_threshold < 0.0:
            stop_px = self.entry_price * (1.0 - stop_loss_threshold)
            if bar_open >= stop_px:
                return bar_open, "stop_loss", stop_source
            if bar_high >= stop_px:
                return stop_px, "stop_loss", stop_source
        if take_profit > 0.0:
            take_px = self.entry_price * (1.0 - take_profit)
            if bar_open <= take_px:
                return bar_open, "take_profit", ""
            if bar_low <= take_px:
                return take_px, "take_profit", ""
        return None, "", ""

    def _process_holding_bar(self, bar_idx: int, force_close_end: bool) -> Tuple[float, float, bool]:
        self.holding_steps += 1
        exit_price, exit_reason, stop_source = self._stop_or_takeprofit_price(bar_idx)
        if exit_price is not None:
            close_reward = self._close_trade_at_price(exit_price, exit_reason=exit_reason, stop_source=stop_source)
            return 0.0, close_reward, True

        _, _, _, bar_close = self._bar_ohlc(bar_idx)
        cur_ret = self._scaled_trade_return_at_price(bar_close)
        hold_reward = self._holding_reward(cur_ret)
        if force_close_end:
            close_reward = self._close_trade_at_price(bar_close, exit_reason="force_close", stop_source="")
            return float(hold_reward), close_reward, True
        return float(hold_reward), 0.0, False

    def _step_empty(self, action, bar_idx: int, force_close_end: bool) -> Tuple[float, float, float, float, bool]:
        gate_action, entry_signal = self._parse_flat_action(action)
        if self.empty_steps < self.cfg.min_empty_steps:
            self.empty_steps += 1
            return 0.0, 0.0, 0.0, 0.0, False

        if gate_action != int(FlatActions.OPEN):
            self.empty_steps += 1
            wait_reward = self._empty_wait_reward()
            return float(wait_reward), 0.0, 0.0, 0.0, False

        position = self._entry_signal_to_position(entry_signal)
        if position == 0.0:
            self.empty_steps += 1
            wait_reward = self._empty_wait_reward()
            return float(wait_reward), 0.0, 0.0, 0.0, False

        self.position = position
        self.position_state = States.LONG if position > 0.0 else States.SHORT
        bar_open, _, _, _ = self._bar_ohlc(bar_idx)
        self.entry_price = bar_open
        self.entry_step = self.step_count
        self.holding_steps = 0
        self._entry_empty_steps = self.empty_steps
        self.empty_steps = 0
        self.peak_return = 0.0
        self.peak_step = self.step_count
        self.last_unrealized = 0.0
        self._entry_stop_loss_value = self._effective_stop_loss()  # 记录开仓时的止损值
        self._initialize_position_accounting()

        open_reward = -self._round_trip_trade_cost()
        hold_reward, close_reward, trade_end = self._process_holding_bar(bar_idx, force_close_end)
        reward = open_reward + hold_reward + close_reward
        return float(reward), float(open_reward), float(hold_reward), float(close_reward), bool(trade_end)

    def _step_holding(self, action, bar_idx: int, force_close_end: bool) -> Tuple[float, float, float, bool]:
        close_action = self._parse_holding_action(action)
        if self.holding_steps >= self.cfg.max_holding_steps:
            close_action = int(HoldActions.CLOSE)
        elif self.holding_steps < self.cfg.min_holding_steps:
            close_action = int(HoldActions.HOLD)

        if close_action == int(HoldActions.CLOSE):
            bar_open, _, _, _ = self._bar_ohlc(bar_idx)
            close_reward = self._close_trade_at_price(bar_open, exit_reason="close_signal", stop_source="")
            return float(close_reward), 0.0, float(close_reward), True

        hold_reward, close_reward, trade_end = self._process_holding_bar(bar_idx, force_close_end)
        reward = hold_reward + close_reward
        return float(reward), float(hold_reward), float(close_reward), bool(trade_end)

    def step(self, action, log_prob: float = 0.0):
        reward = 0.0
        trade_end = False
        open_r = hold_r = close_r = 0.0
        bar_idx = self.current_idx
        force_close_end = (self.step_count + 1) >= self.max_steps or bar_idx >= (self.total_bars - 1)

        if self.position_state == States.EMPTY:
            reward, open_r, hold_r, close_r, trade_end = self._step_empty(action, bar_idx, force_close_end)
        else:
            reward, hold_r, close_r, trade_end = self._step_holding(action, bar_idx, force_close_end)

        self.step_count += 1
        self.current_idx += 1
        done = self.step_count >= self.max_steps or self.current_idx >= self.total_bars
        return float(reward), bool(done), bool(trade_end), float(open_r), float(hold_r), float(close_r)

    def is_done(self) -> bool:
        return self.step_count >= self.max_steps or self.current_idx >= self.total_bars


class BatchTradingEnv:
    def __init__(
        self,
        symbol_store: Dict[str, SymbolData],
        cfg: Config,
        mode: str,
        num_envs: int,
        max_steps: int,
        fixed_episodes: Optional[List[EvalEpisode]] = None,
    ):
        self.symbol_store = symbol_store
        self.cfg = cfg
        self.mode = mode
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.seq_len = cfg.data.seq_length

        self.symbols = list(symbol_store.keys())
        self.symbol_to_asset_id = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.n_assets = len(self.symbols)

        self.asset_ranges, self.bucket_symbols, self.tier_symbols = _build_asset_ranges(
            symbol_store=self.symbol_store,
            cfg=self.cfg,
            mode=self.mode,
            max_steps=self.max_steps,
        )
        self.non_empty_symbols = [sym for sym, ranges in self.asset_ranges.items() if ranges]
        self.fixed_episodes = list(fixed_episodes) if fixed_episodes else None
        if self.fixed_episodes is not None and len(self.fixed_episodes) != self.num_envs:
            raise ValueError(
                f"fixed_episodes length mismatch: expected {self.num_envs}, got {len(self.fixed_episodes)}"
            )

        if self.fixed_episodes is not None:
            self.env_bucket_targets = [ep.bucket for ep in self.fixed_episodes]
            self.env_tier_targets = [ep.bucket for ep in self.fixed_episodes]  # unused for fixed
            self.envs = [self._create_env_from_episode(ep) for ep in self.fixed_episodes]
        else:
            self.env_bucket_targets = self._build_env_bucket_targets()
            self.env_tier_targets = self._build_env_tier_targets()
            self.envs = [self._create_env(i, preferred_bucket=self.env_bucket_targets[i], preferred_tier=self.env_tier_targets[i]) for i in range(num_envs)]

    def _build_env_bucket_targets(self) -> List[Optional[str]]:
        if not bool(self.cfg.ppo.market_bucket_sampling):
            return [None for _ in range(self.num_envs)]

        counts = _allocate_bucket_counts(
            num_envs=self.num_envs,
            available_counts={bucket: len(syms) for bucket, syms in self.bucket_symbols.items()},
            raw_weights=dict(self.cfg.ppo.market_bucket_ratios),
        )
        if not counts:
            return [None for _ in range(self.num_envs)]

        targets: List[Optional[str]] = []
        for bucket, count in counts.items():
            targets.extend([bucket] * count)
        random.shuffle(targets)
        if len(targets) < self.num_envs:
            targets.extend([None] * (self.num_envs - len(targets)))
        return targets[:self.num_envs]

    def _build_env_tier_targets(self) -> List[Optional[str]]:
        """Tier采样: tier1/tier2/tier3 各采20，共60条轨迹"""
        t1 = self.tier_symbols.get("tier1", [])
        t2 = self.tier_symbols.get("tier2", [])
        t3 = self.tier_symbols.get("tier3", [])
        tier1_sample = min(20, len(t1))
        tier2_sample = min(20, len(t2))
        tier3_sample = min(20, len(t3))
        targets: List[Optional[str]] = (
            ["tier1"] * tier1_sample +
            ["tier2"] * tier2_sample +
            ["tier3"] * tier3_sample
        )
        random.shuffle(targets)
        # 不足 num_envs 时随机填充
        while len(targets) < self.num_envs:
            targets.append(random.choice(["tier1", "tier2", "tier3"]))
        return targets[:self.num_envs]

    def _choose_symbol(self, env_idx: int, preferred_bucket: Optional[str] = None, preferred_tier: Optional[str] = None) -> str:
        # tier采样优先
        if preferred_tier is not None and self.tier_symbols.get(preferred_tier):
            candidates = [s for s in self.tier_symbols[preferred_tier] if self.asset_ranges.get(s)]
            if candidates:
                return random.choice(candidates)
        if preferred_bucket is not None and self.bucket_symbols.get(preferred_bucket):
            return random.choice(self.bucket_symbols[preferred_bucket])
        if not self.non_empty_symbols:
            raise RuntimeError(f"No valid ranges for mode={self.mode}")
        base = self.symbols[env_idx % self.n_assets]
        return base if self.asset_ranges.get(base) else random.choice(self.non_empty_symbols)

    def _build_trading_env(self, sym: str, start_idx: int) -> TradingEnv:
        data = self.symbol_store[sym]
        return TradingEnv(
            symbol=sym,
            features=data.features,
            raw_ohlcva=data.raw_ohlcva,
            time_features=data.time_features,
            prices=data.prices,
            asset_id=self.symbol_to_asset_id[sym],
            market_group=_market_group_from_metadata(sym, data.market, data.asset_class),
            price_basis=self.cfg.data.price_basis,
            corp_action_cash_dividend_per10=data.corp_action_cash_dividend_per10,
            corp_action_share_multiplier=data.corp_action_share_multiplier,
            corp_action_rights_ratio_per10=data.corp_action_rights_ratio_per10,
            corp_action_rights_event=data.corp_action_rights_event,
            cfg=self.cfg.env,
            seq_length=self.seq_len,
            start_idx=int(start_idx),
            max_steps=self.max_steps,
            tier=data.tier,
        )

    def _create_env_from_episode(self, episode: EvalEpisode) -> TradingEnv:
        if episode.symbol not in self.symbol_store:
            raise KeyError(f"Unknown symbol in fixed val manifest: {episode.symbol}")
        return self._build_trading_env(episode.symbol, episode.start_idx)

    def _create_env(self, env_idx: int, preferred_bucket: Optional[str] = None, preferred_tier: Optional[str] = None) -> TradingEnv:
        sym = self._choose_symbol(env_idx, preferred_bucket, preferred_tier)
        ranges = self.asset_ranges[sym]

        r_start, r_end = random.choice(ranges)
        max_start = r_end - self.seq_len - self.max_steps - 10
        min_start = r_start
        if max_start <= min_start:
            max_start = min_start + 1
        start_idx = random.randint(min_start, max_start - 1)
        return self._build_trading_env(sym, start_idx)

    def get_batch_obs(self):
        market = np.stack([e.get_market_features() for e in self.envs]).astype(np.float32)
        teacher_keys = [e.get_teacher_key() for e in self.envs]
        teacher_symbols = [k[0] for k in teacher_keys]
        teacher_indices = np.asarray([k[1] for k in teacher_keys], dtype=np.int64)
        teacher_x, teacher_x_stamp, teacher_y_stamp = zip(*[e.get_teacher_inputs() for e in self.envs])
        pos_info = np.stack([e.get_position_info() for e in self.envs]).astype(np.float32)
        action_masks = np.stack([e.get_action_mask() for e in self.envs]).astype(np.float32)
        asset_ids = np.asarray([e.asset_id for e in self.envs], dtype=np.int64)
        return (
            market,
            teacher_symbols,
            teacher_indices,
            np.stack(teacher_x).astype(np.float32),
            np.stack(teacher_x_stamp).astype(np.float32),
            np.stack(teacher_y_stamp).astype(np.float32),
            pos_info,
            action_masks,
            asset_ids,
        )

    def get_batch_states(self) -> np.ndarray:
        return np.asarray([e.position_state for e in self.envs], dtype=np.int64)

    def step_batch(self, actions: np.ndarray, log_probs: np.ndarray):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        trade_ends = np.zeros(self.num_envs, dtype=np.bool_)
        ro = np.zeros(self.num_envs, dtype=np.float32)
        rh = np.zeros(self.num_envs, dtype=np.float32)
        rc = np.zeros(self.num_envs, dtype=np.float32)
        trades: List[TradeResult] = []

        for i, env in enumerate(self.envs):
            n_before = len(env.completed_trades)
            r, done, te, open_r, hold_r, close_r = env.step(actions[i], float(log_probs[i]))
            rewards[i] = r
            dones[i] = done
            trade_ends[i] = te
            ro[i] = open_r
            rh[i] = hold_r
            rc[i] = close_r

            trades.extend(env.completed_trades[n_before:])
            if done or env.is_done():
                if self.fixed_episodes is not None:
                    self.envs[i] = self._create_env_from_episode(self.fixed_episodes[i])
                else:
                    self.envs[i] = self._create_env(i, preferred_bucket=self.env_bucket_targets[i], preferred_tier=self.env_tier_targets[i])

        return rewards, dones, trade_ends, ro, rh, rc, trades


# -----------------------------
# Model
# -----------------------------


def _cuda_available_safe() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


if os.environ.get("BIGMODEL_SKIP_MAMBA_IMPORT", "0") == "1":
    HAS_MAMBA = False
else:
    try:
        from mamba_ssm import Mamba

        HAS_MAMBA = _cuda_available_safe()
    except Exception:
        HAS_MAMBA = False


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            self.has_mamba = True
        else:
            self.mamba = nn.GRU(d_model, d_model, batch_first=True)
            self.has_mamba = False
        self.dropout = (nn.Identity() if float(dropout) <= 0.0 else nn.Dropout(dropout))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        if self.has_mamba:
            x = self.mamba(x)
        else:
            x, _ = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_temporal_attention: bool = True,
        temporal_attn_hidden: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.use_temporal_attention = bool(use_temporal_attention)
        if self.use_temporal_attention:
            attn_hidden = max(int(temporal_attn_hidden), 16)
            self.temporal_attn = nn.Sequential(
                nn.Linear(hidden_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
            # 0.0 -> equal blend of last-step and attention pooling at init.
            self.temporal_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def _teacher_logits_from_strength(
    strength: torch.Tensor,
    position_states: torch.Tensor,
    action_mask: Optional[torch.Tensor],
    hold_bias: float,
) -> torch.Tensor:
    abs_strength = strength.abs()
    bsz = strength.shape[0]
    logits = torch.zeros((bsz, 2), device=strength.device, dtype=strength.dtype)

    empty = position_states == States.EMPTY
    long_holding = position_states == States.LONG
    short_holding = position_states == States.SHORT

    if empty.any():
        logits[empty, HoldActions.CLOSE] = -hold_bias * abs_strength[empty]
        logits[empty, HoldActions.HOLD] = strength[empty]

    if long_holding.any():
        logits[long_holding, HoldActions.CLOSE] = -strength[long_holding]
        logits[long_holding, HoldActions.HOLD] = strength[long_holding] + hold_bias

    if short_holding.any():
        logits[short_holding, HoldActions.CLOSE] = strength[short_holding]
        logits[short_holding, HoldActions.HOLD] = -strength[short_holding] + hold_bias

    if action_mask is not None and action_mask.shape[-1] == 2:
        logits = logits + (action_mask - 1.0) * 1e9
    return logits


class FrozenKronTeacher:
    def __init__(self, tokenizer, model, cfg: TeacherConfig, device: torch.device):
        self.tokenizer = tokenizer.to(device)
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.max_context = max(int(cfg.kron_max_context), 8)
        self.clip = float(cfg.kron_clip)
        self.logit_scale = float(cfg.kron_logit_scale)
        self.hold_bias = float(cfg.kron_hold_bias)
        self.repr_pool = str(getattr(cfg, "repr_pool", "last") or "last").strip().lower()

        self.tokenizer.eval()
        self.model.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        for p in self.model.parameters():
            p.requires_grad = False

    def _prepare_inputs(
        self,
        x_ohlcva: torch.Tensor,
        x_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_ohlcva.dim() != 3 or x_ohlcva.shape[-1] != 6:
            raise ValueError(f"Expected x_ohlcva shape [B,T,6], got {tuple(x_ohlcva.shape)}")
        if x_stamp.dim() != 3 or x_stamp.shape[-1] != 5:
            raise ValueError(f"Expected x_stamp shape [B,T,5], got {tuple(x_stamp.shape)}")
        if y_stamp.dim() != 3 or y_stamp.shape[-1] != 5:
            raise ValueError(f"Expected y_stamp shape [B,1,5], got {tuple(y_stamp.shape)}")

        x = x_ohlcva.to(self.device, dtype=torch.float32)
        x_stamp = x_stamp.to(self.device, dtype=torch.float32)
        y_stamp = y_stamp.to(self.device, dtype=torch.float32)

        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        x_norm = torch.clamp((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        if x_norm.shape[1] > self.max_context:
            x_in = x_norm[:, -self.max_context:]
            x_stamp_in = x_stamp[:, -self.max_context:]
        else:
            x_in = x_norm
            x_stamp_in = x_stamp
        return x, x_mean, x_std, x_in, x_stamp_in

    @torch.no_grad()
    def predict_repr(
        self,
        x_ohlcva: torch.Tensor,
        x_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, x_in, x_stamp_in = self._prepare_inputs(x_ohlcva, x_stamp, y_stamp)
        pre_ids, post_ids = self.tokenizer.encode(x_in, half=True)
        pre_ids = pre_ids.long()
        post_ids = post_ids.long()
        _, context = self.model.decode_s1(pre_ids, post_ids, stamp=x_stamp_in)
        if self.repr_pool == "mean":
            return context.mean(dim=1).to(torch.float32)
        return context[:, -1, :].to(torch.float32)

    @torch.no_grad()
    def predict_strength(
        self,
        x_ohlcva: torch.Tensor,
        x_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
    ) -> torch.Tensor:
        x, x_mean, x_std, x_in, x_stamp_in = self._prepare_inputs(x_ohlcva, x_stamp, y_stamp)

        pre_ids, post_ids = self.tokenizer.encode(x_in, half=True)
        pre_ids = pre_ids.long()
        post_ids = post_ids.long()

        s1_logits, context = self.model.decode_s1(pre_ids, post_ids, stamp=x_stamp_in)
        s1_next = torch.argmax(s1_logits[:, -1, :], dim=-1, keepdim=True)
        s2_logits = self.model.decode_s2(context, s1_next)
        s2_next = torch.argmax(s2_logits[:, -1, :], dim=-1, keepdim=True)

        full_pre = torch.cat([pre_ids, s1_next], dim=1)
        full_post = torch.cat([post_ids, s2_next], dim=1)
        if full_pre.shape[1] > self.max_context:
            full_pre = full_pre[:, -self.max_context:]
            full_post = full_post[:, -self.max_context:]

        decoded_norm = self.tokenizer.decode([full_pre, full_post], half=True)
        pred_norm = decoded_norm[:, -1, :]
        pred = pred_norm * (x_std.squeeze(1) + 1e-5) + x_mean.squeeze(1)

        last_close = x[:, -1, 3]
        pred_close = pred[:, 3]
        ret = (pred_close - last_close) / (last_close.abs() + 1e-6)
        strength = torch.tanh(ret * 50.0) * self.logit_scale
        return strength

    @torch.no_grad()
    def predict_logits(
        self,
        x_ohlcva: torch.Tensor,
        x_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
        position_states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        strength = self.predict_strength(x_ohlcva, x_stamp, y_stamp)
        position_states = position_states.to(self.device, dtype=torch.long)
        if action_mask is not None:
            action_mask = action_mask.to(self.device, dtype=torch.float32)
        return _teacher_logits_from_strength(
            strength=strength,
            position_states=position_states,
            action_mask=action_mask,
            hold_bias=self.hold_bias,
        )


class CachedKronTeacher:
    def __init__(self, strength_map: Dict[str, np.ndarray], cfg: TeacherConfig, device: torch.device):
        self.strength_map = strength_map
        self.device = device
        self.hold_bias = float(cfg.kron_hold_bias)

    def predict_strength_from_keys(
        self,
        symbols: List[str],
        indices: np.ndarray,
    ) -> torch.Tensor:
        strengths: List[float] = []
        for sym, idx in zip(symbols, indices.tolist()):
            arr = self.strength_map.get(sym)
            if arr is None or idx < 0 or idx >= len(arr):
                strengths.append(0.0)
                continue
            val = float(arr[idx])
            if math.isnan(val):
                val = 0.0
            strengths.append(val)
        return torch.tensor(strengths, device=self.device, dtype=torch.float32)

    def predict_logits_from_keys(
        self,
        symbols: List[str],
        indices: np.ndarray,
        position_states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        strength = self.predict_strength_from_keys(symbols, indices)
        ps = position_states.to(self.device, dtype=torch.long)
        am = action_mask.to(self.device, dtype=torch.float32) if action_mask is not None else None
        return _teacher_logits_from_strength(
            strength=strength,
            position_states=ps,
            action_mask=am,
            hold_bias=self.hold_bias,
        )


class CachedKronReprTeacher:
    def __init__(self, cache_dir: Path, symbol_store: Dict[str, SymbolData], cfg: TeacherConfig, device: torch.device):
        self.cache_dir = cache_dir
        self.device = device
        self.repr_dim = max(int(getattr(cfg, "repr_dim", 256)), 1)
        self._arrays: Dict[str, np.ndarray] = {}
        self._lengths = {sym: len(data.prices) for sym, data in symbol_store.items()}

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{_safe_symbol_filename(symbol)}.npy"

    def _load_symbol(self, symbol: str) -> np.ndarray:
        arr = self._arrays.get(symbol)
        if arr is None:
            arr = np.load(self._path(symbol), mmap_mode="r")
            self._arrays[symbol] = arr
        return arr

    def predict_repr_from_keys(
        self,
        symbols: List[str],
        indices: np.ndarray,
    ) -> torch.Tensor:
        out = np.zeros((len(symbols), self.repr_dim), dtype=np.float32)
        for row_idx, (sym, idx) in enumerate(zip(symbols, indices.tolist())):
            if idx < 0 or idx >= self._lengths.get(sym, 0):
                continue
            arr = self._load_symbol(sym)
            if idx >= arr.shape[0]:
                continue
            vec = np.asarray(arr[idx], dtype=np.float32)
            if vec.shape[-1] != self.repr_dim:
                continue
            if np.isnan(vec).any():
                continue
            out[row_idx] = vec
        return torch.from_numpy(out).to(self.device, dtype=torch.float32)


class StudentPolicy(nn.Module):
    def __init__(self, cfg: Config, n_assets: int):
        super().__init__()
        mc = cfg.model
        hidden = mc.hidden_dim
        self.use_asset_ids = bool(getattr(mc, "use_asset_ids", True))
        self.backbone = MambaBackbone(
            mc.input_dim,
            hidden,
            mc.num_layers,
            mc.dropout,
            use_temporal_attention=mc.use_temporal_attention,
            temporal_attn_hidden=mc.temporal_attn_hidden,
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(6, mc.position_dim),
            nn.ReLU(),
            nn.Linear(mc.position_dim, mc.position_dim),
        )
        self.asset_embed = nn.Embedding(n_assets, mc.asset_embed_dim) if self.use_asset_ids else None

        asset_dim = mc.asset_embed_dim if self.use_asset_ids else 0
        fuse_dim = hidden + mc.position_dim + asset_dim
        self.open_fuse = nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(),
            ((nn.Identity() if float(mc.dropout) <= 0.0 else nn.Dropout(mc.dropout))),
        )
        self.close_fuse = nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(),
            ((nn.Identity() if float(mc.dropout) <= 0.0 else nn.Dropout(mc.dropout))),
        )

        # Flat state: first decide WAIT/OPEN, then conditionally emit signed size.
        self.open_gate_head = nn.Linear(hidden, 2)
        self.entry_size_head = nn.Linear(hidden, 1)
        self.entry_log_std = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.close_head = nn.Linear(hidden, 2)
        self.strength_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            ((nn.Identity() if float(mc.dropout) <= 0.0 else nn.Dropout(mc.dropout * 0.5))),
            nn.Linear(hidden, 1),
        )
        teacher_repr_dim = max(int(getattr(cfg.teacher, "repr_dim", hidden)), 1)
        self.teacher_repr_head = nn.Identity() if teacher_repr_dim == hidden else nn.Linear(hidden, teacher_repr_dim)
        self.max_logit_gap = mc.max_logit_gap

    def _encode_market(self, market: torch.Tensor) -> torch.Tensor:
        return self.backbone(market)

    def _shared_from_encoded(
        self,
        encoded_market: torch.Tensor,
        pos_info: torch.Tensor,
        asset_ids: torch.Tensor,
    ) -> torch.Tensor:
        p = self.pos_proj(pos_info)
        if self.use_asset_ids:
            a = self.asset_embed(asset_ids)
            return torch.cat([encoded_market, p, a], dim=-1)
        return torch.cat([encoded_market, p], dim=-1)

    def _shared(self, market: torch.Tensor, pos_info: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        encoded_market = self._encode_market(market)
        return self._shared_from_encoded(encoded_market, pos_info, asset_ids)

    def _apply_categorical_postprocess(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        clamp: bool = True,
    ) -> torch.Tensor:
        if clamp and self.max_logit_gap < 100:
            max_val = logits.max(dim=-1, keepdim=True).values
            logits = logits.clamp(min=max_val - self.max_logit_gap)
        if action_mask is not None:
            mask = action_mask.to(dtype=logits.dtype)
            logits = logits + (mask - 1.0) * logits.new_tensor(1e9)
        return logits

    def _compute_outputs(
        self,
        market: torch.Tensor,
        pos_info: torch.Tensor,
        asset_ids: torch.Tensor,
        position_states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        clamp: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded_market = self._encode_market(market)
        shared = self._shared_from_encoded(encoded_market, pos_info, asset_ids)
        open_feat = self.open_fuse(shared)
        close_feat = self.close_fuse(shared)

        open_gate_logits = self.open_gate_head(open_feat)
        entry_mean = self.entry_size_head(open_feat).squeeze(-1)
        entry_log_std = self.entry_log_std.clamp(-4.0, 1.0).expand_as(entry_mean)
        close_logits = self.close_head(close_feat)
        if action_mask is not None:
            empty_mask = position_states == States.EMPTY
            hold_mask = ~empty_mask
            if empty_mask.any():
                open_gate_logits = open_gate_logits.clone()
                open_gate_logits[empty_mask] = self._apply_categorical_postprocess(
                    open_gate_logits[empty_mask],
                    action_mask[empty_mask],
                    clamp=False,
                )
            if hold_mask.any():
                close_logits = close_logits.clone()
                close_logits[hold_mask] = self._apply_categorical_postprocess(
                    close_logits[hold_mask],
                    action_mask[hold_mask],
                    clamp=clamp,
                )
        elif clamp:
            close_logits = self._apply_categorical_postprocess(close_logits, action_mask=None, clamp=True)
        strength = self.strength_head(encoded_market).squeeze(-1)
        return {
            "encoded_market": encoded_market,
            "shared": shared,
            "teacher_repr": self.teacher_repr_head(encoded_market),
            "open_gate_logits": open_gate_logits,
            "entry_mean": entry_mean,
            "entry_log_std": entry_log_std,
            "entry_signal": torch.tanh(entry_mean),
            "close_logits": close_logits,
            "strength": strength,
        }

    def predict_strength(self, market: torch.Tensor) -> torch.Tensor:
        encoded_market = self._encode_market(market)
        return self.strength_head(encoded_market).squeeze(-1)

    @staticmethod
    def _atanh_clamped(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _entry_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        pre_tanh = self._atanh_clamped(action)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return log_prob

    @staticmethod
    def _ensure_action_shape(actions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if actions is None:
            return None
        if actions.dim() == 1:
            zeros = torch.zeros_like(actions, dtype=torch.float32)
            return torch.stack([actions.to(torch.float32), zeros], dim=-1)
        return actions.to(torch.float32)

    def _sample_or_score_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        position_states: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = outputs["entry_mean"].device
        bsz = outputs["entry_mean"].shape[0]
        out_actions = torch.zeros((bsz, 2), device=device, dtype=torch.float32)
        log_probs = torch.zeros(bsz, device=device, dtype=torch.float32)
        entropies = torch.zeros(bsz, device=device, dtype=torch.float32)
        actions = self._ensure_action_shape(actions)

        empty = position_states == States.EMPTY
        hold = ~empty

        if empty.any():
            gate_logits = outputs["open_gate_logits"][empty]
            if not deterministic and temperature != 1.0:
                gate_logits = gate_logits / max(float(temperature), 1e-6)
            gate_dist = torch.distributions.Categorical(logits=gate_logits)
            mean = outputs["entry_mean"][empty]
            log_std = outputs["entry_log_std"][empty]
            std = log_std.exp() * max(float(temperature), 1e-6)
            normal = torch.distributions.Normal(mean, std)

            empty_actions = torch.zeros((empty.sum(), 2), device=device, dtype=torch.float32)
            empty_log_probs = torch.zeros(empty.sum(), device=device, dtype=torch.float32)
            empty_entropies = torch.zeros(empty.sum(), device=device, dtype=torch.float32)

            if actions is None:
                gate_actions = torch.argmax(gate_logits, dim=-1) if deterministic else gate_dist.sample()
                empty_actions[:, 0] = gate_actions.to(torch.float32)
                open_selected = gate_actions == int(FlatActions.OPEN)
                if open_selected.any():
                    if deterministic:
                        size_actions = torch.tanh(mean[open_selected])
                    else:
                        size_actions = torch.tanh(normal.rsample()[open_selected])
                    empty_actions[open_selected, 1] = size_actions
                if not deterministic:
                    empty_log_probs = gate_dist.log_prob(gate_actions)
                    empty_entropies = gate_dist.entropy()
                    if open_selected.any():
                        empty_log_probs[open_selected] += self._entry_log_prob(
                            mean[open_selected],
                            log_std[open_selected],
                            empty_actions[open_selected, 1],
                        )
                        empty_entropies[open_selected] += normal.entropy()[open_selected]
            else:
                flat_actions = actions[empty]
                gate_actions = flat_actions[:, 0].round().to(torch.long).clamp(int(FlatActions.WAIT), int(FlatActions.OPEN))
                size_actions = flat_actions[:, 1].clamp(-0.999999, 0.999999)
                open_selected = gate_actions == int(FlatActions.OPEN)
                empty_actions[:, 0] = gate_actions.to(torch.float32)
                empty_actions[open_selected, 1] = size_actions[open_selected]
                empty_log_probs = gate_dist.log_prob(gate_actions)
                empty_entropies = gate_dist.entropy()
                if open_selected.any():
                    empty_log_probs[open_selected] += self._entry_log_prob(
                        mean[open_selected],
                        log_std[open_selected],
                        size_actions[open_selected],
                    )
                    empty_entropies[open_selected] += normal.entropy()[open_selected]

            out_actions[empty] = empty_actions
            log_probs[empty] = empty_log_probs
            entropies[empty] = empty_entropies

        if hold.any():
            logits = outputs["close_logits"][hold]
            if not deterministic and temperature != 1.0:
                logits = logits / max(float(temperature), 1e-6)
            dist = torch.distributions.Categorical(logits=logits)
            if actions is None:
                hold_actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            else:
                hold_actions = actions[hold, 0].round().to(torch.long).clamp(int(HoldActions.CLOSE), int(HoldActions.HOLD))
            out_actions[hold, 0] = hold_actions.to(torch.float32)
            if not deterministic or actions is not None:
                log_probs[hold] = dist.log_prob(hold_actions)
                entropies[hold] = dist.entropy()

        return out_actions, log_probs, entropies

    def get_student_logits(
        self,
        market: torch.Tensor,
        pos_info: torch.Tensor,
        asset_ids: torch.Tensor,
        position_states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        outputs = self._compute_outputs(
            market,
            pos_info,
            asset_ids,
            position_states,
            action_mask=action_mask,
            clamp=clamp,
        )
        surrogate = outputs["open_gate_logits"]
        hold = position_states != States.EMPTY
        if hold.any():
            surrogate = surrogate.clone()
            surrogate[hold] = outputs["close_logits"][hold]
        return surrogate

    def get_logits(
        self,
        market: torch.Tensor,
        pos_info: torch.Tensor,
        asset_ids: torch.Tensor,
        position_states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        return self.get_student_logits(
            market,
            pos_info,
            asset_ids,
            position_states,
            action_mask=action_mask,
            clamp=clamp,
        )

    def forward(
        self,
        market,
        pos_info,
        asset_ids,
        position_states,
        action_mask=None,
    ):
        outputs = self._compute_outputs(
            market,
            pos_info,
            asset_ids,
            position_states,
            action_mask=action_mask,
        )
        actions, log_probs, _ = self._sample_or_score_actions(
            outputs,
            position_states,
            action_mask,
            actions=None,
            deterministic=False,
            temperature=1.0,
        )
        return actions, log_probs

    def get_log_prob_and_entropy(
        self,
        market,
        pos_info,
        asset_ids,
        position_states,
        actions,
        action_mask=None,
    ):
        outputs = self._compute_outputs(
            market,
            pos_info,
            asset_ids,
            position_states,
            action_mask=action_mask,
        )
        _, log_probs, entropies = self._sample_or_score_actions(
            outputs,
            position_states,
            action_mask,
            actions=actions,
            deterministic=False,
            temperature=1.0,
        )
        return log_probs, entropies

    def act_with_temperature(
        self,
        market,
        pos_info,
        asset_ids,
        position_states,
        action_mask=None,
        temperature=1.0,
    ):
        outputs = self._compute_outputs(
            market,
            pos_info,
            asset_ids,
            position_states,
            action_mask=action_mask,
            clamp=False,
        )
        actions, _, _ = self._sample_or_score_actions(
            outputs,
            position_states,
            action_mask,
            actions=None,
            deterministic=False,
            temperature=temperature,
        )
        return actions

    def act_deterministic(
        self,
        market,
        pos_info,
        asset_ids,
        position_states,
        action_mask=None,
    ):
        outputs = self._compute_outputs(
            market,
            pos_info,
            asset_ids,
            position_states,
            action_mask=action_mask,
            clamp=False,
        )
        actions, _, _ = self._sample_or_score_actions(
            outputs,
            position_states,
            action_mask,
            actions=None,
            deterministic=True,
            temperature=1.0,
        )
        return actions


class StudentValue(nn.Module):
    def __init__(self, cfg: Config, n_assets: int):
        super().__init__()
        mc = cfg.model
        hidden = mc.value_hidden_dim
        self.use_asset_ids = bool(getattr(mc, "use_asset_ids", True))
        self.backbone = MambaBackbone(
            mc.input_dim,
            hidden,
            mc.value_num_layers,
            mc.dropout,
            use_temporal_attention=mc.value_use_temporal_attention,
            temporal_attn_hidden=mc.temporal_attn_hidden,
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(6, mc.position_dim),
            nn.ReLU(),
            nn.Linear(mc.position_dim, mc.position_dim),
        )
        self.asset_embed = nn.Embedding(n_assets, mc.asset_embed_dim) if self.use_asset_ids else None

        asset_dim = mc.asset_embed_dim if self.use_asset_ids else 0
        fuse_dim = hidden + mc.position_dim + asset_dim
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(),
            ((nn.Identity() if float(mc.dropout) <= 0.0 else nn.Dropout(mc.dropout))),
            nn.Linear(hidden, 1),
        )

    def forward(self, market, pos_info, asset_ids):
        h = self.backbone(market)
        p = self.pos_proj(pos_info)
        if self.use_asset_ids:
            a = self.asset_embed(asset_ids)
            fused = torch.cat([h, p, a], dim=-1)
        else:
            fused = torch.cat([h, p], dim=-1)
        return self.head(fused).squeeze(-1)


# -----------------------------
# Rollout and collector
# -----------------------------


@dataclass
class RolloutBatch:
    market: np.ndarray
    pos_info: np.ndarray
    teacher_strength: np.ndarray
    teacher_logits: np.ndarray
    teacher_repr: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    trade_ends: np.ndarray
    pos_states: np.ndarray
    action_masks: np.ndarray
    asset_ids: np.ndarray

    next_market: np.ndarray
    next_pos_info: np.ndarray
    next_pos_states: np.ndarray
    next_action_masks: np.ndarray
    next_asset_ids: np.ndarray

    reward_open: np.ndarray
    reward_hold: np.ndarray
    reward_close: np.ndarray
    tier_weights: np.ndarray  # shape [T, N], per-env tier loss weight


class RolloutBuffer:
    def __init__(self):
        self.market = []
        self.pos_info = []
        self.teacher_strength = []
        self.teacher_logits = []
        self.teacher_repr = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.dones = []
        self.trade_ends = []
        self.pos_states = []
        self.action_masks = []
        self.asset_ids = []
        self.reward_open = []
        self.reward_hold = []
        self.reward_close = []
        self.env_tier_weights: Optional[np.ndarray] = None  # [N], set once per rollout

    def append_step(
        self,
        market,
        pos_info,
        teacher_strength,
        teacher_logits,
        teacher_repr,
        actions,
        log_probs,
        rewards,
        dones,
        trade_ends,
        pos_states,
        action_masks,
        asset_ids,
        reward_open,
        reward_hold,
        reward_close,
    ):
        self.market.append(market)
        self.pos_info.append(pos_info)
        self.teacher_strength.append(teacher_strength)
        self.teacher_logits.append(teacher_logits)
        self.teacher_repr.append(teacher_repr)
        self.actions.append(actions)
        self.old_log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.trade_ends.append(trade_ends)
        self.pos_states.append(pos_states)
        self.action_masks.append(action_masks)
        self.asset_ids.append(asset_ids)
        self.reward_open.append(reward_open)
        self.reward_hold.append(reward_hold)
        self.reward_close.append(reward_close)

    def to_batch(self, next_market, next_pos_info, next_pos_states, next_action_masks, next_asset_ids) -> RolloutBatch:
        return RolloutBatch(
            market=np.asarray(self.market, dtype=np.float32),
            pos_info=np.asarray(self.pos_info, dtype=np.float32),
            teacher_strength=np.asarray(self.teacher_strength, dtype=np.float32),
            teacher_logits=np.asarray(self.teacher_logits, dtype=np.float32),
            teacher_repr=np.asarray(self.teacher_repr, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            old_log_probs=np.asarray(self.old_log_probs, dtype=np.float32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=np.bool_),
            trade_ends=np.asarray(self.trade_ends, dtype=np.bool_),
            pos_states=np.asarray(self.pos_states, dtype=np.int64),
            action_masks=np.asarray(self.action_masks, dtype=np.float32),
            asset_ids=np.asarray(self.asset_ids, dtype=np.int64),
            next_market=np.asarray(next_market, dtype=np.float32),
            next_pos_info=np.asarray(next_pos_info, dtype=np.float32),
            next_pos_states=np.asarray(next_pos_states, dtype=np.int64),
            next_action_masks=np.asarray(next_action_masks, dtype=np.float32),
            next_asset_ids=np.asarray(next_asset_ids, dtype=np.int64),
            reward_open=np.asarray(self.reward_open, dtype=np.float32),
            reward_hold=np.asarray(self.reward_hold, dtype=np.float32),
            reward_close=np.asarray(self.reward_close, dtype=np.float32),
            tier_weights=np.ones((len(self.rewards), len(self.rewards[0])), dtype=np.float32)
            if self.env_tier_weights is None
            else np.broadcast_to(
                self.env_tier_weights[None, :],
                (len(self.rewards), len(self.env_tier_weights)),
            ).copy(),
        )


class Collector:
    def __init__(
        self,
        cfg: Config,
        symbol_store: Dict[str, SymbolData],
        device: torch.device,
        teacher: Optional[object] = None,
    ):
        self.cfg = cfg
        self.symbol_store = symbol_store
        self.device = device
        self.teacher = teacher
        self.fixed_val_manifest_cache: Dict[Tuple[int, int], List[EvalEpisode]] = {}

    def _get_fixed_val_manifest(self, fixed_seed: int, num_envs: int, max_steps: int) -> List[EvalEpisode]:
        key = (int(num_envs), int(max_steps))
        if key in self.fixed_val_manifest_cache and not bool(getattr(self.cfg.ppo, "val_fixed_manifest_rebuild", False)):
            return self.fixed_val_manifest_cache[key]

        manifest_path = _default_val_manifest_path(self.cfg, num_envs=num_envs, max_steps=max_steps)
        manifest: Optional[List[EvalEpisode]] = None
        if not bool(getattr(self.cfg.ppo, "val_fixed_manifest_rebuild", False)):
            manifest = _load_eval_manifest(
                manifest_path,
                cfg=self.cfg,
                symbol_store=self.symbol_store,
                num_envs=num_envs,
                max_steps=max_steps,
            )
            if manifest is not None:
                self.fixed_val_manifest_cache[key] = manifest
                log(f"Val fixed manifest loaded: path={manifest_path}, episodes={len(manifest)}")
                return manifest

        manifest = _build_fixed_val_manifest(
            symbol_store=self.symbol_store,
            cfg=self.cfg,
            num_envs=num_envs,
            max_steps=max_steps,
            fixed_seed=int(fixed_seed),
        )
        _write_eval_manifest(
            manifest_path,
            cfg=self.cfg,
            num_envs=num_envs,
            max_steps=max_steps,
            episodes=manifest,
        )
        counts: Dict[str, int] = {}
        for episode in manifest:
            counts[episode.bucket] = counts.get(episode.bucket, 0) + 1
        log(f"Val fixed manifest built: path={manifest_path}, episodes={len(manifest)}, buckets={counts}")
        self.fixed_val_manifest_cache[key] = manifest
        return manifest

    def _compute_teacher_targets(
        self,
        teacher_symbols: List[str],
        teacher_indices: np.ndarray,
        teacher_x: np.ndarray,
        teacher_x_stamp: np.ndarray,
        teacher_y_stamp: np.ndarray,
        pos_states: np.ndarray,
        action_masks: np.ndarray,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.teacher is None:
            return None, None, None

        strength: Optional[torch.Tensor] = None
        teacher_repr: Optional[torch.Tensor] = None
        t_x = t_x_stamp = t_y_stamp = None

        if hasattr(self.teacher, "predict_strength_from_keys"):
            strength = self.teacher.predict_strength_from_keys(teacher_symbols, teacher_indices).to(
                self.device, dtype=torch.float32
            )
        elif hasattr(self.teacher, "predict_strength"):
            t_x = torch.from_numpy(teacher_x).to(self.device)
            t_x_stamp = torch.from_numpy(teacher_x_stamp).to(self.device)
            t_y_stamp = torch.from_numpy(teacher_y_stamp).to(self.device)
            strength = self.teacher.predict_strength(t_x, t_x_stamp, t_y_stamp).to(self.device, dtype=torch.float32)

        if hasattr(self.teacher, "predict_repr_from_keys"):
            teacher_repr = self.teacher.predict_repr_from_keys(teacher_symbols, teacher_indices).to(
                self.device, dtype=torch.float32
            )
        elif hasattr(self.teacher, "predict_repr"):
            if t_x is None:
                t_x = torch.from_numpy(teacher_x).to(self.device)
                t_x_stamp = torch.from_numpy(teacher_x_stamp).to(self.device)
                t_y_stamp = torch.from_numpy(teacher_y_stamp).to(self.device)
            teacher_repr = self.teacher.predict_repr(t_x, t_x_stamp, t_y_stamp).to(self.device, dtype=torch.float32)

        teacher_logits: Optional[torch.Tensor] = None
        if strength is not None:
            t_pos_states = torch.from_numpy(pos_states).to(self.device, dtype=torch.long)
            t_action_masks = torch.from_numpy(action_masks).to(self.device, dtype=torch.float32)
            teacher_logits = _teacher_logits_from_strength(
                strength=strength,
                position_states=t_pos_states,
                action_mask=t_action_masks,
                hold_bias=float(getattr(self.teacher, "hold_bias", self.cfg.teacher.kron_hold_bias)),
            ).to(self.device, dtype=torch.float32)
        return strength, teacher_logits, teacher_repr

    def collect(
        self,
        policy: StudentPolicy,
        mode: str = "train",
        update_count: int = 0,
        deterministic_eval: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        del update_count
        py_random_state = None
        np_random_state = None
        torch_random_state = None
        torch_cuda_random_state = None
        if fixed_seed is not None:
            py_random_state = random.getstate()
            np_random_state = np.random.get_state()
            torch_random_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                torch_cuda_random_state = torch.cuda.get_rng_state_all()
            random.seed(int(fixed_seed))
            np.random.seed(int(fixed_seed))
            torch.manual_seed(int(fixed_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(fixed_seed))

        try:
            if mode == "train":
                num_envs = self.cfg.ppo.envs_per_batch
                max_steps = self.cfg.ppo.trajectory_length
            elif mode == "val":
                if fixed_seed is not None:
                    num_envs = int(getattr(self.cfg.ppo, "val_fixed_envs", 0) or self.cfg.ppo.val_envs)
                    max_steps = int(
                        getattr(self.cfg.ppo, "val_fixed_trajectory_length", 0) or self.cfg.ppo.val_trajectory_length
                    )
                else:
                    num_envs = int(getattr(self.cfg.ppo, "val_random_envs", 0) or self.cfg.ppo.val_envs)
                    max_steps = int(
                        getattr(self.cfg.ppo, "val_random_trajectory_length", 0) or self.cfg.ppo.val_trajectory_length
                    )
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            fixed_episodes = None
            if mode == "val" and fixed_seed is not None:
                fixed_episodes = self._get_fixed_val_manifest(int(fixed_seed), num_envs=num_envs, max_steps=max_steps)

            batch_env = BatchTradingEnv(
                symbol_store=self.symbol_store,
                cfg=self.cfg,
                mode=mode,
                num_envs=num_envs,
                max_steps=max_steps,
                fixed_episodes=fixed_episodes,
            )

            policy.eval()
            all_trades: List[TradeResult] = []

            if mode == "val":
                temp = self.cfg.ppo.val_temperature
                for _ in range(max_steps):
                    market, teacher_symbols, teacher_indices, teacher_x, teacher_x_stamp, teacher_y_stamp, pos_info, action_masks, asset_ids = (
                        batch_env.get_batch_obs()
                    )
                    pos_states = batch_env.get_batch_states()

                    t_market = torch.from_numpy(market).to(self.device)
                    t_pos = torch.from_numpy(pos_info).to(self.device)
                    t_masks = torch.from_numpy(action_masks).to(self.device)
                    t_assets = torch.from_numpy(asset_ids).to(self.device)
                    t_states = torch.from_numpy(pos_states).to(self.device)
                    with torch.no_grad():
                        if deterministic_eval:
                            actions = policy.act_deterministic(
                                t_market,
                                t_pos,
                                t_assets,
                                t_states,
                                t_masks,
                            )
                        else:
                            actions = policy.act_with_temperature(
                                t_market,
                                t_pos,
                                t_assets,
                                t_states,
                                t_masks,
                                temperature=temp,
                            )

                    actions_np = actions.detach().cpu().numpy().astype(np.float32)
                    zero_log_probs = np.zeros((actions_np.shape[0],), dtype=np.float32)
                    _, _, _, _, _, _, trades = batch_env.step_batch(actions_np, zero_log_probs)
                    all_trades.extend(trades)

                return all_trades

            buffer = RolloutBuffer()
            # 设置每个env的tier权重（归一化后乘到advantages上）
            _tier_w_map = {"tier1": 2.5, "tier2": 1.0, "tier3": 0.5}
            buffer.env_tier_weights = np.array(
                [_tier_w_map.get(self.symbol_store[e.symbol].tier, 1.0) for e in batch_env.envs],
                dtype=np.float32,
            )
            for _ in range(max_steps):
                market, teacher_symbols, teacher_indices, teacher_x, teacher_x_stamp, teacher_y_stamp, pos_info, action_masks, asset_ids = (
                    batch_env.get_batch_obs()
                )
                pos_states = batch_env.get_batch_states()

                t_market = torch.from_numpy(market).to(self.device)
                t_pos = torch.from_numpy(pos_info).to(self.device)
                t_masks = torch.from_numpy(action_masks).to(self.device)
                t_assets = torch.from_numpy(asset_ids).to(self.device)
                t_states = torch.from_numpy(pos_states).to(self.device)
                teacher_strength, teacher_logits, teacher_repr = self._compute_teacher_targets(
                    teacher_symbols,
                    teacher_indices,
                    teacher_x,
                    teacher_x_stamp,
                    teacher_y_stamp,
                    pos_states,
                    action_masks,
                )

                with torch.no_grad():
                    actions, log_probs = policy(
                        t_market,
                        t_pos,
                        t_assets,
                        t_states,
                        t_masks,
                    )

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy().astype(np.float32)
                if teacher_strength is None:
                    teacher_strength_np = np.zeros((actions_np.shape[0],), dtype=np.float32)
                    teacher_logits_np = np.zeros((actions_np.shape[0], 2), dtype=np.float32)
                else:
                    teacher_strength_np = teacher_strength.detach().cpu().numpy().astype(np.float32)
                    teacher_logits_np = teacher_logits.detach().cpu().numpy().astype(np.float32)
                if teacher_repr is None:
                    teacher_repr_np = np.zeros((actions_np.shape[0], max(int(getattr(self.cfg.teacher, "repr_dim", 256)), 1)), dtype=np.float32)
                else:
                    teacher_repr_np = teacher_repr.detach().cpu().numpy().astype(np.float32)

                rewards, dones, trade_ends, ro, rh, rc, trades = batch_env.step_batch(actions_np, log_probs_np)
                all_trades.extend(trades)

                buffer.append_step(
                    market=market,
                    pos_info=pos_info,
                    teacher_strength=teacher_strength_np,
                    teacher_logits=teacher_logits_np,
                    teacher_repr=teacher_repr_np,
                    actions=actions_np,
                    log_probs=log_probs_np,
                    rewards=rewards,
                    dones=dones,
                    trade_ends=trade_ends,
                    pos_states=pos_states,
                    action_masks=action_masks,
                    asset_ids=asset_ids,
                    reward_open=ro,
                    reward_hold=rh,
                    reward_close=rc,
                )

            next_market, _, _, _, _, _, next_pos_info, next_action_masks, next_asset_ids = batch_env.get_batch_obs()
            next_pos_states = batch_env.get_batch_states()

            rollout = buffer.to_batch(
                next_market=next_market,
                next_pos_info=next_pos_info,
                next_pos_states=next_pos_states,
                next_action_masks=next_action_masks,
                next_asset_ids=next_asset_ids,
            )

            return rollout, all_trades
        finally:
            if py_random_state is not None:
                random.setstate(py_random_state)
            if np_random_state is not None:
                np.random.set_state(np_random_state)
            if torch_random_state is not None:
                torch.random.set_rng_state(torch_random_state)
            if torch_cuda_random_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_random_state)


# -----------------------------
# PPO trainer with frozen teacher
# -----------------------------


class PPOTrainer:
    def __init__(
        self,
        policy: StudentPolicy,
        value: StudentValue,
        cfg: Config,
        policy_device: torch.device,
        value_device: torch.device,
        teacher_enabled: bool = False,
    ):
        self.policy = policy
        self.value = value
        self.teacher_enabled = bool(teacher_enabled)

        self.cfg = cfg
        self.tc = cfg.ppo
        self.teacher_cfg = cfg.teacher
        self.policy_device = policy_device
        self.value_device = value_device

        self.policy_trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        if not self.policy_trainable_params:
            raise RuntimeError("No trainable policy params. Check phase.mode.")

        self.policy_optimizer = torch.optim.AdamW(
            self.policy_trainable_params,
            lr=self.tc.policy_lr,
            weight_decay=self.tc.weight_decay,
        )

        self.value_trainable_params = [p for p in self.value.parameters() if p.requires_grad]
        if not self.value_trainable_params:
            raise RuntimeError("No trainable value params.")
        self.value_optimizer = torch.optim.AdamW(
            self.value_trainable_params,
            lr=self.tc.value_lr,
            weight_decay=self.tc.weight_decay,
        )

        warmup_updates = max(int(self.tc.warmup_updates), 1)
        self.warmup_updates = warmup_updates
        total_updates = max(int(self.tc.max_updates) - warmup_updates, 1)

        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=total_updates, eta_min=1e-6
        )
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer, T_max=total_updates, eta_min=1e-6
        )

        self.update_count = 0

    def _amp_enabled_for(self, device: torch.device) -> bool:
        return bool(self.tc.use_bf16_amp) and device.type == "cuda"

    def _autocast_ctx(self, device: torch.device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self._amp_enabled_for(device))

    @staticmethod
    def _flatten_time_batch(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.reshape(-1)
        return x.reshape(-1, *x.shape[2:])

    @staticmethod
    def _flatten_env_batch(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.reshape(-1)
        return x.reshape(x.shape[0], *x.shape[1:])

    @staticmethod
    def _to_tensor(arr: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
        t = torch.from_numpy(arr)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    def _value_forward_chunked(
        self,
        market: torch.Tensor,
        pos_info: torch.Tensor,
        asset_ids: torch.Tensor,
    ) -> torch.Tensor:
        n = market.shape[0]
        bs = max(int(self.tc.value_infer_batch_size), 1)
        outs: List[torch.Tensor] = []
        for start in range(0, n, bs):
            end = min(start + bs, n)
            with self._autocast_ctx(self.value_device):
                out = self.value(market[start:end], pos_info[start:end], asset_ids[start:end])
            outs.append(out.to(torch.float32))
        return torch.cat(outs, dim=0) if outs else torch.empty((0,), device=self.value_device)

    @staticmethod
    def _linear_schedule(start: float, end: float, step: int, decay_updates: int) -> float:
        decay = max(int(decay_updates), 1)
        progress = min(float(step) / float(decay), 1.0)
        return float(start + (end - start) * progress)

    def _teacher_coef(self) -> float:
        if not self.teacher_cfg.enabled or not self.teacher_enabled:
            return 0.0
        return self._linear_schedule(
            float(self.teacher_cfg.kl_coef_start),
            float(self.teacher_cfg.kl_coef_end),
            self.update_count,
            int(self.teacher_cfg.kl_decay_updates),
        )

    def _teacher_logit_coef(self) -> float:
        if _teacher_distill_mode(self.teacher_cfg) != "action":
            return 0.0
        if not self.teacher_cfg.enabled or not self.teacher_enabled:
            return 0.0
        return self._linear_schedule(
            max(float(self.teacher_cfg.teacher_logit_weight), 0.0),
            max(float(self.teacher_cfg.teacher_logit_weight_end), 0.0),
            self.update_count,
            int(self.teacher_cfg.teacher_logit_decay_updates),
        )

    def _repr_coef(self) -> float:
        if _teacher_distill_mode(self.teacher_cfg) != "representation":
            return 0.0
        if not self.teacher_cfg.enabled or not self.teacher_enabled:
            return 0.0
        return self._linear_schedule(
            max(float(getattr(self.teacher_cfg, "repr_coef_start", 0.0)), 0.0),
            max(float(getattr(self.teacher_cfg, "repr_coef_end", 0.0)), 0.0),
            self.update_count,
            int(getattr(self.teacher_cfg, "repr_decay_updates", 1)),
        )

    def _compute_gae(self, rollout: RolloutBatch):
        market = self._to_tensor(rollout.market, self.value_device, torch.float32)
        pos_info = self._to_tensor(rollout.pos_info, self.value_device, torch.float32)
        asset_ids = self._to_tensor(rollout.asset_ids, self.value_device, torch.long)

        next_market = self._to_tensor(rollout.next_market, self.value_device, torch.float32)
        next_pos_info = self._to_tensor(rollout.next_pos_info, self.value_device, torch.float32)
        next_asset_ids = self._to_tensor(rollout.next_asset_ids, self.value_device, torch.long)

        rewards = self._to_tensor(rollout.rewards, self.value_device, torch.float32)
        dones = self._to_tensor(rollout.dones.astype(np.float32), self.value_device, torch.float32)
        trade_ends = self._to_tensor(rollout.trade_ends.astype(np.float32), self.value_device, torch.float32)

        t_steps, n_env = rewards.shape

        with torch.no_grad():
            flat_market = self._flatten_time_batch(market)
            flat_pos_info = self._flatten_time_batch(pos_info)
            flat_asset_ids = self._flatten_time_batch(asset_ids)
            values = self._value_forward_chunked(flat_market, flat_pos_info, flat_asset_ids).reshape(t_steps, n_env)

            flat_next_market = self._flatten_env_batch(next_market)
            flat_next_pos_info = self._flatten_env_batch(next_pos_info)
            flat_next_asset_ids = self._flatten_env_batch(next_asset_ids)
            next_values = self._value_forward_chunked(
                flat_next_market,
                flat_next_pos_info,
                flat_next_asset_ids,
            ).reshape(n_env)

        not_done = 1.0 - dones
        not_trade_end = 1.0 - trade_ends
        bootstrap_mask = not_done * not_trade_end

        next_values_seq = torch.zeros_like(values)
        if t_steps > 1:
            next_values_seq[:-1] = values[1:]
        next_values_seq[-1] = next_values

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(n_env, device=self.value_device)

        for t in reversed(range(t_steps)):
            next_v = next_values_seq[t]
            delta = rewards[t] + self.tc.gamma * next_v * bootstrap_mask[t] - values[t]
            gae = delta + self.tc.gamma * self.tc.gae_lambda * bootstrap_mask[t] * gae
            advantages[t] = gae

        returns = advantages + values

        if self.tc.normalize_advantages:
            flat_adv = advantages.reshape(-1)
            advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        if self.tc.normalize_returns:
            flat_ret = returns.reshape(-1)
            returns = (returns - flat_ret.mean()) / (flat_ret.std() + 1e-8)

        return advantages, returns, values

    def _build_policy_batch(self, rollout: RolloutBatch, advantages_value_dev: torch.Tensor):
        market = self._to_tensor(rollout.market, self.policy_device, torch.float32)
        pos_info = self._to_tensor(rollout.pos_info, self.policy_device, torch.float32)
        teacher_strength = self._to_tensor(rollout.teacher_strength, self.policy_device, torch.float32)
        teacher_logits = self._to_tensor(rollout.teacher_logits, self.policy_device, torch.float32)
        teacher_repr = self._to_tensor(rollout.teacher_repr, self.policy_device, torch.float32)
        actions = self._to_tensor(rollout.actions, self.policy_device, torch.float32)
        old_log_probs = self._to_tensor(rollout.old_log_probs, self.policy_device, torch.float32)
        pos_states = self._to_tensor(rollout.pos_states, self.policy_device, torch.long)
        action_masks = self._to_tensor(rollout.action_masks, self.policy_device, torch.float32)
        asset_ids = self._to_tensor(rollout.asset_ids, self.policy_device, torch.long)
        advantages = advantages_value_dev.to(self.policy_device)
        tier_weights = self._to_tensor(rollout.tier_weights, self.policy_device, torch.float32)

        return {
            "market": self._flatten_time_batch(market),
            "pos_info": self._flatten_time_batch(pos_info),
            "teacher_strength": self._flatten_time_batch(teacher_strength),
            "teacher_logits": self._flatten_time_batch(teacher_logits),
            "teacher_repr": self._flatten_time_batch(teacher_repr),
            "actions": self._flatten_time_batch(actions),
            "old_log_probs": self._flatten_time_batch(old_log_probs),
            "pos_states": self._flatten_time_batch(pos_states),
            "action_masks": self._flatten_time_batch(action_masks),
            "asset_ids": self._flatten_time_batch(asset_ids),
            "advantages": self._flatten_time_batch(advantages),
            "tier_weights": self._flatten_time_batch(tier_weights),
        }

    def _build_value_batch(self, rollout: RolloutBatch, returns_value_dev: torch.Tensor):
        market = self._to_tensor(rollout.market, self.value_device, torch.float32)
        pos_info = self._to_tensor(rollout.pos_info, self.value_device, torch.float32)
        asset_ids = self._to_tensor(rollout.asset_ids, self.value_device, torch.long)

        return {
            "market": self._flatten_time_batch(market),
            "pos_info": self._flatten_time_batch(pos_info),
            "asset_ids": self._flatten_time_batch(asset_ids),
            "returns": self._flatten_time_batch(returns_value_dev),
        }

    def _train_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        n = batch["actions"].shape[0]
        total_policy_loss = 0.0
        total_ent = 0.0
        total_kl = 0.0
        total_teacher_kl = 0.0
        total_strength_loss = 0.0
        total_repr_loss = 0.0
        early_stopped = False

        strength_base_coef = self._teacher_coef()
        teacher_logit_coef = self._teacher_logit_coef()
        strength_coef = strength_base_coef * max(float(self.teacher_cfg.strength_loss_weight), 0.0)
        repr_coef = self._repr_coef()
        teacher_trust_scale = max(float(getattr(self.teacher_cfg, "kl_trust_scale", 1.0)), 1e-6)
        log_ratio_clip = max(float(getattr(self.tc, "log_ratio_clip", 6.0)), 1.0)
        hard_kl_limit = max(
            float(self.tc.kl_target) * max(float(getattr(self.tc, "minibatch_kl_stop_mult", 3.0)), 1.0),
            float(self.tc.kl_target),
        )

        # Gradient accumulation setup
        accumulation_steps = max(int(getattr(self.tc, "gradient_accumulation_steps", 1)), 1)

        for epoch in range(self.tc.policy_epochs):
            perm = torch.randperm(n, device=self.policy_device)
            epoch_policy = 0.0
            epoch_ent = 0.0
            epoch_kl = 0.0
            epoch_teacher = 0.0
            epoch_strength = 0.0
            epoch_repr = 0.0
            n_batches = 0
            hard_stopped = False

            for mini_batch_idx, start in enumerate(range(0, n, self.tc.mini_batch_size)):
                idx = perm[start:start + self.tc.mini_batch_size]

                mb_market = batch["market"][idx]
                mb_pos_info = batch["pos_info"][idx]
                mb_asset_ids = batch["asset_ids"][idx]
                mb_pos_states = batch["pos_states"][idx]
                mb_action_masks = batch["action_masks"][idx]
                mb_actions = batch["actions"][idx]

                with self._autocast_ctx(self.policy_device):
                    outputs = self.policy._compute_outputs(
                        mb_market,
                        mb_pos_info,
                        mb_asset_ids,
                        mb_pos_states,
                        action_mask=mb_action_masks,
                        clamp=True,
                    )
                    outputs = {
                        k: (v.to(torch.float32) if torch.is_tensor(v) and v.dtype != torch.float32 else v)
                        for k, v in outputs.items()
                    }
                    _, log_probs, entropy = self.policy._sample_or_score_actions(
                        outputs,
                        mb_pos_states,
                        mb_action_masks,
                        actions=mb_actions,
                        deterministic=False,
                        temperature=1.0,
                    )

                old_lp = batch["old_log_probs"][idx]
                adv = batch["advantages"][idx] * batch["tier_weights"][idx]

                log_ratio = (log_probs - old_lp).clamp(-log_ratio_clip, log_ratio_clip)
                ratio = log_ratio.exp()
                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - self.tc.clip_epsilon, 1.0 + self.tc.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                ent = entropy.mean()
                loss = policy_loss - self.tc.entropy_coef * ent

                dir_balance_penalty = torch.tensor(0.0, device=self.policy_device)
                if self.tc.enable_direction_balance:
                    open_mask = mb_pos_states == States.EMPTY
                    if open_mask.any():
                        entry_signal = outputs["entry_signal"][open_mask]
                        dir_balance_penalty = entry_signal.mean().pow(2)
                        loss = loss + self.tc.direction_balance_strength * dir_balance_penalty

                teacher_kl_value = torch.tensor(0.0, device=self.policy_device)
                strength_loss_value = torch.tensor(0.0, device=self.policy_device)
                repr_loss_value = torch.tensor(0.0, device=self.policy_device)
                if teacher_logit_coef > 0.0 and self.teacher_enabled:
                    student_logits = outputs["open_gate_logits"]
                    hold_mask = mb_pos_states != States.EMPTY
                    if hold_mask.any():
                        student_logits = student_logits.clone()
                        student_logits[hold_mask] = outputs["close_logits"][hold_mask]
                    target_teacher_logits = batch["teacher_logits"][idx]
                    with torch.no_grad():
                        target_teacher_probs = torch.softmax(target_teacher_logits * teacher_trust_scale, dim=-1)
                    teacher_kl_value = F.kl_div(
                        F.log_softmax(student_logits, dim=-1),
                        target_teacher_probs,
                        reduction="batchmean",
                    )
                    loss = loss + teacher_logit_coef * teacher_kl_value
                if strength_coef > 0.0 and self.teacher_enabled:
                    target_strength = batch["teacher_strength"][idx]
                    strength_loss_value = F.smooth_l1_loss(outputs["strength"], target_strength)
                    loss = loss + strength_coef * strength_loss_value
                if repr_coef > 0.0 and self.teacher_enabled:
                    target_teacher_repr = batch["teacher_repr"][idx]
                    valid_repr = target_teacher_repr.abs().sum(dim=-1) > 0
                    if valid_repr.any():
                        student_teacher_repr = outputs["teacher_repr"][valid_repr]
                        target_teacher_repr = target_teacher_repr[valid_repr]
                        student_teacher_repr = F.normalize(student_teacher_repr, dim=-1)
                        target_teacher_repr = F.normalize(target_teacher_repr, dim=-1)
                        repr_loss_value = 1.0 - (student_teacher_repr * target_teacher_repr).sum(dim=-1).mean()
                        loss = loss + repr_coef * repr_loss_value

                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

                with torch.no_grad():
                    batch_kl = ((ratio - 1.0) - log_ratio).mean().item()

                if (not math.isfinite(batch_kl)) or batch_kl > hard_kl_limit:
                    early_stopped = True
                    hard_stopped = True
                    break
                if not bool(torch.isfinite(loss).all().item()):
                    early_stopped = True
                    hard_stopped = True
                    break

                # Backward pass (gradients accumulate)
                loss.backward()

                # Update parameters every accumulation_steps mini-batches
                if (mini_batch_idx + 1) % accumulation_steps == 0 or (start + self.tc.mini_batch_size) >= n:
                    grad_norm = nn.utils.clip_grad_norm_(self.policy_trainable_params, self.tc.max_grad_norm)
                    grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    if not math.isfinite(grad_norm_value):
                        self.policy_optimizer.zero_grad()
                        early_stopped = True
                        hard_stopped = True
                        break
                    self.policy_optimizer.step()
                    self.policy_optimizer.zero_grad()

                epoch_policy += policy_loss.item()
                epoch_ent += ent.item()
                epoch_kl += batch_kl
                epoch_teacher += float(teacher_kl_value.item())
                epoch_strength += float(strength_loss_value.item())
                epoch_repr += float(repr_loss_value.item())
                n_batches += 1

            avg_kl = epoch_kl / max(n_batches, 1)
            total_policy_loss += epoch_policy / max(n_batches, 1)
            total_ent += epoch_ent / max(n_batches, 1)
            total_kl += avg_kl
            total_teacher_kl += epoch_teacher / max(n_batches, 1)
            total_strength_loss += epoch_strength / max(n_batches, 1)
            total_repr_loss += epoch_repr / max(n_batches, 1)

            if hard_stopped or avg_kl > self.tc.kl_target:
                early_stopped = True
                break

        used_epochs = epoch + 1
        return {
            "policy_loss": total_policy_loss / used_epochs,
            "entropy": total_ent / used_epochs,
            "kl": total_kl / used_epochs,
            "teacher_kl": total_teacher_kl / used_epochs,
            "strength_loss": total_strength_loss / used_epochs,
            "repr_loss": total_repr_loss / used_epochs,
            "teacher_coef": teacher_logit_coef,
            "strength_coef": strength_coef,
            "repr_coef": repr_coef,
            "policy_epochs_actual": used_epochs,
            "kl_early_stop": early_stopped,
        }

    def _train_value(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        n = batch["returns"].shape[0]
        total_loss = 0.0

        for _ in range(self.tc.value_epochs):
            perm = torch.randperm(n, device=self.value_device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.tc.mini_batch_size):
                idx = perm[start:start + self.tc.mini_batch_size]
                with self._autocast_ctx(self.value_device):
                    pred = self.value(
                        batch["market"][idx],
                        batch["pos_info"][idx],
                        batch["asset_ids"][idx],
                    )
                pred = pred.to(torch.float32)
                loss = nn.functional.smooth_l1_loss(pred, batch["returns"][idx]) * self.tc.value_loss_coef

                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.value_trainable_params, self.tc.max_grad_norm)
                self.value_optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            total_loss += epoch_loss / max(n_batches, 1)

        return {"value_loss": total_loss / max(self.tc.value_epochs, 1)}

    def update(self, rollout: RolloutBatch) -> Dict[str, float]:
        self.policy.train()
        self.value.train()

        t_update0 = time.time()
        t0 = time.time()
        advantages, returns, values = self._compute_gae(rollout)
        gae_s = time.time() - t0
        if torch.isnan(advantages).any() or torch.isnan(returns).any():
            return {"skipped": True, "reason": "nan_in_gae"}

        t0 = time.time()
        policy_batch = self._build_policy_batch(rollout, advantages)
        value_batch = self._build_value_batch(rollout, returns)
        batch_build_s = time.time() - t0

        def _run_timed(fn, batch):
            started = time.time()
            return fn(batch), time.time() - started

        if bool(self.tc.parallel_value_policy_update) and self.policy_device != self.value_device:
            try:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_value = ex.submit(_run_timed, self._train_value, value_batch)
                    f_policy = ex.submit(_run_timed, self._train_policy, policy_batch)
                    value_result = f_value.result()
                    policy_result = f_policy.result()
                value_stats, value_train_s = value_result
                policy_stats, policy_train_s = policy_result
            except Exception as e:
                log(f"[warn] parallel update failed, fallback to serial: {e}")
                t0 = time.time()
                value_stats = self._train_value(value_batch)
                value_train_s = time.time() - t0

                t0 = time.time()
                policy_stats = self._train_policy(policy_batch)
                policy_train_s = time.time() - t0
        else:
            t0 = time.time()
            value_stats = self._train_value(value_batch)
            value_train_s = time.time() - t0

            t0 = time.time()
            policy_stats = self._train_policy(policy_batch)
            policy_train_s = time.time() - t0

        t0 = time.time()
        self.policy_scheduler.step()
        self.value_scheduler.step()
        sched_s = time.time() - t0
        self.update_count += 1

        if self.update_count <= self.warmup_updates:
            factor = self.update_count / max(self.warmup_updates, 1)
            for g in self.policy_optimizer.param_groups:
                g["lr"] = self.tc.policy_lr * factor
            for g in self.value_optimizer.param_groups:
                g["lr"] = self.tc.value_lr * factor

        return {
            **policy_stats,
            **value_stats,
            "update": self.update_count,
            "n_steps": int(np.prod(rollout.rewards.shape)),
            "reward_mean": float(rollout.rewards.mean()),
            "reward_std": float(rollout.rewards.std()),
            "reward_open_mean": float(rollout.reward_open.mean()),
            "reward_hold_mean": float(rollout.reward_hold.mean()),
            "reward_close_mean": float(rollout.reward_close.mean()),
            "adv_mean": float(advantages.mean().item()),
            "adv_std": float(advantages.std().item()),
            "ret_mean": float(returns.mean().item()),
            "ret_std": float(returns.std().item()),
            "value_mean": float(values.mean().item()),
            "value_std": float(values.std().item()),
            "gae_s": float(gae_s),
            "batch_build_s": float(batch_build_s),
            "value_train_s": float(value_train_s),
            "policy_train_s": float(policy_train_s),
            "sched_s": float(sched_s),
            "update_compute_s": float(time.time() - t_update0),
            "policy_lr": float(self.policy_optimizer.param_groups[0]["lr"]),
            "value_lr": float(self.value_optimizer.param_groups[0]["lr"]),
        }

    def state_dict(self) -> Dict:
        return {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "policy_scheduler": self.policy_scheduler.state_dict(),
            "value_scheduler": self.value_scheduler.state_dict(),
            "update_count": self.update_count,
            "config": asdict(self.cfg),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.policy.load_state_dict(state.get("policy", {}), strict=False)
        if "value" in state:
            self.value.load_state_dict(state["value"], strict=False)

        if "policy_optimizer" in state:
            try:
                self.policy_optimizer.load_state_dict(state["policy_optimizer"])
            except Exception:
                log("[warn] policy optimizer state mismatch, reinit optimizer state")

        if "value_optimizer" in state:
            try:
                self.value_optimizer.load_state_dict(state["value_optimizer"])
            except Exception:
                log("[warn] value optimizer state mismatch, reinit optimizer state")

        if "policy_scheduler" in state:
            try:
                self.policy_scheduler.load_state_dict(state["policy_scheduler"])
            except Exception:
                pass

        if "value_scheduler" in state:
            try:
                self.value_scheduler.load_state_dict(state["value_scheduler"])
            except Exception:
                pass

        self.update_count = int(state.get("update_count", 0))


# -----------------------------
# Checkpoint / validation
# -----------------------------


def save_checkpoint(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _infer_market_group(symbol: str) -> str:
    sym = (symbol or "").upper()
    if sym.endswith(".SZ"):
        return "CN_A_SZ"
    if sym.endswith(".SS"):
        return "CN_A_SH"
    if sym.endswith(".CNFUT"):
        return "CN_FUT"
    if sym.endswith(".GLBFUT"):
        return "GLB_FUT"
    if sym.endswith("USDT"):
        return "CRYPTO"
    if sym.endswith(".HK"):
        return "HK_EQ"
    if "." in sym:
        return f"OTHER_{sym.rsplit('.', 1)[-1]}"
    if sym.isalpha() and len(sym) <= 5:
        return "US_EQ"
    return "OTHER"


def _finalize_group_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, raw in stats.items():
        pnl_values = np.asarray(raw.get("pnl_values", []), dtype=np.float32)
        trades = int(raw.get("trades", 0))
        wins = int(raw.get("wins", 0))
        losses = pnl_values[pnl_values <= 0]
        positives = pnl_values[pnl_values > 0]
        out[key] = {
            "pnl": float(raw.get("pnl", 0.0)),
            "trades": trades,
            "wins": wins,
            "win_rate": float(wins / max(trades, 1) * 100.0),
            "avg_pnl": float(pnl_values.mean()) if pnl_values.size > 0 else 0.0,
            "avg_win": float(positives.mean()) if positives.size > 0 else 0.0,
            "avg_loss": float(losses.mean()) if losses.size > 0 else 0.0,
            "num_long": int(raw.get("num_long", 0)),
            "num_short": int(raw.get("num_short", 0)),
        }
    return out


def _format_ranked_stats_for_log(
    stats: Dict[str, Dict[str, float]],
    top_k: int = 3,
    reverse: bool = True,
) -> str:
    if not stats:
        return "n/a"
    ranked = sorted(stats.items(), key=lambda item: item[1].get("pnl", 0.0), reverse=reverse)
    parts = []
    for name, item in ranked[:top_k]:
        parts.append(
            f"{name}:{item.get('pnl', 0.0):+.3f}/{int(item.get('trades', 0))}t/{item.get('win_rate', 0.0):.0f}%"
        )
    return "; ".join(parts) if parts else "n/a"


def _log_trade_metrics(prefix: str, metrics: Dict[str, float], include_symbol_detail: bool = False) -> None:
    if not metrics:
        return

    log(
        f"  {prefix} quality: pnl={metrics['total_pnl']:+.4f}, trades={metrics['num_trades']}, "
        f"win={metrics['win_rate']:.1f}%, exp={metrics['expectancy']:+.5f}, med={metrics['median_pnl']:+.5f}, "
        f"avgW/L=({metrics['avg_win']:+.5f}/{metrics['avg_loss']:+.5f}), PF={metrics['profit_factor']:.2f}, "
        f"W/L={metrics['win_loss_ratio']:.2f}, sharpe={metrics['sharpe']:.3f}, sortino={metrics['sortino']:.3f}, "
        f"DD={metrics['max_drawdown']:.4f}"
    )
    log(
        f"  {prefix} breadth: sym={metrics['active_symbols']}, mkt={metrics['active_markets']}, "
        f"profitable(sym/mkt)=({metrics['profitable_symbols_ratio']:.1f}%/{metrics['profitable_markets_ratio']:.1f}%), "
        f"L/S={metrics['num_long']}/{metrics['num_short']} "
        f"(ratio={metrics['long_short_ratio']:.2f}, bal={metrics['long_short_balance']:.1%}), "
        f"trades/1k={metrics['trades_per_1k_steps']:.1f}, hold(avg/L/S)=({metrics['avg_holding_steps']:.1f}/"
        f"{metrics['avg_holding_long']:.1f}/{metrics['avg_holding_short']:.1f}), "
        f"empty(avg/L/S)=({metrics['avg_empty_steps']:.1f}/{metrics['long_avg_empty']:.1f}/{metrics['short_avg_empty']:.1f})"
    )
    log(
        f"  {prefix} exits: stop={metrics['stop_loss_trades']} "
        f"(fixed={metrics['stop_loss_fixed_trades']}, vol={metrics['stop_loss_vol_trades']}, "
        f"fixed%={metrics['stop_loss_fixed_ratio']:.1f}, vol%={metrics['stop_loss_vol_ratio']:.1f}), "
        f"signal={metrics['close_signal_trades']}, tp={metrics['take_profit_trades']}, force={metrics['force_close_trades']}, "
        f"sl_rate={metrics.get('stop_loss_rate', 0.0):.1f}%, "
        f"avg_sl_val={metrics.get('avg_stop_loss_value', 0.0):.4f}({metrics.get('avg_stop_loss_value', 0.0)*100:.2f}%), "
        f"avg_sl_pnl={metrics.get('avg_stop_loss_pnl', 0.0):+.5f}"
    )
    if metrics.get('market_stats'):
        log(f"  {prefix} markets+: {_format_ranked_stats_for_log(metrics['market_stats'], top_k=4, reverse=True)}")
        log(f"  {prefix} markets-: {_format_ranked_stats_for_log(metrics['market_stats'], top_k=3, reverse=False)}")
    if include_symbol_detail and metrics.get('symbol_stats'):
        log(f"  {prefix} symbols+: {_format_ranked_stats_for_log(metrics['symbol_stats'], top_k=5, reverse=True)}")
        log(f"  {prefix} symbols-: {_format_ranked_stats_for_log(metrics['symbol_stats'], top_k=5, reverse=False)}")


def compute_trade_metrics(
    trades: List[TradeResult],
    symbols: Optional[List[str]] = None,
    val_steps: int = 0,
) -> Dict[str, float]:
    if not trades:
        return {
            "val_pnl": 0.0,
            "total_pnl": 0.0,
            "val_trades": 0,
            "num_trades": 0,
            "num_long": 0,
            "num_short": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "median_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "pnl_std": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "profit_factor": 0.0,
            "win_loss_ratio": 0.0,
            "sortino": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "long_short_balance": 0.0,
            "long_short_ratio": 0.0,
            "trades_per_1k_steps": 0.0,
            "avg_holding": 0.0,
            "avg_holding_steps": 0.0,
            "avg_empty": 0.0,
            "avg_empty_steps": 0.0,
            "long_avg_holding": 0.0,
            "avg_holding_long": 0.0,
            "short_avg_holding": 0.0,
            "avg_holding_short": 0.0,
            "long_avg_empty": 0.0,
            "short_avg_empty": 0.0,
            "stop_loss_trades": 0,
            "stop_loss_fixed_trades": 0,
            "stop_loss_vol_trades": 0,
            "stop_loss_fixed_ratio": 0.0,
            "stop_loss_vol_ratio": 0.0,
            "stop_loss_rate": 0.0,
            "avg_stop_loss_value": 0.0,
            "avg_stop_loss_pnl": 0.0,
            "take_profit_trades": 0,
            "close_signal_trades": 0,
            "force_close_trades": 0,
            "active_symbols": 0,
            "active_markets": 0,
            "profitable_symbols_ratio": 0.0,
            "profitable_markets_ratio": 0.0,
            "market_stats": {},
            "symbol_stats": {},
            "tier_stats": {},
        }

    pnls = np.asarray([t.pnl for t in trades], dtype=np.float32)
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    avg_win = float(wins.mean()) if wins.size > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.size > 0 else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size > 0 and float(losses.sum()) != 0.0 else 0.0
    win_loss_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0.0 else 0.0

    pnl_std = float(pnls.std()) if pnls.size > 1 else 0.0
    downside = pnls[pnls < 0]
    downside_std = float(downside.std()) if downside.size > 1 else 1e-8
    sharpe = float(pnls.mean() / (pnl_std + 1e-8)) if pnls.size > 1 else 0.0
    sortino = float(pnls.mean() / (downside_std + 1e-8))

    equity_curve = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(pnls)])
    running_peak = np.maximum.accumulate(equity_curve)
    max_drawdown = float(np.max(running_peak - equity_curve)) if equity_curve.size > 0 else 0.0

    n = len(trades)
    n_long = len(long_trades)
    n_short = len(short_trades)
    long_pnls = [t.pnl for t in long_trades]
    short_pnls = [t.pnl for t in short_trades]
    total_pnl = float(pnls.sum())
    trades_per_1k = (float(n) / max(int(val_steps), 1)) * 1000.0 if val_steps > 0 else 0.0

    symbol_stats_raw: Dict[str, Dict[str, float]] = {}
    market_stats_raw: Dict[str, Dict[str, float]] = {}
    tier_stats_raw: Dict[str, Dict[str, float]] = {}
    for t in trades:
        if symbols and 0 <= t.asset_id < len(symbols):
            sym = symbols[t.asset_id]
        else:
            sym = f"asset_{t.asset_id}"
        market = t.market_group or _infer_market_group(sym)
        tier = t.tier if t.tier else "tier3"

        for bucket, key in ((symbol_stats_raw, sym), (market_stats_raw, market), (tier_stats_raw, tier)):
            if key not in bucket:
                bucket[key] = {"pnl": 0.0, "trades": 0, "wins": 0, "num_long": 0, "num_short": 0, "pnl_values": []}
            bucket[key]["pnl"] += float(t.pnl)
            bucket[key]["trades"] += 1
            bucket[key]["num_long"] += int(t.direction == 1)
            bucket[key]["num_short"] += int(t.direction == -1)
            bucket[key]["pnl_values"].append(float(t.pnl))
            if t.pnl > 0:
                bucket[key]["wins"] += 1

    symbol_stats = _finalize_group_stats(symbol_stats_raw)
    market_stats = _finalize_group_stats(market_stats_raw)
    tier_stats = _finalize_group_stats(tier_stats_raw)

    stop_loss_trades = sum(1 for t in trades if t.exit_reason == "stop_loss")
    stop_loss_fixed_trades = sum(1 for t in trades if t.exit_reason == "stop_loss" and t.stop_source == "fixed")
    stop_loss_vol_trades = sum(1 for t in trades if t.exit_reason == "stop_loss" and t.stop_source == "volatility")
    take_profit_trades = sum(1 for t in trades if t.exit_reason == "take_profit")
    close_signal_trades = sum(1 for t in trades if t.exit_reason == "close_signal")
    force_close_trades = sum(1 for t in trades if t.exit_reason == "force_close")
    stop_loss_fixed_ratio = (stop_loss_fixed_trades / max(stop_loss_trades, 1)) * 100.0
    stop_loss_vol_ratio = (stop_loss_vol_trades / max(stop_loss_trades, 1)) * 100.0

    # 止损率：触发止损的交易占总交易比例
    stop_loss_rate = (stop_loss_trades / max(n, 1)) * 100.0
    # 平均止损值：所有交易开仓时的ATR止损比例均值（负数，绝对值越大止损越宽）
    sl_values = [t.stop_loss_value for t in trades if t.stop_loss_value != 0.0]
    avg_stop_loss_value = float(np.mean(sl_values)) if sl_values else 0.0
    # 触发止损交易的平均实际亏损
    sl_pnls = [t.pnl for t in trades if t.exit_reason == "stop_loss"]
    avg_stop_loss_pnl = float(np.mean(sl_pnls)) if sl_pnls else 0.0

    profitable_symbols = sum(1 for s in symbol_stats.values() if s["pnl"] > 0)
    profitable_markets = sum(1 for s in market_stats.values() if s["pnl"] > 0)
    profitable_symbols_ratio = (profitable_symbols / max(len(symbol_stats), 1)) * 100.0
    profitable_markets_ratio = (profitable_markets / max(len(market_stats), 1)) * 100.0

    _tier_pnl_weights = {"tier1": 0.6, "tier2": 0.25, "tier3": 0.15}
    weighted_pnl = sum(
        tier_stats_raw.get(t, {}).get("pnl", 0.0) * w
        for t, w in _tier_pnl_weights.items()
    )

    return {
        "val_pnl": total_pnl,
        "weighted_pnl": float(weighted_pnl),
        "total_pnl": total_pnl,
        "val_trades": int(n),
        "num_trades": int(n),
        "num_long": int(n_long),
        "num_short": int(n_short),
        "win_rate": float((pnls > 0).mean() * 100.0),
        "expectancy": float(pnls.mean()),
        "median_pnl": float(np.median(pnls)),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "pnl_std": float(pnl_std),
        "max_win": float(pnls.max()),
        "max_loss": float(pnls.min()),
        "profit_factor": float(profit_factor),
        "win_loss_ratio": float(win_loss_ratio),
        "sortino": float(sortino),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "long_win_rate": float((np.asarray(long_pnls) > 0).mean() * 100.0) if long_pnls else 0.0,
        "short_win_rate": float((np.asarray(short_pnls) > 0).mean() * 100.0) if short_pnls else 0.0,
        "long_pnl": float(sum(long_pnls)),
        "short_pnl": float(sum(short_pnls)),
        "long_short_balance": float(1.0 - abs(n_long - n_short) / max(n, 1)),
        "long_short_ratio": float(n_long / max(n_short, 1)),
        "trades_per_1k_steps": float(trades_per_1k),
        "avg_holding": float(np.mean([t.holding_steps for t in trades])),
        "avg_holding_steps": float(np.mean([t.holding_steps for t in trades])),
        "avg_empty": float(np.mean([t.empty_steps for t in trades])),
        "avg_empty_steps": float(np.mean([t.empty_steps for t in trades])),
        "long_avg_holding": float(np.mean([t.holding_steps for t in long_trades])) if long_trades else 0.0,
        "avg_holding_long": float(np.mean([t.holding_steps for t in long_trades])) if long_trades else 0.0,
        "short_avg_holding": float(np.mean([t.holding_steps for t in short_trades])) if short_trades else 0.0,
        "avg_holding_short": float(np.mean([t.holding_steps for t in short_trades])) if short_trades else 0.0,
        "long_avg_empty": float(np.mean([t.empty_steps for t in long_trades])) if long_trades else 0.0,
        "short_avg_empty": float(np.mean([t.empty_steps for t in short_trades])) if short_trades else 0.0,
        "stop_loss_trades": int(stop_loss_trades),
        "stop_loss_fixed_trades": int(stop_loss_fixed_trades),
        "stop_loss_vol_trades": int(stop_loss_vol_trades),
        "stop_loss_fixed_ratio": float(stop_loss_fixed_ratio),
        "stop_loss_vol_ratio": float(stop_loss_vol_ratio),
        "stop_loss_rate": float(stop_loss_rate),
        "avg_stop_loss_value": float(avg_stop_loss_value),
        "avg_stop_loss_pnl": float(avg_stop_loss_pnl),
        "take_profit_trades": int(take_profit_trades),
        "close_signal_trades": int(close_signal_trades),
        "force_close_trades": int(force_close_trades),
        "active_symbols": int(len(symbol_stats)),
        "active_markets": int(len(market_stats)),
        "profitable_symbols_ratio": float(profitable_symbols_ratio),
        "profitable_markets_ratio": float(profitable_markets_ratio),
        "market_stats": market_stats,
        "symbol_stats": symbol_stats,
        "tier_stats": tier_stats,
    }

def _resolve_hf_snapshot(repo_id: str) -> Optional[str]:
    if not repo_id or "/" not in repo_id:
        return None
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    if not base.exists():
        return None
    ref_main = base / "refs" / "main"
    if ref_main.exists():
        rev = ref_main.read_text(encoding="utf-8").strip()
        if rev:
            snap = base / "snapshots" / rev
            if snap.exists():
                return str(snap)
    snap_dir = base / "snapshots"
    if not snap_dir.exists():
        return None
    candidates = sorted([p for p in snap_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def load_kron_teacher(cfg: Config, policy_device: torch.device) -> Optional[FrozenKronTeacher]:
    if not cfg.teacher.enabled:
        return None

    try:
        from kronos_vendor import Kronos, KronosTokenizer
    except Exception as e:
        log(f"[warn] failed to import kronos_vendor: {e}")
        return None

    teacher_device = choose_device(cfg.teacher.kron_device)
    if teacher_device.type == "cpu" and policy_device.type == "cuda":
        teacher_device = policy_device

    tokenizer_ref = cfg.teacher.kron_tokenizer_path or cfg.teacher.kron_tokenizer_repo
    model_ref = cfg.teacher.kron_model_path or cfg.teacher.kron_model_repo

    try:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_ref)
    except Exception as e:
        cached_tok = _resolve_hf_snapshot(cfg.teacher.kron_tokenizer_repo)
        if tokenizer_ref == cfg.teacher.kron_tokenizer_repo and cached_tok:
            log(f"[warn] tokenizer online load failed, fallback to cache: {cached_tok}")
            tokenizer = KronosTokenizer.from_pretrained(cached_tok)
        else:
            log(f"[warn] failed to load Kron tokenizer ({tokenizer_ref}): {e}")
            return None

    try:
        model = Kronos.from_pretrained(model_ref)
    except Exception as e:
        cached_model = _resolve_hf_snapshot(cfg.teacher.kron_model_repo)
        if model_ref == cfg.teacher.kron_model_repo and cached_model:
            log(f"[warn] model online load failed, fallback to cache: {cached_model}")
            model = Kronos.from_pretrained(cached_model)
        else:
            log(f"[warn] failed to load Kron model ({model_ref}): {e}")
            return None

    teacher = FrozenKronTeacher(tokenizer=tokenizer, model=model, cfg=cfg.teacher, device=teacher_device)
    n_tok = sum(p.numel() for p in teacher.tokenizer.parameters())
    n_model = sum(p.numel() for p in teacher.model.parameters())
    log(
        f"Kron teacher ready on {teacher_device}: "
        f"tokenizer/model params=({n_tok:,}/{n_model:,})"
    )
    return teacher


def _safe_symbol_filename(symbol: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in symbol)


def _segment_bounds(seg: Segment, split: str) -> Tuple[int, int]:
    if split == "train":
        return seg.train_start, seg.train_end
    if split == "val":
        return seg.val_start, seg.val_end
    if split == "test":
        return seg.test_start, seg.test_end
    raise ValueError(f"Unsupported split for cache: {split}")


def _cache_indices_for_symbol(
    data: SymbolData,
    seq_len: int,
    split_names: List[str],
    overlap_lookback: int,
) -> np.ndarray:
    n = len(data.prices)
    mask = np.zeros(n, dtype=np.bool_)
    for seg in data.segment_indices:
        for split in split_names:
            s, e = _segment_bounds(seg, split)
            overlap = _effective_overlap_lookback(seq_len, overlap_lookback, split)
            start = _decision_start_index(s, seq_len, overlap)
            end = min(e, n)
            if end > start:
                mask[start:end] = True
    return np.where(mask)[0].astype(np.int64)


def _load_cache_meta(cache_dir: Path) -> Optional[Dict[str, object]]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else None
    except Exception:
        return None


def _is_cache_meta_compatible(
    meta: Optional[Dict[str, object]],
    cfg: Config,
    split_names: List[str],
) -> Tuple[bool, str]:
    if not meta:
        return False, "missing meta.json"

    expected_overlap = _effective_overlap_lookback(
        cfg.data.seq_length,
        cfg.data.overlap_lookback,
        "val",
    )
    distill_mode = _teacher_distill_mode(cfg.teacher)
    expected = {
        "seq_length": int(cfg.data.seq_length),
        "splits": list(split_names),
        "train_ratio": float(cfg.data.train_ratio),
        "val_ratio": float(cfg.data.val_ratio),
        "overlap_lookback": int(expected_overlap),
        "distill_mode": distill_mode,
    }
    if distill_mode == "representation":
        expected["repr_dim"] = int(getattr(cfg.teacher, "repr_dim", 256))

    for key, expected_value in expected.items():
        actual = meta.get(key)
        if isinstance(expected_value, float):
            if actual is None or abs(float(actual) - expected_value) > 1e-8:
                return False, f"meta mismatch: {key}={actual} != {expected_value}"
        else:
            if actual != expected_value:
                return False, f"meta mismatch: {key}={actual} != {expected_value}"

    return True, "ok"


def _split_names_from_cfg(cfg: Config) -> List[str]:
    split_names = [s.strip().lower() for s in cfg.teacher.cache_splits if s.strip()]
    if not split_names:
        split_names = ["train", "val"]
    for split in split_names:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid teacher.cache_splits entry: {split}")
    return split_names


def _teacher_cache_dir(cfg: Config) -> Path:
    if _teacher_distill_mode(cfg.teacher) == "representation":
        return Path(getattr(cfg.teacher, "repr_cache_dir", "") or "")
    return Path(cfg.teacher.cache_dir)


def _cache_shape_for_symbol(cfg: Config, n: int) -> Tuple[int, ...]:
    if _teacher_distill_mode(cfg.teacher) == "representation":
        return (n, max(int(getattr(cfg.teacher, "repr_dim", 256)), 1))
    return (n,)


def _write_kron_cache_meta(
    cfg: Config,
    symbol_store: Dict[str, SymbolData],
    total_targets: int,
    total_built: int,
) -> None:
    cache_dir = _teacher_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_len = int(cfg.data.seq_length)
    distill_mode = _teacher_distill_mode(cfg.teacher)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seq_length": seq_len,
        "splits": _split_names_from_cfg(cfg),
        "train_ratio": float(cfg.data.train_ratio),
        "val_ratio": float(cfg.data.val_ratio),
        "overlap_lookback": int(_effective_overlap_lookback(seq_len, cfg.data.overlap_lookback, "val")),
        "distill_mode": distill_mode,
        "repr_dim": int(getattr(cfg.teacher, "repr_dim", 256)) if distill_mode == "representation" else 0,
        "symbols": list(symbol_store.keys()),
        "total_targets": int(total_targets),
        "total_built": int(total_built),
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _collect_pending_cache_jobs(
    cfg: Config,
    symbol_store: Dict[str, SymbolData],
) -> Tuple[List[Tuple[str, int]], int, int]:
    split_names = _split_names_from_cfg(cfg)
    seq_len = int(cfg.data.seq_length)
    cache_dir = _teacher_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected_repr = _teacher_distill_mode(cfg.teacher) == "representation"

    jobs: List[Tuple[str, int]] = []
    total_targets = 0
    pending_targets = 0
    for sym, data in symbol_store.items():
        path = cache_dir / f"{_safe_symbol_filename(sym)}.npy"
        n = len(data.prices)
        expected_shape = _cache_shape_for_symbol(cfg, n)
        arr: Optional[np.ndarray] = None
        if path.exists():
            try:
                loaded = np.load(path)
                if loaded.shape == expected_shape:
                    arr = loaded.astype(np.float16, copy=False)
            except Exception:
                arr = None
        if arr is None:
            arr = np.full(expected_shape, np.nan, dtype=np.float16)

        idx = _cache_indices_for_symbol(data, seq_len, split_names, int(cfg.data.overlap_lookback))
        total_targets += int(idx.size)
        if idx.size == 0:
            missing = idx
        elif expected_repr:
            missing = idx[np.isnan(arr[idx]).any(axis=1)]
        else:
            missing = idx[np.isnan(arr[idx])]
        if missing.size > 0:
            jobs.append((sym, int(missing.size)))
            pending_targets += int(missing.size)

    jobs.sort(key=lambda item: item[1], reverse=True)
    return jobs, int(total_targets), int(pending_targets)


def _shard_pending_cache_jobs(
    jobs: List[Tuple[str, int]],
    num_shards: int,
) -> Tuple[List[List[str]], List[int]]:
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    for sym, weight in jobs:
        shard_idx = min(range(num_shards), key=lambda idx: loads[idx])
        shards[shard_idx].append(sym)
        loads[shard_idx] += int(weight)
    return shards, loads


def load_kron_repr_cache(
    cfg: Config,
    symbol_store: Dict[str, SymbolData],
    device: torch.device,
) -> Optional[CachedKronReprTeacher]:
    if not cfg.teacher.use_offline_cache:
        return None
    cache_dir = Path(getattr(cfg.teacher, "repr_cache_dir", "") or "")
    if not cache_dir.exists():
        return None

    split_names = _split_names_from_cfg(cfg)
    meta = _load_cache_meta(cache_dir)
    meta_ok, meta_reason = _is_cache_meta_compatible(meta, cfg, split_names)
    if not meta_ok:
        log(f"Kron repr cache invalidated: {meta_reason}")
        return None

    repr_dim = max(int(getattr(cfg.teacher, "repr_dim", 256)), 1)
    for sym, data in symbol_store.items():
        path = cache_dir / f"{_safe_symbol_filename(sym)}.npy"
        if not path.exists():
            return None
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (len(data.prices), repr_dim):
            return None
    log(f"Kron repr cache loaded: {cache_dir}, symbols={len(symbol_store)}, dim={repr_dim}")
    return CachedKronReprTeacher(cache_dir, symbol_store, cfg.teacher, device=device)


def build_kron_repr_cache(
    cfg: Config,
    symbol_store: Dict[str, SymbolData],
    teacher: FrozenKronTeacher,
    symbols_subset: Optional[List[str]] = None,
    write_meta: bool = True,
) -> None:
    cache_dir = Path(getattr(cfg.teacher, "repr_cache_dir", "") or "")
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_names = _split_names_from_cfg(cfg)
    seq_len = int(cfg.data.seq_length)
    batch_size = max(int(cfg.teacher.cache_batch_size), 1)
    repr_dim = max(int(getattr(cfg.teacher, "repr_dim", 256)), 1)
    total_targets = 0
    total_built = 0
    selected_symbols = symbols_subset if symbols_subset is not None else list(symbol_store.keys())

    for sym in selected_symbols:
        data = symbol_store[sym]
        t0 = time.time()
        path = cache_dir / f"{_safe_symbol_filename(sym)}.npy"
        n = len(data.prices)

        arr: Optional[np.ndarray] = None
        if path.exists() and not bool(cfg.teacher.cache_force_rebuild):
            try:
                loaded = np.load(path)
                if loaded.shape == (n, repr_dim):
                    arr = loaded.astype(np.float16, copy=False)
            except Exception:
                arr = None
        if arr is None:
            arr = np.full((n, repr_dim), np.nan, dtype=np.float16)

        idx = _cache_indices_for_symbol(data, seq_len, split_names, int(cfg.data.overlap_lookback))
        total_targets += int(idx.size)
        if idx.size == 0:
            np.save(path, arr.astype(np.float16))
            continue

        missing_mask = np.isnan(arr[idx]).any(axis=1)
        missing = idx[missing_mask]
        if missing.size == 0:
            log(f"Kron repr cache hit {sym}: ready={int(idx.size)}")
            continue

        for start in range(0, int(missing.size), batch_size):
            chunk_idx = missing[start:start + batch_size]
            x = np.stack([data.raw_ohlcva[i - seq_len:i] for i in chunk_idx]).astype(np.float32)
            x_stamp = np.stack([data.time_features[i - seq_len:i] for i in chunk_idx]).astype(np.float32)
            y_stamp = np.stack([data.time_features[i:i + 1] for i in chunk_idx]).astype(np.float32)

            t_x = torch.from_numpy(x).to(teacher.device)
            t_x_stamp = torch.from_numpy(x_stamp).to(teacher.device)
            t_y_stamp = torch.from_numpy(y_stamp).to(teacher.device)
            reprs = teacher.predict_repr(t_x, t_x_stamp, t_y_stamp)
            if reprs.shape[-1] != repr_dim:
                raise ValueError(
                    f"Kron repr dim mismatch: teacher returned {reprs.shape[-1]}, cfg.teacher.repr_dim={repr_dim}"
                )
            arr[chunk_idx] = reprs.detach().cpu().numpy().astype(np.float16)

        np.save(path, arr.astype(np.float16))
        total_built += int(missing.size)
        log(
            f"Kron repr cache built {sym}: missing={int(missing.size)}/{int(idx.size)}, "
            f"time={time.time() - t0:.1f}s"
        )

    if write_meta:
        _write_kron_cache_meta(cfg, symbol_store, total_targets=total_targets, total_built=total_built)

    log(
        f"Kron repr cache summary: dir={cache_dir}, total_targets={total_targets}, "
        f"newly_built={total_built}, write_meta={write_meta}"
    )


def load_kron_strength_cache(cfg: Config, symbol_store: Dict[str, SymbolData]) -> Optional[Dict[str, np.ndarray]]:
    if not cfg.teacher.use_offline_cache:
        return None
    cache_dir = Path(cfg.teacher.cache_dir)
    if not cache_dir.exists():
        return None

    split_names = _split_names_from_cfg(cfg)

    meta = _load_cache_meta(cache_dir)
    meta_ok, meta_reason = _is_cache_meta_compatible(meta, cfg, split_names)
    if not meta_ok:
        log(f"Kron cache invalidated: {meta_reason}")
        return None

    seq_len = int(cfg.data.seq_length)
    out: Dict[str, np.ndarray] = {}
    for sym, data in symbol_store.items():
        path = cache_dir / f"{_safe_symbol_filename(sym)}.npy"
        if not path.exists():
            return None
        arr = np.load(path)
        if arr.shape != (len(data.prices),):
            return None
        idx = _cache_indices_for_symbol(data, seq_len, split_names, int(cfg.data.overlap_lookback))
        if idx.size > 0 and np.isnan(arr[idx]).any():
            return None
        out[sym] = arr.astype(np.float16, copy=False)
    log(f"Kron cache loaded: {cache_dir}, symbols={len(out)}, splits={split_names}")
    return out


def build_kron_strength_cache(
    cfg: Config,
    symbol_store: Dict[str, SymbolData],
    teacher: FrozenKronTeacher,
    symbols_subset: Optional[List[str]] = None,
    write_meta: bool = True,
) -> Dict[str, np.ndarray]:
    cache_dir = Path(cfg.teacher.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_names = _split_names_from_cfg(cfg)

    meta = _load_cache_meta(cache_dir)
    meta_ok, meta_reason = _is_cache_meta_compatible(meta, cfg, split_names)
    force_full_rebuild = bool(cfg.teacher.cache_force_rebuild) or not meta_ok
    if force_full_rebuild and meta_reason != "ok":
        log(f"Kron cache full rebuild: {meta_reason}")

    seq_len = int(cfg.data.seq_length)
    batch_size = max(int(cfg.teacher.cache_batch_size), 1)
    total_targets = 0
    total_built = 0
    strength_map: Dict[str, np.ndarray] = {}
    selected_symbols = symbols_subset if symbols_subset is not None else list(symbol_store.keys())

    for sym in selected_symbols:
        data = symbol_store[sym]
        t0 = time.time()
        path = cache_dir / f"{_safe_symbol_filename(sym)}.npy"
        n = len(data.prices)

        arr: Optional[np.ndarray] = None
        if path.exists() and not force_full_rebuild:
            try:
                loaded = np.load(path)
                if loaded.shape == (n,):
                    arr = loaded.astype(np.float16, copy=False)
            except Exception:
                arr = None
        if arr is None:
            arr = np.full((n,), np.nan, dtype=np.float16)

        idx = _cache_indices_for_symbol(data, seq_len, split_names, int(cfg.data.overlap_lookback))
        total_targets += int(idx.size)
        if idx.size == 0:
            strength_map[sym] = arr
            continue

        missing = idx[np.isnan(arr[idx])]
        if missing.size > 0:
            for start in range(0, int(missing.size), batch_size):
                chunk_idx = missing[start:start + batch_size]
                x = np.stack([data.raw_ohlcva[i - seq_len:i] for i in chunk_idx]).astype(np.float32)
                x_stamp = np.stack([data.time_features[i - seq_len:i] for i in chunk_idx]).astype(np.float32)
                y_stamp = np.stack([data.time_features[i:i + 1] for i in chunk_idx]).astype(np.float32)

                t_x = torch.from_numpy(x).to(teacher.device)
                t_x_stamp = torch.from_numpy(x_stamp).to(teacher.device)
                t_y_stamp = torch.from_numpy(y_stamp).to(teacher.device)
                strengths = teacher.predict_strength(t_x, t_x_stamp, t_y_stamp)
                arr[chunk_idx] = strengths.detach().cpu().numpy().astype(np.float16)

            np.save(path, arr.astype(np.float16))
            total_built += int(missing.size)
            log(
                f"Kron cache built {sym}: missing={int(missing.size)}/{int(idx.size)}, "
                f"time={time.time() - t0:.1f}s"
            )
        else:
            log(f"Kron cache hit {sym}: ready={int(idx.size)}")

        strength_map[sym] = arr

    if write_meta:
        _write_kron_cache_meta(cfg, symbol_store, total_targets=total_targets, total_built=total_built)

    log(
        f"Kron cache summary: dir={cache_dir}, total_targets={total_targets}, "
        f"newly_built={total_built}, write_meta={write_meta}"
    )
    return strength_map


def load_student_init(
    student_policy: StudentPolicy,
    student_value: StudentValue,
    checkpoint_path: str,
    strict: bool,
) -> bool:
    if not checkpoint_path:
        return False

    p = Path(checkpoint_path)
    if not p.exists():
        log(f"[warn] student init checkpoint not found: {checkpoint_path}")
        return False

    ckpt = torch.load(p, map_location="cpu")
    if not isinstance(ckpt, dict):
        log(f"[warn] unsupported student init format: {checkpoint_path}")
        return False

    loaded = False
    if "policy" in ckpt and isinstance(ckpt["policy"], dict):
        student_policy.load_state_dict(ckpt["policy"], strict=strict)
        loaded = True
    elif "policy_state_dict" in ckpt and isinstance(ckpt["policy_state_dict"], dict):
        student_policy.load_state_dict(ckpt["policy_state_dict"], strict=strict)
        loaded = True
    elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        student_policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        loaded = True

    if "value" in ckpt and isinstance(ckpt["value"], dict):
        student_value.load_state_dict(ckpt["value"], strict=False)

    if loaded:
        log(f"Student initialized from {checkpoint_path}")
    else:
        log(f"[warn] no policy weights found in {checkpoint_path}")

    return loaded


def build_email_notifier(cfg: Config):
    if not cfg.email.enabled:
        return None
    try:
        from core.email_notifier import EmailNotifier, create_email_notifier
    except Exception as e:
        log(f"[warn] failed to import core.email_notifier: {e}")
        return None

    notifier = None
    config_path = cfg.email.config_path.strip()
    if config_path and Path(config_path).exists():
        try:
            notifier = create_email_notifier(config_path)
            log(f"Email notifier loaded from config file: {config_path}")
        except Exception as e:
            log(f"[warn] failed to load email config from {config_path}: {e}")
            notifier = None

    if notifier is None:
        notifier = EmailNotifier(
            smtp_server=cfg.email.smtp_server,
            smtp_port=cfg.email.smtp_port,
            sender_email=cfg.email.sender_email,
            sender_password=cfg.email.sender_password,
            recipient_email=cfg.email.recipient_email,
            enabled=cfg.email.enabled,
        )
        log("Email notifier initialized from runtime config.")

    if not getattr(notifier, "enabled", False):
        log("[warn] email notifier disabled due to incomplete/invalid settings.")
        return None

    log(f"Email notifier enabled, recipient={cfg.email.recipient_email}")
    return notifier


# -----------------------------
# Main
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_str.startswith("cuda"):
        idx = int(device_str.split(":")[1]) if ":" in device_str else 0
        if idx >= torch.cuda.device_count():
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end PPO trainer with frozen teacher regularization")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max-updates", type=int, default=-1, help="Override ppo.max_updates")
    parser.add_argument("--cache-only", action="store_true", help="Only build offline Kron cache for the assigned shard")
    parser.add_argument("--cache-meta-only", action="store_true", help="Only validate cache completeness and refresh meta.json")
    parser.add_argument("--cache-device", type=str, default="", help="Override teacher.kron_device for cache-only mode")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of cache shards in cache-only mode")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index in cache-only mode")
    return parser.parse_args()


def configure_policy_phase(policy: StudentPolicy, mode: str) -> Tuple[int, int]:
    mode = (mode or "").strip().lower()

    if mode == "stage1_policy":
        for name, p in policy.named_parameters():
            p.requires_grad = not name.startswith("backbone.")
    elif mode == "stage2_mamba":
        for name, p in policy.named_parameters():
            p.requires_grad = name.startswith("backbone.")
    elif mode == "stage3_joint":
        for p in policy.parameters():
            p.requires_grad = True
    else:
        raise ValueError(
            f"Unsupported phase.mode={mode}. "
            "Use stage1_policy | stage2_mamba | stage3_joint."
        )

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    return trainable, frozen


def run_cache_only(cfg: Config, args: argparse.Namespace) -> None:
    if args.cache_device:
        cfg.teacher.kron_device = args.cache_device

    symbol_store = build_symbol_store(cfg)
    jobs, total_targets, pending_targets = _collect_pending_cache_jobs(cfg, symbol_store)
    log(
        f"Cache status: ready_symbols={len(symbol_store) - len(jobs)}/{len(symbol_store)}, "
        f"pending_symbols={len(jobs)}, pending_targets={pending_targets}/{total_targets}"
    )

    if args.cache_meta_only:
        if pending_targets > 0:
            raise RuntimeError(
                f"Cannot refresh cache meta while cache incomplete: "
                f"pending_symbols={len(jobs)}, pending_targets={pending_targets}"
            )
        _write_kron_cache_meta(cfg, symbol_store, total_targets=total_targets, total_built=0)
        log(f"Kron cache meta refreshed: dir={_teacher_cache_dir(cfg)}")
        return

    if not cfg.teacher.enabled:
        raise RuntimeError("teacher.enabled must be true in cache-only mode")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be within [0, num_shards)")

    shards, loads = _shard_pending_cache_jobs(jobs, args.num_shards)
    assigned_symbols = shards[args.shard_index]
    assigned_targets = loads[args.shard_index]
    log(
        f"Cache shard {args.shard_index}/{args.num_shards}: device={cfg.teacher.kron_device}, "
        f"symbols={len(assigned_symbols)}, pending_targets={assigned_targets}"
    )
    if not assigned_symbols:
        log("Cache shard already complete; nothing to build")
        return

    cache_device = choose_device(cfg.teacher.kron_device)
    teacher = load_kron_teacher(cfg, cache_device)
    if teacher is None:
        raise RuntimeError(f"Failed to initialize Kron teacher on {cfg.teacher.kron_device}")

    if _teacher_distill_mode(cfg.teacher) == "representation":
        build_kron_repr_cache(
            cfg,
            symbol_store,
            teacher,
            symbols_subset=assigned_symbols,
            write_meta=False,
        )
    else:
        build_kron_strength_cache(
            cfg,
            symbol_store,
            teacher,
            symbols_subset=assigned_symbols,
            write_meta=False,
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config if args.config else None)

    if args.resume:
        cfg.run.resume = args.resume
    if args.max_updates > 0:
        cfg.ppo.max_updates = int(args.max_updates)

    if args.cache_only or args.cache_meta_only:
        run_cache_only(cfg, args)
        return

    set_seed(cfg.run.seed)

    policy_device = choose_device(cfg.run.policy_device)
    value_device = choose_device(cfg.run.value_device)

    log("=" * 88)
    log("E2E PPO: Frozen Teacher + Trainable Student")
    log("=" * 88)
    log(
        f"devices policy/value=({policy_device}/{value_device}), "
        f"seq={cfg.data.seq_length}, max_updates={cfg.ppo.max_updates}, "
        f"teacher_enabled={cfg.teacher.enabled}"
    )

    symbol_store = build_symbol_store(cfg)
    n_assets = len(symbol_store)
    symbol_list = list(symbol_store.keys())
    email_notifier = build_email_notifier(cfg)

    policy = StudentPolicy(cfg, n_assets=n_assets).to(policy_device)
    value = StudentValue(cfg, n_assets=n_assets).to(value_device)

    # Optional frozen Kron teacher (online or offline-cached)
    teacher: Optional[object] = None
    teacher_mode = _teacher_distill_mode(cfg.teacher)
    if cfg.teacher.enabled and cfg.teacher.use_offline_cache:
        if teacher_mode == "representation":
            teacher = load_kron_repr_cache(cfg, symbol_store, device=policy_device)
            if teacher is None:
                teacher_online = load_kron_teacher(cfg, policy_device)
                if teacher_online is not None:
                    build_kron_repr_cache(cfg, symbol_store, teacher_online)
                    del teacher_online
                    if policy_device.type == "cuda":
                        torch.cuda.empty_cache()
                teacher = load_kron_repr_cache(cfg, symbol_store, device=policy_device)
            if teacher is not None:
                log(
                    f"Using offline Kron repr cache teacher: dir={cfg.teacher.repr_cache_dir}, "
                    f"splits={cfg.teacher.cache_splits}, dim={cfg.teacher.repr_dim}"
                )
        else:
            cache_map = load_kron_strength_cache(cfg, symbol_store)
            if cache_map is None:
                teacher_online = load_kron_teacher(cfg, policy_device)
                if teacher_online is not None:
                    cache_map = build_kron_strength_cache(cfg, symbol_store, teacher_online)
                    del teacher_online
                    if policy_device.type == "cuda":
                        torch.cuda.empty_cache()
            if cache_map is not None:
                teacher = CachedKronTeacher(cache_map, cfg.teacher, device=policy_device)
                log(
                    f"Using offline Kron cache teacher: dir={cfg.teacher.cache_dir}, "
                    f"splits={cfg.teacher.cache_splits}"
                )
    elif cfg.teacher.enabled:
        teacher = load_kron_teacher(cfg, policy_device)

    if cfg.teacher.enabled and teacher is None:
        cfg.teacher.enabled = False
        log("Teacher disabled automatically (Kron unavailable/cache unavailable)")

    # Optional student init
    load_student_init(policy, value, cfg.student_init.checkpoint, cfg.student_init.strict)
    policy_trainable, policy_frozen = configure_policy_phase(policy, cfg.phase.mode)

    n_policy = sum(p.numel() for p in policy.parameters())
    n_value = sum(p.numel() for p in value.parameters())
    log(
        f"student params policy/value=({n_policy:,}/{n_value:,}), "
        f"phase={cfg.phase.mode}, policy trainable/frozen=({policy_trainable:,}/{policy_frozen:,})"
    )

    trainer = PPOTrainer(
        policy=policy,
        value=value,
        cfg=cfg,
        policy_device=policy_device,
        value_device=value_device,
        teacher_enabled=teacher is not None,
    )
    collector = Collector(cfg, symbol_store, policy_device, teacher=teacher)

    # Initialize EMA for policy if enabled
    policy_ema = None
    if bool(getattr(cfg.ppo, "use_policy_ema", False)):
        try:
            from core.ema import EMA
            ema_decay = float(getattr(cfg.ppo, "policy_ema_decay", 0.999))
            policy_ema = EMA(policy, decay=ema_decay)
            log(f"Policy EMA enabled: decay={ema_decay}")
        except ImportError:
            log("[warn] EMA module not found, disabling EMA")
            policy_ema = None

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = output_dir / "latest.pt"
    best_ckpt = output_dir / "best.pt"

    start_update = 0
    best_val_pnl = -float("inf")

    resume_path = Path(cfg.run.resume) if cfg.run.resume else (latest_ckpt if latest_ckpt.exists() else None)
    if resume_path is not None and resume_path.exists():
        state = load_checkpoint(resume_path)
        if state is not None:
            trainer.load_state_dict(state)
            start_update = trainer.update_count
            best_val_pnl = float(state.get("best_val_pnl", best_val_pnl))
            log(f"Resumed from {resume_path}, update={start_update}, best_val_pnl={best_val_pnl:.6f}")

    log(f"Start training from update={start_update}")

    for update in range(start_update, cfg.ppo.max_updates):
        t0 = time.time()

        policy.eval()
        collect_t0 = time.time()
        rollout, train_trades = collector.collect(policy, mode="train", update_count=update)
        collect_s = time.time() - collect_t0
        stats = trainer.update(rollout)

        # Update EMA after training step
        if policy_ema is not None:
            policy_ema.update()

        train_metrics = compute_trade_metrics(
            train_trades,
            symbols=symbol_list,
            val_steps=int(stats.get("n_steps", 0)) if isinstance(stats, dict) else 0,
        ) if isinstance(stats, dict) and not stats.get("skipped") else {}

        if stats.get("skipped"):
            log(f"Update {update}: skipped ({stats.get('reason', 'unknown')})")
            continue

        elapsed = time.time() - t0
        if update % cfg.ppo.log_interval == 0:
            log(
                f"Update {update}: steps={stats['n_steps']}, trades={len(train_trades)}, "
                f"p_loss={stats['policy_loss']:.4f}, v_loss={stats['value_loss']:.4f}, "
                f"ent={stats['entropy']:.4f}, kl={stats['kl']:.4f}, t_kl={stats['teacher_kl']:.4f}, "
                f"s_loss={stats['strength_loss']:.4f}, repr={stats['repr_loss']:.4f}, t_coef={stats['teacher_coef']:.4f}, "
                f"s_coef={stats['strength_coef']:.4f}, repr_coef={stats['repr_coef']:.4f}, reward={stats['reward_mean']:+.4f}±{stats['reward_std']:.4f}, "
                f"adv={stats['adv_mean']:+.4f}±{stats['adv_std']:.4f}, value={stats['value_mean']:+.4f}±{stats['value_std']:.4f}, "
                f"r(open/hold/close)=({stats['reward_open_mean']:.4f}/{stats['reward_hold_mean']:.4f}/{stats['reward_close_mean']:.4f}), "
                f"time(c/u/g/b/v/p)=({collect_s:.1f}/{stats['update_compute_s']:.1f}/{stats['gae_s']:.1f}/"
                f"{stats['batch_build_s']:.1f}/{stats['value_train_s']:.1f}/{stats['policy_train_s']:.1f}), "
                f"lr(p/v)=({stats['policy_lr']:.2e}/{stats['value_lr']:.2e}), {elapsed:.1f}s"
            )
            if train_metrics:
                _log_trade_metrics("Train", train_metrics, include_symbol_detail=False)

        # periodic validation - only use ValRandom
        if update > 0 and update % cfg.ppo.val_interval == 0:
            policy.eval()
            val_random_envs = int(getattr(cfg.ppo, "val_random_envs", 0) or cfg.ppo.val_envs)
            val_random_traj = int(getattr(cfg.ppo, "val_random_trajectory_length", 0) or cfg.ppo.val_trajectory_length)
            val_random_steps = val_random_envs * val_random_traj

            val_random_trades = collector.collect(
                policy,
                mode="val",
                update_count=update,
                deterministic_eval=False,
                fixed_seed=None,
            )
            val_random_metrics = compute_trade_metrics(val_random_trades, symbols=symbol_list, val_steps=val_random_steps)
            val_pnl = float(val_random_metrics["weighted_pnl"])

            # Evaluate EMA model if enabled
            val_ema_metrics = None
            if policy_ema is not None:
                policy_ema.apply_shadow()
                val_ema_trades = collector.collect(policy, mode="val", update_count=update, deterministic_eval=False, fixed_seed=None)
                policy_ema.restore()
                val_ema_metrics = compute_trade_metrics(val_ema_trades, symbols=symbol_list, val_steps=val_random_steps)
                val_ema_pnl = float(val_ema_metrics["weighted_pnl"])
                log(f"ValEMA {update}: pnl={val_ema_metrics['val_pnl']:+.4f}, weighted_pnl={val_ema_metrics['weighted_pnl']:+.4f}, trades={val_ema_metrics['val_trades']}, win={val_ema_metrics['win_rate']:.1f}%, sharpe={val_ema_metrics['sharpe']:.3f}, DD={val_ema_metrics['max_drawdown']:.4f}")
                if val_ema_pnl > val_pnl:
                    val_pnl = val_ema_pnl
                    val_random_metrics = val_ema_metrics

            is_best = val_pnl > best_val_pnl
            if is_best:
                best_val_pnl = val_pnl

            log(f"ValRandom {update}: pnl={val_random_metrics['val_pnl']:+.4f}, weighted_pnl={val_random_metrics['weighted_pnl']:+.4f}, trades={val_random_metrics['val_trades']}, win={val_random_metrics['win_rate']:.1f}% (L:{val_random_metrics['long_win_rate']:.1f}%/S:{val_random_metrics['short_win_rate']:.1f}%), exp={val_random_metrics['expectancy']:+.6f}, med={val_random_metrics['median_pnl']:+.6f}, PF={val_random_metrics['profit_factor']:.2f}, W/L={val_random_metrics['win_loss_ratio']:.2f}, sharpe={val_random_metrics['sharpe']:.3f}, sortino={val_random_metrics['sortino']:.3f}, DD={val_random_metrics['max_drawdown']:.4f}, best={best_val_pnl:+.4f}")

            # Log tier stats
            _tier_parts = []
            _tier_w_map = {"tier1": 0.6, "tier2": 0.25, "tier3": 0.15}
            _weighted_pnl = 0.0
            _weight_sum = 0.0
            tier_stats_raw = val_random_metrics.get("tier_stats", {})
            for tier_key in ["tier1", "tier2", "tier3"]:
                _ts = tier_stats_raw.get(tier_key, {})
                _pnl = _ts.get("pnl", 0.0)
                _wr = _ts.get("win_rate", 0.0)
                _n = _ts.get("trades", 0)
                _tier_parts.append(f"{tier_key}: pnl={_pnl:+.2f} win={_wr:.1f}% sharpe={_ts.get('sharpe', 0.0):.3f} n={_n}")
                _w = _tier_w_map.get(tier_key, 0.0)
                _weighted_pnl += _pnl * _w
                _weight_sum += _w
            _weighted_pnl /= max(_weight_sum, 1e-8)
            log(f"  TierStats: {' | '.join(_tier_parts)} | weighted_pnl={_weighted_pnl:+.4f}")
            _log_trade_metrics("ValRandom", val_random_metrics, include_symbol_detail=is_best)

            state = trainer.state_dict()
            state["best_val_pnl"] = best_val_pnl
            state["val_metrics"] = val_random_metrics
            state["val_random_metrics"] = val_random_metrics
            if val_ema_metrics is not None:
                state["val_ema_metrics"] = val_ema_metrics
            if policy_ema is not None:
                state["policy_ema"] = policy_ema.state_dict()
            state["update_count_ckpt"] = update
            save_checkpoint(latest_ckpt, state)
            if is_best:
                val_random_metrics["best_checkpoint"] = str(best_ckpt)
                if policy_ema is not None and val_ema_metrics is not None and val_ema_pnl >= float(val_random_metrics["val_pnl"]):
                    policy_ema.apply_shadow()
                    save_checkpoint(best_ckpt, state)
                    policy_ema.restore()
                    log(f"Saved new best checkpoint (EMA): {best_ckpt}")
                else:
                    save_checkpoint(best_ckpt, state)
                    log(f"Saved new best checkpoint: {best_ckpt}")
                _log_trade_metrics("NewBest", val_random_metrics, include_symbol_detail=True)
                if email_notifier is not None:
                    try:
                        sent = email_notifier.send_new_best_notification(update_count=update, score=val_pnl, metrics=val_random_metrics)
                        log(f"New-best email sent={bool(sent)}")
                    except Exception as e:
                        log(f"[warn] email notification failed: {e}")

        # periodic latest save
        if update > 0 and update % cfg.ppo.save_interval == 0:
            state = trainer.state_dict()
            state["best_val_pnl"] = best_val_pnl
            state["update_count_ckpt"] = update
            save_checkpoint(latest_ckpt, state)

    # final save
    state = trainer.state_dict()
    state["best_val_pnl"] = best_val_pnl
    state["update_count_ckpt"] = cfg.ppo.max_updates
    save_checkpoint(latest_ckpt, state)

    log("Training complete.")
    log(f"Best val pnl: {best_val_pnl:.6f}")
    log(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
