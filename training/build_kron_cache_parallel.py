#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("BIGMODEL_SKIP_MAMBA_IMPORT", "1")

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from train_e2e_ppo import (
    Config,
    SymbolData,
    _cache_indices_for_symbol,
    _effective_overlap_lookback,
    _safe_symbol_filename,
    build_symbol_store,
    choose_device,
    load_config,
    load_kron_teacher,
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def split_names_from_cfg(cfg: Config) -> List[str]:
    split_names = [s.strip().lower() for s in cfg.teacher.cache_splits if s.strip()]
    if not split_names:
        split_names = ["train", "val"]
    for split in split_names:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid teacher.cache_splits entry: {split}")
    return split_names


def load_or_init_cache_array(path: Path, n: int) -> np.ndarray:
    if path.exists():
        try:
            arr = np.load(path)
            if arr.shape == (n,):
                return arr.astype(np.float16, copy=False)
        except Exception:
            pass
    return np.full((n,), np.nan, dtype=np.float16)


def cache_missing_indices(cfg: Config, data: SymbolData, split_names: Sequence[str], arr: np.ndarray) -> np.ndarray:
    idx = _cache_indices_for_symbol(
        data,
        int(cfg.data.seq_length),
        list(split_names),
        int(cfg.data.overlap_lookback),
    )
    if idx.size == 0:
        return idx
    return idx[np.isnan(arr[idx])]


def collect_jobs(cfg: Config, symbol_store: Dict[str, SymbolData]) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    cache_dir = Path(cfg.teacher.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_names = split_names_from_cfg(cfg)

    jobs: List[Tuple[str, int]] = []
    total_targets = 0
    ready_symbols = 0
    pending_targets = 0

    for sym, data in symbol_store.items():
        path = cache_dir / f"{_safe_symbol_filename(sym)}.npy"
        arr = load_or_init_cache_array(path, len(data.prices))
        idx = _cache_indices_for_symbol(data, int(cfg.data.seq_length), split_names, int(cfg.data.overlap_lookback))
        total_targets += int(idx.size)
        missing = idx[np.isnan(arr[idx])] if idx.size > 0 else idx
        if missing.size == 0:
            ready_symbols += 1
            continue
        jobs.append((sym, int(missing.size)))
        pending_targets += int(missing.size)

    jobs.sort(key=lambda item: item[1], reverse=True)
    stats = {
        "symbols_total": len(symbol_store),
        "symbols_ready": ready_symbols,
        "symbols_pending": len(jobs),
        "targets_total": int(total_targets),
        "targets_pending": int(pending_targets),
    }
    return jobs, stats


def shard_jobs_greedy(jobs: Sequence[Tuple[str, int]], num_shards: int) -> Tuple[List[List[str]], List[int]]:
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    for sym, weight in jobs:
        shard_idx = min(range(num_shards), key=lambda idx: loads[idx])
        shards[shard_idx].append(sym)
        loads[shard_idx] += int(weight)
    return shards, loads


def atomic_save_array(path: Path, arr: np.ndarray) -> None:
    tmp_path = path.parent / f".{path.stem}.tmp.{os.getpid()}.npy"
    np.save(tmp_path, arr.astype(np.float16, copy=False))
    os.replace(tmp_path, path)


def write_meta(cfg: Config, symbol_store: Dict[str, SymbolData], total_built: int) -> None:
    split_names = split_names_from_cfg(cfg)
    seq_len = int(cfg.data.seq_length)
    total_targets = 0
    for data in symbol_store.values():
        idx = _cache_indices_for_symbol(data, seq_len, split_names, int(cfg.data.overlap_lookback))
        total_targets += int(idx.size)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seq_length": seq_len,
        "splits": split_names,
        "train_ratio": float(cfg.data.train_ratio),
        "val_ratio": float(cfg.data.val_ratio),
        "overlap_lookback": int(_effective_overlap_lookback(seq_len, cfg.data.overlap_lookback, "val")),
        "symbols": list(symbol_store.keys()),
        "total_targets": int(total_targets),
        "total_built": int(total_built),
    }
    cache_dir = Path(cfg.teacher.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remaining Kron cache in parallel across independent GPU processes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--write-meta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbol_store = build_symbol_store(cfg)
    jobs, stats = collect_jobs(cfg, symbol_store)

    log(
        "Cache status: "
        f"ready_symbols={stats['symbols_ready']}/{stats['symbols_total']}, "
        f"pending_symbols={stats['symbols_pending']}, "
        f"pending_targets={stats['targets_pending']}/{stats['targets_total']}"
    )

    if args.status_only:
        return

    if args.write_meta:
        if jobs:
            raise RuntimeError(
                f"Cannot write meta while cache incomplete: pending_symbols={stats['symbols_pending']}, "
                f"pending_targets={stats['targets_pending']}"
            )
        write_meta(cfg, symbol_store, total_built=0)
        log("meta.json refreshed")
        return

    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if not args.device:
        raise ValueError("--device is required for build mode")

    shards, loads = shard_jobs_greedy(jobs, args.num_shards)
    assigned = shards[args.shard_index]
    assigned_load = loads[args.shard_index]
    log(
        f"Shard {args.shard_index}/{args.num_shards} on {args.device}: "
        f"symbols={len(assigned)}, pending_targets={assigned_load}"
    )

    if not assigned:
        log("No pending symbols assigned to this shard")
        return

    cfg.teacher.kron_device = args.device
    torch.set_num_threads(1)
    teacher_device = choose_device(args.device)
    teacher = load_kron_teacher(cfg, teacher_device)
    if teacher is None:
        raise RuntimeError(f"failed to initialize Kron teacher on {args.device}")
    log(f"Worker teacher device resolved to {teacher.device}")

    split_names = split_names_from_cfg(cfg)
    seq_len = int(cfg.data.seq_length)
    batch_size = max(int(cfg.teacher.cache_batch_size), 1)
    cache_dir = Path(cfg.teacher.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    built_symbols = 0
    built_targets = 0
    started = time.time()
    for index, sym in enumerate(assigned, start=1):
        data = symbol_store[sym]
        path = cache_dir / f"{_safe_symbol_filename(sym)}.npy"
        arr = load_or_init_cache_array(path, len(data.prices))
        missing = cache_missing_indices(cfg, data, split_names, arr)
        if missing.size == 0:
            continue

        t0 = time.time()
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

        atomic_save_array(path, arr)
        built_symbols += 1
        built_targets += int(missing.size)
        log(
            f"[{args.device}] cache built {sym}: missing={int(missing.size)}, "
            f"symbol={index}/{len(assigned)}, time={time.time() - t0:.1f}s"
        )

    elapsed = time.time() - started
    log(
        f"Shard done on {args.device}: built_symbols={built_symbols}, built_targets={built_targets}, "
        f"elapsed={elapsed/60.0:.1f}m"
    )


if __name__ == "__main__":
    main()
