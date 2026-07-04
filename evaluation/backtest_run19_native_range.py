#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from compare_run19_native_vs_freqtrade_202509_202605 import (
    MAX_STEPS,
    SEQ_LEN,
    build_symbol_store_from_feathers,
    pair_to_feather_name,
    pair_to_symbol,
    run_native,
    summarize,
)


def load_pairs(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if not isinstance(pairs, list):
        raise ValueError(f"Expected pair list in {path}")
    return [str(p) for p in pairs]


def load_complete_ohlcv(
    pairs: list[str],
    datadir: Path,
    start: str,
    end: str,
    seq_len: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    warmup_start = start_ts - pd.Timedelta(hours=4 * seq_len)
    expected_decision_bars = int((end_ts - start_ts) / pd.Timedelta(hours=4))
    last_required = end_ts - pd.Timedelta(hours=4)

    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}

    for pair in pairs:
        symbol = pair_to_symbol(pair)
        path = datadir / pair_to_feather_name(pair)
        if not path.exists():
            rows.append(
                {
                    "pair": pair,
                    "symbol": symbol,
                    "status": "missing_feather",
                    "data_start": "",
                    "data_end": "",
                    "bars_loaded": 0,
                    "decision_bars": 0,
                }
            )
            continue

        df = pd.read_feather(path)
        if "date" not in df.columns:
            raise ValueError(f"{path} has no date column")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        data_start = df["date"].min()
        data_end = df["date"].max()

        df = df[(df["date"] >= warmup_start) & (df["date"] < end_ts)].copy()
        df = df.sort_values("date").reset_index(drop=True)
        decision_bars = int(((df["date"] >= start_ts) & (df["date"] < end_ts)).sum())

        status = "ok"
        if len(df) < seq_len + 2 or decision_bars < 2:
            status = "insufficient_timerange_data"
        elif data_start > warmup_start:
            status = "partial_warmup"
        elif data_end < last_required:
            status = "incomplete_end"
        elif decision_bars != expected_decision_bars:
            status = "missing_decision_bars"

        rows.append(
            {
                "pair": pair,
                "symbol": symbol,
                "status": status,
                "data_start": str(data_start) if pd.notna(data_start) else "",
                "data_end": str(data_end) if pd.notna(data_end) else "",
                "bars_loaded": len(df),
                "decision_bars": decision_bars,
            }
        )
        if status == "ok":
            frames[symbol] = df

    return frames, pd.DataFrame(rows)


def write_native_report(
    path: Path,
    native_summary: pd.DataFrame,
    availability: pd.DataFrame,
    start: str,
    end: str,
    days: float,
) -> None:
    overall = summarize("Run19 Native", native_summary, days)
    unavailable = availability[availability["status"].ne("ok")]
    per_pair = native_summary.sort_values("pnl_pct_sum", ascending=False)

    lines = [
        "# Run19 Native Backtest",
        "",
        f"- Decision window: `{start}` to `{end}`",
        "- Observation window: `180 x 4h` candles before the decision start, excluded from results",
        f"- Effective days: `{days:.2f}`",
        f"- Complete pairs loaded: `{overall['symbols']}` / `{len(availability)}` requested",
        "",
        "## Overall",
        "",
        "| Symbols | Symbols with trades | PNL sum | Annualized PNL sum | Trades | Win rate |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['symbols']} | {overall['symbols_with_trades']} | "
            f"{overall['pnl_pct_sum']:.2f}% | {overall['annualized_pct_sum']:.2f}% | "
            f"{overall['trades']} | {overall['win_rate'] * 100:.2f}% |"
        ),
        "",
        "## Top Pairs",
        "",
        per_pair.head(15).to_markdown(index=False),
        "",
        "## Bottom Pairs",
        "",
        per_pair.tail(15).to_markdown(index=False),
        "",
        "## Unavailable Or Incomplete Pairs",
        "",
        unavailable[["pair", "status", "data_start", "data_end", "bars_loaded", "decision_bars"]]
        .to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="top100_pairs.json")
    parser.add_argument("--datadir", default="user_data/data/binance/futures")
    parser.add_argument("--cfg", default="configs/e2e_ppo_run19_sl15_valrandom_seq180.json")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt")
    parser.add_argument("--start", default="2026-05-01 00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-08 00:00:00+00:00")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", default="run19_native_20260501_20260608")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    frames, availability = load_complete_ohlcv(
        pairs=pairs,
        datadir=Path(args.datadir),
        start=args.start,
        end=args.end,
        seq_len=SEQ_LEN,
    )
    symbol_store = build_symbol_store_from_feathers(frames, args.start, args.end, SEQ_LEN)
    if not symbol_store:
        raise RuntimeError("No symbols have complete data for the requested decision window")

    native_summary, native_trades, native_executable_summary = run_native(
        cfg_path=args.cfg,
        checkpoint=args.checkpoint,
        symbol_store=symbol_store,
        frames=frames,
        device_name=args.device,
        max_steps=MAX_STEPS,
    )

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    days = (end_ts - start_ts).total_seconds() / 86400.0

    availability.to_csv(outdir / "availability.csv", index=False)
    native_summary.to_csv(outdir / "native_summary.csv", index=False)
    native_executable_summary.to_csv(outdir / "native_freqtrade_executable_summary.csv", index=False)
    native_trades.to_csv(outdir / "native_trades.csv", index=False)
    write_native_report(outdir / "report.md", native_summary, availability, args.start, args.end, days)

    result = {
        "outdir": str(outdir),
        "decision_window": {"start": args.start, "end": args.end},
        "observation_window_bars_excluded": SEQ_LEN,
        "requested_pairs": len(availability),
        "complete_pairs_loaded": len(symbol_store),
        "incomplete_or_unavailable_pairs": int(availability["status"].ne("ok").sum()),
        "native": summarize("Run19 Native", native_summary, days),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
