#!/usr/bin/env python3
"""Grid-scan run19 stop-loss settings on the aligned 2025-09..2026-05 set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from compare_run19_native_vs_freqtrade_202509_202605 import (
    DEFAULT_TIMERANGE,
    REQUESTED_END,
    REQUESTED_START,
    SEQ_LEN,
    build_symbol_store_from_feathers,
    load_common_ohlcv,
    load_pairs,
    symbol_to_pair,
)
from train_e2e_ppo import (
    StudentPolicy,
    TradingEnv,
    _market_group_from_metadata,
    load_config,
    set_seed,
)


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _run_one(
    cfg_path: str,
    checkpoint: str,
    frames: dict[str, pd.DataFrame],
    device_name: str,
    max_steps: int,
    stop_loss: float,
    stop_loss_vol_multiple: float,
    take_profit: float,
) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    cfg.data.seq_length = SEQ_LEN
    cfg.data.overlap_lookback = SEQ_LEN
    cfg.env.stop_loss = float(stop_loss)
    cfg.env.stop_loss_vol_multiple = float(stop_loss_vol_multiple)
    cfg.env.take_profit = float(take_profit)
    set_seed(cfg.run.seed)

    symbol_store = build_symbol_store_from_feathers(
        frames=frames,
        start=REQUESTED_START,
        end=REQUESTED_END,
        seq_len=SEQ_LEN,
    )
    symbols = list(symbol_store.keys())
    symbol_to_asset_id = {sym: idx for idx, sym in enumerate(symbols)}
    dates_by_symbol = {
        sym: pd.to_datetime(frames[sym]["date"], utc=True).reset_index(drop=True)
        for sym in symbols
        if sym in frames
    }

    device = torch.device(device_name)
    policy = StudentPolicy(cfg, n_assets=len(symbol_store)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    envs: list[TradingEnv] = []
    for sym in symbols:
        data = symbol_store[sym]
        seg = data.segment_indices[0]
        start_idx = max(seg.test_start - cfg.data.seq_length, 0)
        steps = min(max_steps, max(seg.test_end - (start_idx + cfg.data.seq_length), 1))
        envs.append(
            TradingEnv(
                symbol=sym,
                features=data.features,
                raw_ohlcva=data.raw_ohlcva,
                time_features=data.time_features,
                prices=data.prices,
                asset_id=symbol_to_asset_id[sym],
                market_group=_market_group_from_metadata(sym, data.market, data.asset_class),
                price_basis=cfg.data.price_basis,
                corp_action_cash_dividend_per10=data.corp_action_cash_dividend_per10,
                corp_action_share_multiplier=data.corp_action_share_multiplier,
                corp_action_rights_ratio_per10=data.corp_action_rights_ratio_per10,
                corp_action_rights_event=data.corp_action_rights_event,
                cfg=cfg.env,
                seq_length=cfg.data.seq_length,
                start_idx=start_idx,
                max_steps=steps,
                tier=data.tier,
            )
        )

    rows: list[dict[str, Any]] = []
    step = 0
    while envs and step < max_steps:
        active = [e for e in envs if not e.is_done()]
        if not active:
            break

        market = np.stack([e.get_market_features() for e in active]).astype(np.float32)
        pos_info = np.stack([e.get_position_info() for e in active]).astype(np.float32)
        masks = np.stack([e.get_action_mask() for e in active]).astype(np.float32)
        asset_ids = np.asarray([e.asset_id for e in active], dtype=np.int64)
        states = np.asarray([e.position_state for e in active], dtype=np.int64)

        with torch.no_grad():
            actions = policy.act_deterministic(
                torch.from_numpy(market).to(device),
                torch.from_numpy(pos_info).to(device),
                torch.from_numpy(asset_ids).to(device),
                torch.from_numpy(states).to(device),
                torch.from_numpy(masks).to(device),
            ).cpu().numpy()

        for env, action in zip(active, actions):
            bar_idx = int(env.current_idx)
            before = len(env.completed_trades)
            env.step(action, 0.0)
            dates = dates_by_symbol.get(env.symbol)
            for trade in env.completed_trades[before:]:
                exit_date = dates.iloc[bar_idx] if dates is not None and bar_idx < len(dates) else pd.NaT
                rows.append(
                    {
                        "symbol": env.symbol,
                        "pair": symbol_to_pair(env.symbol),
                        "pnl": float(trade.pnl),
                        "direction": int(trade.direction),
                        "holding_steps": int(trade.holding_steps),
                        "exit_idx": int(bar_idx),
                        "exit_date": exit_date,
                        "exit_reason": str(trade.exit_reason),
                        "stop_source": str(trade.stop_source),
                        "stop_loss_value": float(getattr(trade, "stop_loss_value", 0.0)),
                    }
                )
        step += 1

    return pd.DataFrame(rows)


def _summarize(
    trades: pd.DataFrame,
    *,
    label: str,
    stop_loss: float,
    stop_loss_vol_multiple: float,
    take_profit: float,
    symbols: int,
    wallet: float,
    base_stake: float,
    days: float,
) -> dict[str, Any]:
    if trades.empty:
        return {
            "label": label,
            "stop_loss": stop_loss,
            "stop_loss_vol_multiple": stop_loss_vol_multiple,
            "take_profit": take_profit,
            "symbols": symbols,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "pnl_sum_pct": 0.0,
            "profit_abs": 0.0,
            "account_roi_pct": 0.0,
            "simple_annual_roi_pct": 0.0,
            "closed_trade_max_dd_pct": 0.0,
        }

    pnl = trades["pnl"].astype(float)
    profit_abs = pnl * float(base_stake)
    ordered = trades.assign(profit_abs=profit_abs)
    if "exit_date" in ordered:
        ordered = ordered.sort_values(["exit_date", "pair", "exit_idx"], kind="stable")
    equity = profit_abs.loc[ordered.index].cumsum()
    curve = np.concatenate([[0.0], equity.to_numpy(dtype=np.float64)])
    peak = np.maximum.accumulate(curve)
    max_dd_abs = float(np.max(peak - curve)) if curve.size else 0.0
    wins = int((pnl > 0.0).sum())
    total_profit = float(profit_abs.sum())
    by_reason = ordered.groupby("exit_reason")["profit_abs"].agg(["count", "sum"]).to_dict("index")
    stop_rows = ordered[ordered["exit_reason"].eq("stop_loss")]
    return {
        "label": label,
        "stop_loss": float(stop_loss),
        "stop_loss_vol_multiple": float(stop_loss_vol_multiple),
        "take_profit": float(take_profit),
        "symbols": int(symbols),
        "trades": int(len(ordered)),
        "wins": wins,
        "win_rate": float(wins / max(len(ordered), 1)),
        "pnl_sum_pct": float(pnl.sum() * 100.0),
        "profit_abs": total_profit,
        "account_roi_pct": float(total_profit / max(wallet, 1e-8) * 100.0),
        "simple_annual_roi_pct": float(total_profit / max(wallet, 1e-8) * 100.0 / days * 365.0),
        "closed_trade_max_dd_pct": float(max_dd_abs / max(wallet, 1e-8) * 100.0),
        "stop_loss_trades": int(by_reason.get("stop_loss", {}).get("count", 0)),
        "stop_loss_profit_abs": float(by_reason.get("stop_loss", {}).get("sum", 0.0)),
        "take_profit_trades": int(by_reason.get("take_profit", {}).get("count", 0)),
        "take_profit_profit_abs": float(by_reason.get("take_profit", {}).get("sum", 0.0)),
        "close_signal_trades": int(by_reason.get("close_signal", {}).get("count", 0)),
        "close_signal_profit_abs": float(by_reason.get("close_signal", {}).get("sum", 0.0)),
        "force_close_trades": int(by_reason.get("force_close", {}).get("count", 0)),
        "force_close_profit_abs": float(by_reason.get("force_close", {}).get("sum", 0.0)),
        "avg_stop_loss_value": float(stop_rows["stop_loss_value"].mean()) if len(stop_rows) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="top100_pairs.json")
    parser.add_argument("--datadir", default="user_data/data/binance/futures")
    parser.add_argument("--cfg", default="configs/e2e_ppo_run19_sl15_valrandom_seq180.json")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-stake", type=float, default=1000.0)
    parser.add_argument("--wallet", type=float, default=100000.0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--stop-mults", default="0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0")
    parser.add_argument("--fixed-stops", default="-0.02,-0.03,-0.04,-0.05,-0.06,-0.08")
    parser.add_argument("--take-profits", default="0.0")
    parser.add_argument("--outdir", default="stoploss_scan_202509_202605")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    frames, availability = load_common_ohlcv(
        pairs=pairs,
        datadir=Path(args.datadir),
        start=REQUESTED_START,
        end=REQUESTED_END,
        seq_len=SEQ_LEN,
    )
    days = (pd.Timestamp(REQUESTED_END) - pd.Timestamp(REQUESTED_START)).total_seconds() / 86400.0
    availability.to_csv(outdir / "availability.csv", index=False)

    combos: list[tuple[str, float, float, float]] = []
    for take_profit in _parse_float_list(args.take_profits):
        for mult in _parse_float_list(args.stop_mults):
            combos.append((f"atr_{mult:g}_tp_{take_profit:g}", 0.0, mult, take_profit))
        for fixed in _parse_float_list(args.fixed_stops):
            combos.append((f"fixed_{abs(fixed):g}_tp_{take_profit:g}", fixed, 0.0, take_profit))

    rows: list[dict[str, Any]] = []
    for i, (label, fixed, mult, take_profit) in enumerate(combos, start=1):
        print(f"[{i}/{len(combos)}] {label}", flush=True)
        trades = _run_one(
            cfg_path=args.cfg,
            checkpoint=args.checkpoint,
            frames=frames,
            device_name=args.device,
            max_steps=int(args.max_steps),
            stop_loss=fixed,
            stop_loss_vol_multiple=mult,
            take_profit=take_profit,
        )
        trades.to_csv(outdir / f"trades_{label}.csv", index=False)
        rows.append(
            _summarize(
                trades,
                label=label,
                stop_loss=fixed,
                stop_loss_vol_multiple=mult,
                take_profit=take_profit,
                symbols=len(frames),
                wallet=float(args.wallet),
                base_stake=float(args.base_stake),
                days=days,
            )
        )
        pd.DataFrame(rows).sort_values("account_roi_pct", ascending=False).to_csv(
            outdir / "summary_partial.csv",
            index=False,
        )

    summary = pd.DataFrame(rows).sort_values("account_roi_pct", ascending=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    best = summary.iloc[0].to_dict() if len(summary) else {}
    (outdir / "best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
