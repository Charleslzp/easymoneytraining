#!/usr/bin/env python3
"""
Compare run19 native environment and Freqtrade PPOStrategyV2 on the same local
Binance futures 4h candles for the requested 2025-09 to 2026-05 window.

The script intentionally uses Freqtrade's local feather candles as the common
data source for the native run.  The original run19 parquet currently ends at
2026-03-12 for many symbols, so it cannot cover the requested end date.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from train_e2e_ppo import (
    Config,
    Segment,
    StudentPolicy,
    SymbolData,
    TradingEnv,
    _market_group_from_metadata,
    load_config,
    set_seed,
)


REQUESTED_START = "2025-09-01 00:00:00+00:00"
REQUESTED_END = "2026-05-14 00:00:00+00:00"
DEFAULT_TIMERANGE = "20250901-20260514"
SEQ_LEN = 180
MAX_STEPS = 10_000


def _live_norm_get_market_features(self: TradingEnv) -> np.ndarray:
    s = max(0, self.current_idx - self.seq_len)
    e = self.current_idx
    raw = self.raw_ohlcva[s:e, :5].copy()
    out = np.zeros_like(raw, dtype=np.float32)
    for i in range(4):
        p = raw[:, i]
        ret = np.zeros(len(p), dtype=np.float32)
        ret[1:] = (p[1:] - p[:-1]) / (p[:-1] + 1e-8)
        out[:, i] = ret
    vol = raw[:, 4]
    lv = np.log(vol + 1.0)
    out[:, 4] = (lv - lv.mean()) / (lv.std() + 1e-8)
    return out


TradingEnv.get_market_features = _live_norm_get_market_features


def pair_to_symbol(pair: str) -> str:
    return pair.split("/")[0] + "USDT"


def symbol_to_pair(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    return f"{base}/USDT:USDT"


def pair_to_feather_name(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_") + "-4h-futures.feather"


def load_pairs(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if not isinstance(pairs, list):
        raise ValueError(f"Expected pair list in {path}")
    return [str(p) for p in pairs]


def load_common_ohlcv(
    pairs: list[str],
    datadir: Path,
    start: str,
    end: str,
    seq_len: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    warmup_start = start_ts - pd.Timedelta(hours=4 * seq_len)
    rows: list[dict[str, Any]] = []
    out: dict[str, pd.DataFrame] = {}

    for pair in pairs:
        path = datadir / pair_to_feather_name(pair)
        symbol = pair_to_symbol(pair)
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
        if status in {"ok", "partial_warmup"}:
            out[symbol] = df

    return out, pd.DataFrame(rows)


def build_symbol_store_from_feathers(
    frames: dict[str, pd.DataFrame],
    start: str,
    end: str,
    seq_len: int,
) -> dict[str, SymbolData]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    symbol_store: dict[str, SymbolData] = {}

    for symbol, frame in frames.items():
        g = frame.reset_index(drop=True)
        dt = pd.to_datetime(g["date"], utc=True)
        decision_start = int(dt.searchsorted(start_ts, side="left"))
        decision_end = int(dt.searchsorted(end_ts, side="left"))
        warmup_start = max(decision_start - seq_len, 0)

        if decision_end - decision_start < 2 or decision_start <= 0:
            continue

        ohlcv = g[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        amount = (ohlcv[:, 3] * ohlcv[:, 4]).astype(np.float32)
        raw_ohlcva = np.concatenate([ohlcv, amount[:, None]], axis=1)
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
        features = np.zeros_like(ohlcv, dtype=np.float32)
        prices = g["close"].astype(np.float32).values
        seg = Segment(
            train_start=0,
            train_end=warmup_start,
            val_start=warmup_start,
            val_end=decision_start,
            test_start=decision_start,
            test_end=decision_end,
        )
        symbol_store[symbol] = SymbolData(
            symbol=symbol,
            features=features,
            raw_ohlcva=raw_ohlcva,
            time_features=time_features,
            prices=prices,
            segment_indices=[seg],
            market="binance_futures",
            asset_class="crypto",
            tier="tier3",
            corp_action_cash_dividend_per10=np.zeros(len(g), dtype=np.float32),
            corp_action_share_multiplier=np.ones(len(g), dtype=np.float32),
            corp_action_rights_ratio_per10=np.zeros(len(g), dtype=np.float32),
            corp_action_rights_event=np.zeros(len(g), dtype=bool),
        )

    return symbol_store


def summarize_native_trade_rows(symbols: list[str], trade_rows: list[dict[str, Any]]) -> pd.DataFrame:
    by_symbol = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for row in trade_rows:
        item = by_symbol[row["symbol"]]
        item["pnl"] += float(row["pnl"])
        item["trades"] += 1
        item["wins"] += int(float(row["pnl"]) > 0.0)

    summary_rows = []
    for sym in symbols:
        item = by_symbol[sym]
        trades = int(item["trades"])
        wins = int(item["wins"])
        summary_rows.append(
            {
                "symbol": sym,
                "pair": symbol_to_pair(sym),
                "pnl_pct_sum": float(item["pnl"] * 100.0),
                "trades": trades,
                "wins": wins,
                "win_rate": wins / trades if trades else 0.0,
            }
        )
    return pd.DataFrame(summary_rows)


def run_native(
    cfg_path: str,
    checkpoint: str,
    symbol_store: dict[str, SymbolData],
    frames: dict[str, pd.DataFrame],
    device_name: str,
    max_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(cfg_path)
    cfg.data.seq_length = SEQ_LEN
    cfg.data.overlap_lookback = SEQ_LEN
    set_seed(cfg.run.seed)

    device = torch.device(device_name)
    policy = StudentPolicy(cfg, n_assets=len(symbol_store)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    symbols = list(symbol_store.keys())
    symbol_to_asset_id = {sym: idx for idx, sym in enumerate(symbols)}
    dates_by_symbol = {
        sym: pd.to_datetime(frames[sym]["date"], utc=True).reset_index(drop=True)
        for sym in symbols
        if sym in frames
    }
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

    all_trades = []
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
            for trade in env.completed_trades[before:]:
                dates = dates_by_symbol.get(env.symbol)
                holding_steps = int(trade.holding_steps)
                if str(trade.exit_reason) == "close_signal":
                    entry_idx = max(bar_idx - max(holding_steps, 1), 0)
                else:
                    entry_idx = max(bar_idx - max(holding_steps, 1) + 1, 0)
                exit_date = str(dates.iloc[bar_idx]) if dates is not None and bar_idx < len(dates) else ""
                entry_date = str(dates.iloc[entry_idx]) if dates is not None and entry_idx < len(dates) else ""
                terminal_unexecutable = bool(
                    trade.exit_reason == "force_close"
                    and holding_steps == 1
                    and dates is not None
                    and bar_idx >= len(dates) - 1
                )
                all_trades.append(
                    {
                        "symbol": env.symbol,
                        "pair": symbol_to_pair(env.symbol),
                        "pnl": trade.pnl,
                        "direction": trade.direction,
                        "holding_steps": holding_steps,
                        "entry_idx": entry_idx,
                        "exit_idx": bar_idx,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "exit_reason": trade.exit_reason,
                        "stop_source": trade.stop_source,
                        "stop_loss_value": float(getattr(trade, "stop_loss_value", 0.0)),
                        "terminal_entry_rejected_by_freqtrade": terminal_unexecutable,
                    }
                )
        step += 1

    trade_df = pd.DataFrame(all_trades)
    native_summary = summarize_native_trade_rows(symbols, all_trades)
    executable_rows = [r for r in all_trades if not r["terminal_entry_rejected_by_freqtrade"]]
    native_executable_summary = summarize_native_trade_rows(symbols, executable_rows)
    return native_summary, trade_df, native_executable_summary


def _load_backtest_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [
                n
                for n in zf.namelist()
                if n.endswith(".json") and not n.endswith("meta.json") and not n.endswith("config.json")
            ]
            if not names:
                raise ValueError(f"No result json inside {path}")
            return json.loads(zf.read(names[0]))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_freqtrade_result(backtest_dir: Path) -> Path:
    files = list(backtest_dir.glob("backtest-result-*.json")) + list(backtest_dir.glob("backtest-result-*.zip"))
    if not files:
        raise FileNotFoundError(f"No Freqtrade result files in {backtest_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def analyze_freqtrade_result(
    path: Path,
    pairs: list[str],
    strategy_name: str,
    base_stake: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = _load_backtest_json(path)
    strategy_data = (data.get("strategy") or {}).get(strategy_name)
    if strategy_data is None:
        available = sorted((data.get("strategy") or {}).keys())
        raise ValueError(f"{strategy_name} not found in {path}; available={available}")
    trades = strategy_data.get("trades", [])
    pair_set = set(pairs)
    rows = [t for t in trades if t.get("pair") in pair_set]
    trade_rows = []
    by_pair = defaultdict(
        lambda: {
            "pnl": 0.0,
            "pnl_native_basis": 0.0,
            "funding_fees": 0.0,
            "trades": 0,
            "wins": 0,
            "profit_abs": 0.0,
        }
    )
    for trade in rows:
        pair = trade["pair"]
        profit_abs = float(trade.get("profit_abs", 0.0))
        profit_ratio = float(trade.get("profit_ratio", 0.0))
        funding_fees = float(trade.get("funding_fees", 0.0) or 0.0)
        pnl = profit_abs / max(float(base_stake), 1e-8)
        pnl_native_basis = (profit_abs - funding_fees) / max(float(base_stake), 1e-8)
        item = by_pair[pair]
        item["pnl"] += pnl
        item["pnl_native_basis"] += pnl_native_basis
        item["funding_fees"] += funding_fees
        item["profit_abs"] += profit_abs
        item["trades"] += 1
        item["wins"] += int(pnl > 0.0)
        trade_rows.append(
            {
                "pair": pair,
                "symbol": pair_to_symbol(pair),
                "pnl": pnl,
                "pnl_native_basis": pnl_native_basis,
                "profit_ratio": profit_ratio,
                "profit_abs": profit_abs,
                "funding_fees": funding_fees,
                "open_date": trade.get("open_date"),
                "close_date": trade.get("close_date"),
                "is_short": trade.get("is_short"),
                "exit_reason": trade.get("exit_reason"),
            }
        )

    summary_rows = []
    for pair in pairs:
        item = by_pair[pair]
        trades_n = int(item["trades"])
        wins = int(item["wins"])
        summary_rows.append(
            {
                "pair": pair,
                "symbol": pair_to_symbol(pair),
                "pnl_pct_sum": float(item["pnl"] * 100.0),
                "pnl_pct_sum_native_basis": float(item["pnl_native_basis"] * 100.0),
                "funding_pct_sum": float(item["funding_fees"] / max(float(base_stake), 1e-8) * 100.0),
                "profit_abs": float(item["profit_abs"]),
                "funding_fees": float(item["funding_fees"]),
                "trades": trades_n,
                "wins": wins,
                "win_rate": wins / trades_n if trades_n else 0.0,
            }
        )
    meta = {
        "path": str(path),
        "strategy_name": strategy_name,
        "backtest_start": strategy_data.get("backtest_start"),
        "backtest_end": strategy_data.get("backtest_end"),
        "strategy_total_trades": strategy_data.get("total_trades"),
        "strategy_profit_total": strategy_data.get("profit_total"),
        "strategy_profit_total_abs": strategy_data.get("profit_total_abs"),
        "base_stake": float(base_stake),
        "funding_fees_abs": float(sum(float(t.get("funding_fees", 0.0) or 0.0) for t in rows)),
    }
    profit_total = float(meta["strategy_profit_total"] or 0.0)
    profit_total_abs = float(meta["strategy_profit_total_abs"] or 0.0)
    if abs(profit_total) > 1e-12:
        meta["account_start_balance_est"] = profit_total_abs / profit_total
    else:
        meta["account_start_balance_est"] = math.nan
    return pd.DataFrame(summary_rows), pd.DataFrame(trade_rows), meta


def summarize(
    label: str,
    df: pd.DataFrame,
    days: float,
    pnl_col: str = "pnl_pct_sum",
) -> dict[str, Any]:
    total_trades = int(df["trades"].sum())
    wins = int(df["wins"].sum())
    pnl = float(df[pnl_col].sum())
    return {
        "label": label,
        "symbols": int(len(df)),
        "symbols_with_trades": int((df["trades"] > 0).sum()),
        "pnl_pct_sum": pnl,
        "trades": total_trades,
        "win_rate": wins / total_trades if total_trades else 0.0,
        "annualized_pct_sum": pnl / days * 365.0 if days > 0 else math.nan,
    }


def write_report(
    path: Path,
    native_summary: pd.DataFrame,
    native_executable_summary: pd.DataFrame,
    freq_summary: pd.DataFrame,
    availability: pd.DataFrame,
    meta: dict[str, Any],
    start: str,
    end: str,
    days: float,
) -> None:
    native = summarize("Run19 Native", native_summary, days)
    native_executable = summarize("Run19 Native (Freqtrade-executable)", native_executable_summary, days)
    freq = summarize("Freqtrade", freq_summary, days)
    freq_native_basis = summarize(
        "Freqtrade (native basis, no funding)",
        freq_summary,
        days,
        pnl_col="pnl_pct_sum_native_basis",
    )
    merged = native_executable_summary.merge(
        freq_summary,
        on=["pair", "symbol"],
        how="outer",
        suffixes=("_native", "_freqtrade"),
    ).fillna(0)
    top_delta = merged.assign(
        delta=merged["pnl_pct_sum_native"] - merged["pnl_pct_sum_native_basis"]
    )
    top_delta = top_delta.reindex(top_delta["delta"].abs().sort_values(ascending=False).index).head(12)
    unavailable = availability[availability["status"].ne("ok")]
    funding_pct = float(freq_summary["funding_pct_sum"].sum()) if "funding_pct_sum" in freq_summary else 0.0
    profit_abs = float(meta.get("strategy_profit_total_abs") or 0.0)
    start_balance = float(meta.get("account_start_balance_est") or math.nan)
    account_roi = profit_abs / start_balance * 100.0 if start_balance > 0 else math.nan
    loaded_pair_capital = len(native_summary) * float(meta.get("base_stake") or 0.0)
    requested_pair_capital = len(availability) * float(meta.get("base_stake") or 0.0)
    freq_loaded_pair_roi = profit_abs / loaded_pair_capital * 100.0 if loaded_pair_capital > 0 else math.nan
    freq_requested_pair_roi = profit_abs / requested_pair_capital * 100.0 if requested_pair_capital > 0 else math.nan
    native_loaded_pair_roi = native["pnl_pct_sum"] / native["symbols"] if native["symbols"] > 0 else math.nan
    native_requested_pair_roi = native["pnl_pct_sum"] / len(availability) if len(availability) > 0 else math.nan

    lines = [
        "# Run19 Native vs Freqtrade: 2025-09 to 2026-05",
        "",
        f"- Requested window: `{start}` to `{end}`",
        f"- Effective days: `{days:.2f}`",
        f"- Common data source for native run: `user_data/data/binance/futures/*-4h-futures.feather`",
        f"- Freqtrade result: `{meta.get('path', '')}`",
        f"- Freqtrade reported window: `{meta.get('backtest_start')}` to `{meta.get('backtest_end')}`",
        f"- Freqtrade PNL sum basis: `profit_abs / base_stake`, base_stake=`{meta.get('base_stake')}`",
        f"- Freqtrade account ROI basis: `profit_abs / starting_balance`, starting_balance~`{start_balance:.2f}`",
        f"- Freqtrade funding contribution: `{funding_pct:.4f}%` of base-stake sum",
        f"- Requested pairs: `{len(availability)}`; native/Freqtrade local 4h data loaded: `{len(native_summary)}`",
        "",
        "## Overall",
        "",
        "| Engine | Symbols | Symbols with trades | PNL sum | Annualized PNL sum | Trades | Win rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| {native['label']} | {native['symbols']} | {native['symbols_with_trades']} | "
            f"{native['pnl_pct_sum']:.2f}% | {native['annualized_pct_sum']:.2f}% | "
            f"{native['trades']} | {native['win_rate'] * 100:.2f}% |"
        ),
        (
            f"| {native_executable['label']} | {native_executable['symbols']} | "
            f"{native_executable['symbols_with_trades']} | "
            f"{native_executable['pnl_pct_sum']:.2f}% | "
            f"{native_executable['annualized_pct_sum']:.2f}% | "
            f"{native_executable['trades']} | {native_executable['win_rate'] * 100:.2f}% |"
        ),
        (
            f"| {freq['label']} | {freq['symbols']} | {freq['symbols_with_trades']} | "
            f"{freq['pnl_pct_sum']:.2f}% | {freq['annualized_pct_sum']:.2f}% | "
            f"{freq['trades']} | {freq['win_rate'] * 100:.2f}% |"
        ),
        (
            f"| {freq_native_basis['label']} | {freq_native_basis['symbols']} | "
            f"{freq_native_basis['symbols_with_trades']} | "
            f"{freq_native_basis['pnl_pct_sum']:.2f}% | "
            f"{freq_native_basis['annualized_pct_sum']:.2f}% | "
            f"{freq_native_basis['trades']} | {freq_native_basis['win_rate'] * 100:.2f}% |"
        ),
        "",
        (
            "`Run19 Native (Freqtrade-executable)` removes native trades opened and force-closed "
            "on the final candle, because Freqtrade shifts signals by one candle and rejects "
            "entries on its last backtest row."
        ),
        (
            "`Freqtrade (native basis, no funding)` removes Freqtrade futures funding fees from "
            "`profit_abs`; native run19 does not model funding. The remaining residual is mostly "
            "exchange amount precision/rounding."
        ),
        "",
        "## Portfolio ROI Context",
        "",
        "| Basis | Run19 Native | Freqtrade |",
        "|---|---:|---:|",
        f"| Freqtrade account wallet |  | {account_roi:.4f}% |",
        f"| Equal loaded pairs (`{len(native_summary)}` x base stake) | {native_loaded_pair_roi:.4f}% | {freq_loaded_pair_roi:.4f}% |",
        f"| Equal requested pairs (`{len(availability)}` x base stake) | {native_requested_pair_roi:.4f}% | {freq_requested_pair_roi:.4f}% |",
        "",
        (
            "`PNL sum` is a sum of per-pair/base-stake percentage points. It is useful for "
            "engine alignment, but it is not the same denominator as Freqtrade's account ROI."
        ),
        "",
        "## Largest Executable Native Minus Freqtrade Native-Basis Deltas",
        "",
        top_delta[
            [
                "pair",
                "pnl_pct_sum_native",
                "pnl_pct_sum_native_basis",
                "funding_pct_sum",
                "delta",
                "trades_native",
                "trades_freqtrade",
            ]
        ].to_markdown(index=False),
        "",
        "## Data Limitations",
        "",
        (
            "The original run19 parquet was not used as the native market-data source for this run "
            "because it does not cover the full requested end date for many symbols. "
            "Using Freqtrade's local candles keeps both engines on the same OHLCV bars."
        ),
        "",
        f"Unavailable or partial requested pairs: `{len(unavailable)}`",
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
    parser.add_argument("--freqtrade-result", default="")
    parser.add_argument("--freqtrade-strategy", default="PPORun19SessionStrategy")
    parser.add_argument("--base-stake", type=float, default=1000.0)
    parser.add_argument("--start", default=REQUESTED_START)
    parser.add_argument("--end", default=REQUESTED_END)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", default="comparison_202509_202605")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    frames, availability = load_common_ohlcv(
        pairs=pairs,
        datadir=Path(args.datadir),
        start=args.start,
        end=args.end,
        seq_len=SEQ_LEN,
    )
    ok_pairs = [symbol_to_pair(sym) for sym in frames]
    symbol_store = build_symbol_store_from_feathers(frames, args.start, args.end, SEQ_LEN)

    native_summary, native_trades, native_executable_summary = run_native(
        cfg_path=args.cfg,
        checkpoint=args.checkpoint,
        symbol_store=symbol_store,
        frames=frames,
        device_name=args.device,
        max_steps=MAX_STEPS,
    )

    if args.freqtrade_result:
        freq_path = Path(args.freqtrade_result)
    else:
        freq_path = latest_freqtrade_result(Path("user_data/backtest_results"))
    freq_summary, freq_trades, freq_meta = analyze_freqtrade_result(
        freq_path,
        ok_pairs,
        args.freqtrade_strategy,
        args.base_stake,
    )

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    days = (end_ts - start_ts).total_seconds() / 86400.0

    availability.to_csv(outdir / "availability.csv", index=False)
    native_summary.to_csv(outdir / "native_summary.csv", index=False)
    native_executable_summary.to_csv(outdir / "native_freqtrade_executable_summary.csv", index=False)
    native_trades.to_csv(outdir / "native_trades.csv", index=False)
    freq_summary.to_csv(outdir / "freqtrade_summary.csv", index=False)
    freq_trades.to_csv(outdir / "freqtrade_trades.csv", index=False)
    with open(outdir / "freqtrade_meta.json", "w", encoding="utf-8") as f:
        json.dump(freq_meta, f, indent=2)
    write_report(
        path=outdir / "report.md",
        native_summary=native_summary,
        native_executable_summary=native_executable_summary,
        freq_summary=freq_summary,
        availability=availability,
        meta=freq_meta,
        start=args.start,
        end=args.end,
        days=days,
    )

    print(json.dumps({
        "outdir": str(outdir),
        "native": summarize("Run19 Native", native_summary, days),
        "native_freqtrade_executable": summarize(
            "Run19 Native (Freqtrade-executable)",
            native_executable_summary,
            days,
        ),
        "freqtrade": summarize("Freqtrade", freq_summary, days),
        "freqtrade_native_basis_no_funding": summarize(
            "Freqtrade (native basis, no funding)",
            freq_summary,
            days,
            pnl_col="pnl_pct_sum_native_basis",
        ),
        "freqtrade_funding_pct_sum": float(freq_summary["funding_pct_sum"].sum()),
        "freqtrade_account_start_balance_est": freq_meta.get("account_start_balance_est"),
        "freqtrade_account_roi_pct": (
            float(freq_meta.get("strategy_profit_total_abs") or 0.0)
            / float(freq_meta.get("account_start_balance_est") or math.nan)
            * 100.0
        ),
        "freqtrade_result": str(freq_path),
        "pairs_loaded": len(symbol_store),
    }, indent=2))


if __name__ == "__main__":
    main()
