#!/usr/bin/env python3
"""Incrementally extend the Run19 4h crypto parquet with newer candles."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/home/pythonuser/pythonproject/bigmodel/data_4h/processed/crypto_sorted_full_history_4h.parquet")
DEFAULT_BIGMODEL_DOWNLOADER = Path("/home/pythonuser/pythonproject/bigmodel/download_crypto_multi_sorted.py")
DEFAULT_OUTPUT = ROOT_DIR / "data_4h/processed/crypto_sorted_full_history_4h_202607.parquet"


def _load_downloader(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Downloader module not found: {path}")
    spec = importlib.util.spec_from_file_location("bigmodel_download_crypto_multi_sorted", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load downloader module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _end_date_default() -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()


def _normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Source parquet missing required columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False).dt.tz_localize(None)
    if "exchange" not in df.columns:
        df["exchange"] = "binance"
    if "market_group" not in df.columns:
        df["market_group"] = "CRYPTO"
    return df


def _last_rows(df: pd.DataFrame, limit_symbols: int = 0, symbols: List[str] | None = None) -> List[Tuple[str, str, pd.Timestamp]]:
    if symbols:
        wanted = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
        df = df[df["symbol"].astype(str).str.upper().isin(wanted)]
    rows: List[Tuple[str, str, pd.Timestamp]] = []
    last = df.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=True).tail(1)
    if limit_symbols > 0:
        last = last.head(limit_symbols)
    for row in last.itertuples(index=False):
        rows.append((str(row.symbol), str(getattr(row, "exchange", "binance")), pd.Timestamp(row.timestamp)))
    return rows


def _start_date_for(last_ts: pd.Timestamp, lookback_days: int) -> str:
    return (last_ts.to_pydatetime().date() - timedelta(days=lookback_days)).isoformat()


def _download_one(
    symbol: str,
    exchange: str,
    last_ts: pd.Timestamp,
    end_date: str,
    lookback_days: int,
    downloaders: Dict[str, Callable[[str, str, str], pd.DataFrame]],
) -> Tuple[str, str, int, str, pd.DataFrame]:
    fn = downloaders.get(exchange)
    if fn is None:
        return symbol, exchange, 0, f"unsupported exchange {exchange}", pd.DataFrame()
    start_date = _start_date_for(last_ts, lookback_days)
    try:
        df = fn(symbol, start_date, end_date)
    except Exception as exc:  # noqa: BLE001 - downloader failures are per-symbol.
        return symbol, exchange, 0, repr(exc), pd.DataFrame()
    if df is None or df.empty:
        return symbol, exchange, 0, "empty", pd.DataFrame()
    df = _normalize_source(df)
    df = df[df["timestamp"] > last_ts].copy()
    if df.empty:
        return symbol, exchange, 0, "up_to_date", pd.DataFrame()
    df["symbol"] = symbol
    df["exchange"] = exchange
    df["market_group"] = "CRYPTO"
    return symbol, exchange, len(df), "", df


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Existing parquet to extend")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output parquet path")
    parser.add_argument("--downloader", default=str(DEFAULT_BIGMODEL_DOWNLOADER), help="bigmodel downloader module")
    parser.add_argument("--end-date", default=_end_date_default(), help="Exclusive UTC end date, YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent symbol downloads")
    parser.add_argument("--lookback-days", type=int, default=2, help="Overlap when fetching increments")
    parser.add_argument("--limit-symbols", type=int, default=0, help="Smoke-test limit; 0 means all symbols")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol allowlist for smoke tests")
    parser.add_argument("--require-min-timestamp", default="", help="Fail unless output max timestamp reaches this value")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source parquet not found: {source}")

    module = _load_downloader(Path(args.downloader).expanduser())
    downloaders = {
        "binance": module.download_binance_4h,
        "okx": module.download_okx_4h,
        "bybit": module.download_bybit_4h,
        "gate": module.download_gate_4h,
        "coinbase": module.download_coinbase_4h,
    }

    base_df = _normalize_source(pd.read_parquet(source))
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tasks = _last_rows(base_df, limit_symbols=max(args.limit_symbols, 0), symbols=symbols)
    print(f"source={source}")
    print(f"rows={len(base_df):,} symbols={base_df['symbol'].nunique():,}")
    print(f"source_range={base_df['timestamp'].min()} -> {base_df['timestamp'].max()}")
    print(f"updating_symbols={len(tasks):,} end_date={args.end_date} workers={args.workers}")

    new_parts: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = [
            pool.submit(
                _download_one,
                symbol,
                exchange,
                last_ts,
                args.end_date,
                max(args.lookback_days, 0),
                downloaders,
            )
            for symbol, exchange, last_ts in tasks
        ]
        for future in as_completed(futures):
            symbol, exchange, n_rows, error, df = future.result()
            done += 1
            if not df.empty:
                new_parts.append(df)
            if error and error not in {"up_to_date"}:
                failures.append({"symbol": symbol, "exchange": exchange, "status": error})
            if done % 25 == 0 or done == len(futures):
                added = sum(len(part) for part in new_parts)
                print(f"progress={done}/{len(futures)} added_rows={added:,} failures={len(failures):,}")

    if new_parts:
        all_columns = list(base_df.columns)
        for part in new_parts:
            for column in part.columns:
                if column not in all_columns:
                    all_columns.append(column)
        combined = pd.concat([_ensure_columns(base_df, all_columns)] + [_ensure_columns(p, all_columns) for p in new_parts], ignore_index=True)
        combined = combined.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")
        combined = combined.reset_index(drop=True)
    else:
        combined = base_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    failed_path = output.with_suffix(".failed.csv")
    pd.DataFrame(failures).to_csv(failed_path, index=False)

    print(f"output={output}")
    print(f"output_rows={len(combined):,} symbols={combined['symbol'].nunique():,}")
    print(f"output_range={combined['timestamp'].min()} -> {combined['timestamp'].max()}")
    print(f"new_rows={len(combined) - len(base_df):,} failures={len(failures):,}")
    print(f"failures_csv={failed_path}")
    if args.require_min_timestamp:
        required = pd.Timestamp(args.require_min_timestamp)
        observed = pd.Timestamp(combined["timestamp"].max())
        if observed < required:
            raise SystemExit(f"output max timestamp {observed} is earlier than required {required}")


if __name__ == "__main__":
    main()
