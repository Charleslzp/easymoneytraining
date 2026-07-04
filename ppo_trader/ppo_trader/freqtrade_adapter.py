"""Freqtrade adapter built on top of the run19 session manager.

This module keeps the stateful run19 replay logic inside the wheel.  A
Freqtrade strategy should only delegate dataframe and callback work here.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from .session import SessionManager
from .utils import Actions, States


logger = logging.getLogger(__name__)


class Run19FreqtradeAdapter:
    """Build Freqtrade signals from the same session state machine as run19."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        seq_length: int = 180,
        base_stake: float = 1000.0,
        stop_loss: Optional[float] = None,
        stop_loss_vol_multiple: Optional[float] = None,
        take_profit: Optional[float] = None,
        cache_tag: str = "RUN19_WHEEL_ADAPTER_V1",
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.seq_length = int(seq_length)
        self.base_stake = float(base_stake)
        self.env_overrides = {
            key: float(value)
            for key, value in {
                "stop_loss": stop_loss,
                "stop_loss_vol_multiple": stop_loss_vol_multiple,
                "take_profit": take_profit,
            }.items()
            if value is not None
        }
        self.cache_tag = str(cache_tag)
        self.manager = SessionManager(
            self.model_path,
            device=self.device,
            seq_length=self.seq_length,
            env_overrides=self.env_overrides,
        )
        self.signal_cache: Dict[str, Dict[str, pd.Series]] = {}
        self.entry_records: Dict[str, Dict[str, float]] = {}
        self.stop_records: Dict[str, float] = {}
        self.exit_records: Dict[str, Dict[str, object]] = {}
        logger.info(
            "[Run19FreqtradeAdapter] model=%s device=%s seq_len=%s base_stake=%s env_overrides=%s",
            self.model_path,
            self.device,
            self.seq_length,
            self.base_stake,
            self.env_overrides,
        )

    @staticmethod
    def ts_key(value) -> int:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())

    @staticmethod
    def bar_from_row(row) -> np.ndarray:
        return np.array(
            [
                pd.Timestamp(row["date"]).timestamp(),
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            ],
            dtype=np.float64,
        )

    def clear_cache(self) -> None:
        self.signal_cache.clear()
        self.entry_records.clear()
        self.stop_records.clear()
        self.exit_records.clear()

    def cache_key(self, dataframe: DataFrame, pair: str) -> str:
        return (
            f"{pair}_{len(dataframe)}_{dataframe.iloc[0]['date']}_"
            f"{dataframe.iloc[-1]['date']}_SEQ{self.seq_length}_{self.cache_tag}"
        )

    def entry_key(self, pair: str, value) -> str:
        return f"{pair}_{self.ts_key(value)}"

    def stop_key(self, pair: str, entry_time, current_time) -> str:
        return f"{pair}_{self.ts_key(entry_time)}_{self.ts_key(current_time)}"

    def exit_key(self, pair: str, entry_time, current_time) -> str:
        return f"{pair}_{self.ts_key(entry_time)}_{self.ts_key(current_time)}"

    def record_entry(
        self,
        pair: str,
        entry_time,
        side: str,
        stop_loss: float,
        size: float,
        entry_price: float,
        stop_price: float,
    ) -> None:
        self.entry_records[self.entry_key(pair, entry_time)] = {
            "side": 1.0 if side == "long" else -1.0,
            "stop_loss": float(stop_loss),
            "size": abs(float(size)),
            "entry_price": float(entry_price),
            "stop_price": float(stop_price),
        }

    def find_entry_record(self, pair: str, current_time) -> Optional[Dict[str, float]]:
        exact = self.entry_records.get(self.entry_key(pair, current_time))
        if exact is not None:
            return exact

        target_ts = self.ts_key(current_time)
        best_key = None
        best_delta = None
        prefix = f"{pair}_"
        for key in self.entry_records:
            if not key.startswith(prefix):
                continue
            try:
                ts = int(key.split("_")[-1])
            except ValueError:
                continue
            delta = abs(ts - target_ts)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_key = key
        return self.entry_records.get(best_key) if best_key is not None else None

    def record_stop(self, pair: str, entry_time, current_time, stop_price: float) -> None:
        stop_price = float(stop_price)
        if stop_price > 0.0 and np.isfinite(stop_price):
            self.stop_records[self.stop_key(pair, entry_time, current_time)] = stop_price

    def find_stop_price(self, pair: str, entry_time, current_time) -> Optional[float]:
        exact = self.stop_records.get(self.stop_key(pair, entry_time, current_time))
        if exact is not None:
            return exact

        entry_ts = self.ts_key(entry_time)
        current_ts = self.ts_key(current_time)
        prefix = f"{pair}_{entry_ts}_"
        best_key = None
        best_ts = None
        for key in self.stop_records:
            if not key.startswith(prefix):
                continue
            try:
                ts = int(key.split("_")[-1])
            except ValueError:
                continue
            if ts <= current_ts and (best_ts is None or ts > best_ts):
                best_ts = ts
                best_key = key
        return self.stop_records.get(best_key) if best_key is not None else None

    def stoploss_ratio(
        self,
        pair: str,
        entry_time,
        current_time,
        current_rate: float,
        open_rate: float,
        is_short: bool,
        leverage: float = 1.0,
    ) -> Optional[float]:
        """Return the Freqtrade stoploss ratio for a native run19 stop.

        Freqtrade handles stoplosses that trigger on the entry candle with a
        pessimistic branch based on ``trade.stop_loss_pct`` relative to the
        entry open.  Native run19 exits at the ATR stop price.  Returning the
        same-candle ratio relative to ``open_rate`` makes that branch resolve
        to the native stop instead of the candle high/low.
        """
        stop_price = self.find_stop_price(pair, entry_time, current_time)
        if stop_price is None:
            return None

        stop_price = float(stop_price)
        if stop_price <= 0.0 or not np.isfinite(stop_price):
            return None

        same_candle = self.ts_key(entry_time) == self.ts_key(current_time)
        reference_rate = float(open_rate if same_candle else current_rate)
        if reference_rate <= 0.0 or not np.isfinite(reference_rate):
            return None

        if bool(is_short):
            ratio = (stop_price / reference_rate) - 1.0
        else:
            ratio = 1.0 - (stop_price / reference_rate)

        if not np.isfinite(ratio):
            return None
        ratio = max(min(float(ratio) * float(leverage or 1.0), 1.0), 0.0)
        if ratio <= 0.0:
            return None
        return ratio

    def record_exit(
        self,
        pair: str,
        entry_time,
        exit_time,
        reason: str,
        exit_price: float,
    ) -> None:
        price = float(exit_price)
        if price <= 0.0 or not np.isfinite(price):
            return
        self.exit_records[self.exit_key(pair, entry_time, exit_time)] = {
            "reason": str(reason or "exit"),
            "exit_price": price,
        }

    def find_exit_record(self, pair: str, entry_time, current_time) -> Optional[Dict[str, object]]:
        return self.exit_records.get(self.exit_key(pair, entry_time, current_time))

    def exit_reason(self, pair: str, entry_time, current_time) -> Optional[str]:
        record = self.find_exit_record(pair, entry_time, current_time)
        if record is None:
            return None
        reason = str(record.get("reason", "") or "")
        if reason == "close_signal":
            return "ppo_close"
        return reason or None

    def exit_price(self, pair: str, entry_time, current_time) -> Optional[float]:
        record = self.find_exit_record(pair, entry_time, current_time)
        if record is None:
            return None
        price = float(record.get("exit_price", 0.0))
        if price > 0.0 and np.isfinite(price):
            return price
        return None

    def new_session(self, pair: str, dataframe: DataFrame):
        old_id = self.manager.get_session_id(pair)
        if old_id is not None:
            self.manager.remove_session(old_id)
        safe_pair = pair.replace("/", "_").replace(":", "_")
        session_id = f"{safe_pair}_{self.ts_key(dataframe.iloc[0]['date'])}_{len(dataframe)}"
        return self.manager.get_session(self.manager.create_session(pair, session_id=session_id))

    def ensure_signals(self, dataframe: DataFrame, pair: str) -> DataFrame:
        cache_key = self.cache_key(dataframe, pair)
        cached = self.signal_cache.get(cache_key)
        if cached is not None:
            for col, values in cached.items():
                dataframe[col] = values.copy()
            return dataframe

        session = self.new_session(pair, dataframe)
        signal_cols = {
            "enter_long": 0,
            "enter_short": 0,
            "exit_long": 0,
            "exit_short": 0,
            "enter_tag": "",
            "exit_tag": "",
            "ppo_action": int(Actions.DO_NOTHING),
            "ppo_size": 0.0,
            "ppo_stop_loss": 0.0,
            "ppo_position_size": 0.0,
            "ppo_state_before": 0,
            "ppo_holding_before": 0,
            "ppo_empty_before": 0,
        }
        for col, value in signal_cols.items():
            dataframe[col] = value

        pending = None
        active_entry_time = None
        for row_pos, idx in enumerate(dataframe.index):
            row = dataframe.loc[idx]
            bar = self.bar_from_row(row)
            ready = session.add_bar(bar)

            if pending is not None:
                result_entry_time = active_entry_time
                result = session.execute_action(
                    pending["action"],
                    size=pending["size"],
                    entry_price=float(row["open"]),
                    bar_ohlcv=bar,
                    stop_loss_pct=None,
                    force_close_end=row_pos == len(dataframe) - 1,
                )
                stop_entry_time = active_entry_time
                if result.get("opened", False):
                    side = "long" if int(pending["action"]) == int(Actions.OPEN_LONG) else "short"
                    active_entry_time = row["date"]
                    result_entry_time = active_entry_time
                    stop_entry_time = active_entry_time
                    self.record_entry(
                        pair=pair,
                        entry_time=active_entry_time,
                        side=side,
                        stop_loss=float(result.get("stop_loss_pct", 0.0)),
                        size=abs(float(result.get("position_size", 0.0))),
                        entry_price=float(result.get("opened_entry_price", row["open"])),
                        stop_price=float(result.get("stop_price", 0.0)),
                    )
                if result.get("closed", False) and result_entry_time is not None:
                    reason = str(result.get("exit_reason", "") or "exit")
                    self.record_exit(
                        pair=pair,
                        entry_time=result_entry_time,
                        exit_time=row["date"],
                        reason=reason,
                        exit_price=float(result.get("exit_price", 0.0)),
                    )
                    if stop_entry_time is not None and reason == "stop_loss":
                        self.record_stop(
                            pair,
                            stop_entry_time,
                            row["date"],
                            float(result.get("stop_price", 0.0)),
                        )
                    active_entry_time = None
                elif int(session.service.position.state) == int(States.EMPTY):
                    active_entry_time = None
                pending = None

            if not ready or row_pos < self.seq_length - 1:
                continue

            state_before = int(session.service.position.state)
            holding_before = int(session.service.position.holding_steps)
            empty_before = int(session.service.position.empty_steps)
            action, size, stop_loss = session.predict()
            action = int(action)
            size = float(size)

            dataframe.loc[idx, "ppo_action"] = action
            dataframe.loc[idx, "ppo_size"] = size
            dataframe.loc[idx, "ppo_stop_loss"] = float(stop_loss)
            dataframe.loc[idx, "ppo_position_size"] = (
                abs(size) * max(float(session.service.position_scale), 0.0)
            )
            dataframe.loc[idx, "ppo_state_before"] = state_before
            dataframe.loc[idx, "ppo_holding_before"] = holding_before
            dataframe.loc[idx, "ppo_empty_before"] = empty_before

            has_next_bar = row_pos + 1 < len(dataframe)
            # Freqtrade shifts entry signals by one candle and then rejects
            # entries on the last backtest row.  A model entry on row N-2
            # would execute on row N-1 in native run19, but Freqtrade will
            # skip it.  Treat that terminal entry as non-executable here so
            # the wheel state machine and Freqtrade trades stay aligned.
            has_executable_entry_bar = row_pos + 2 < len(dataframe)
            pending_action = action
            pending_size = size
            if action in (int(Actions.OPEN_LONG), int(Actions.OPEN_SHORT)) and not has_executable_entry_bar:
                pending_action = int(Actions.DO_NOTHING)
                pending_size = 0.0

            if action == int(Actions.OPEN_LONG) and has_executable_entry_bar:
                dataframe.loc[idx, "enter_long"] = 1
                dataframe.loc[idx, "enter_tag"] = "ppo_long"
            elif action == int(Actions.OPEN_SHORT) and has_executable_entry_bar:
                dataframe.loc[idx, "enter_short"] = 1
                dataframe.loc[idx, "enter_tag"] = "ppo_short"
            elif action == int(Actions.CLOSE) and has_next_bar:
                dataframe.loc[idx, "exit_long"] = 1
                dataframe.loc[idx, "exit_short"] = 1
                dataframe.loc[idx, "exit_tag"] = "ppo_close"

            pending = {"action": pending_action, "size": pending_size}

        self.signal_cache[cache_key] = {col: dataframe[col].copy() for col in signal_cols}
        logger.info(
            "[Run19FreqtradeAdapter] %s signals: long=%s short=%s close=%s",
            pair,
            int(dataframe["enter_long"].sum()),
            int(dataframe["enter_short"].sum()),
            int(((dataframe["exit_long"] == 1) | (dataframe["exit_short"] == 1)).sum()),
        )
        return dataframe

    def stake_amount(
        self,
        pair: str,
        current_time,
        min_stake: Optional[float],
        max_stake: float,
    ) -> float:
        record = self.find_entry_record(pair, current_time)
        model_size = abs(float(record.get("size", 1.0))) if record else 1.0
        stake = self.base_stake * model_size
        if min_stake is not None:
            stake = max(stake, float(min_stake))
        return float(min(stake, float(max_stake)))


__all__ = ["Run19FreqtradeAdapter"]
