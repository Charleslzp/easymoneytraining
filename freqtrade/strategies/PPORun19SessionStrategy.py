"""Freqtrade strategy that delegates run19 replay to the installed wheel."""

import importlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)

SEQ_LEN = 180
MODEL_PATH = "checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_project_import_path(path_value: str) -> bool:
    if path_value == "":
        return True
    try:
        resolved = Path(path_value).resolve()
    except Exception:
        return False
    return resolved in {PROJECT_ROOT, PROJECT_ROOT / "ppo_trader"}


def _import_ppo_trader_wheel():
    """Import ppo_trader from site-packages, not the local source tree."""
    loaded = sys.modules.get("ppo_trader")
    loaded_path = str(getattr(loaded, "__file__", "")) if loaded is not None else ""
    if (
        loaded is not None
        and "site-packages" in loaded_path
        and hasattr(loaded, "SessionManager")
        and hasattr(loaded, "Run19FreqtradeAdapter")
    ):
        return loaded

    for name in list(sys.modules):
        if name == "ppo_trader" or name.startswith("ppo_trader."):
            del sys.modules[name]

    original_path = list(sys.path)
    try:
        sys.path = [p for p in original_path if not _is_project_import_path(p)]
        module = importlib.import_module("ppo_trader")
    finally:
        sys.path = original_path

    module_path = str(getattr(module, "__file__", ""))
    if "site-packages" not in module_path:
        raise ImportError(f"Expected ppo_trader wheel from site-packages, got {module_path}")
    if not hasattr(module, "SessionManager"):
        raise ImportError(f"Installed ppo_trader does not export SessionManager: {module_path}")
    if not hasattr(module, "Run19FreqtradeAdapter"):
        raise ImportError(f"Installed ppo_trader does not export Run19FreqtradeAdapter: {module_path}")
    return module


class PPORun19SessionStrategy(IStrategy):
    timeframe = "4h"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = SEQ_LEN

    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = True

    _adapter = None
    _initialized = False
    _ppo_module = None

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._adapter = None
        self._initialized = False
        self._ppo_module = None
        self._base_stake = float(self.config.get("ppo", {}).get("base_stake", 1000.0))
        logger.info("[PPORun19SessionStrategy] initialized, seq_len=%s", SEQ_LEN)

    def _init_manager(self) -> None:
        if self._initialized:
            return

        ppo = _import_ppo_trader_wheel()
        import torch

        ppo_cfg = self.config.get("ppo", {})
        model_path = ppo_cfg.get("model_path", MODEL_PATH)
        device_cfg = str(ppo_cfg.get("device", "auto")).lower()
        if device_cfg == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_cfg == "cpu" or device_cfg.startswith("cuda"):
            device = device_cfg
        else:
            device = "cpu"

        self._ppo_module = ppo
        self._adapter = ppo.Run19FreqtradeAdapter(
            model_path=model_path,
            device=device,
            seq_length=SEQ_LEN,
            base_stake=self._base_stake,
            stop_loss=ppo_cfg.get("stop_loss"),
            stop_loss_vol_multiple=ppo_cfg.get("stop_loss_vol_multiple"),
            take_profit=ppo_cfg.get("take_profit"),
            cache_tag=(
                "NATIVEALIGN_WHEEL_ONLY_V2_"
                f"SL{ppo_cfg.get('stop_loss', 'cfg')}_"
                f"ATR{ppo_cfg.get('stop_loss_vol_multiple', 'cfg')}_"
                f"TP{ppo_cfg.get('take_profit', 'cfg')}"
            ),
        )
        self._initialized = True
        logger.info("[PPORun19SessionStrategy] Run19FreqtradeAdapter initialized from %s", ppo.__file__)
        logger.info("[PPORun19SessionStrategy] model=%s device=%s seq_len=%s", model_path, device, SEQ_LEN)

    def _ensure_signals(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        self._init_manager()
        return self._adapter.ensure_signals(dataframe, pair)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self._ensure_signals(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self._ensure_signals(dataframe, metadata)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self._ensure_signals(dataframe, metadata)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        self._init_manager()
        return self._adapter.exit_reason(pair, trade.open_date_utc, current_time)

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: Optional[str],
        **kwargs,
    ) -> Optional[float]:
        self._init_manager()
        return self._adapter.exit_price(pair, trade.open_date_utc, current_time)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        self._init_manager()
        return self._adapter.stoploss_ratio(
            pair=pair,
            entry_time=trade.open_date_utc,
            current_time=current_time,
            current_rate=current_rate,
            open_rate=trade.open_rate,
            is_short=bool(trade.is_short),
            leverage=1.0,
        )

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        self._init_manager()
        return self._adapter.stake_amount(pair, current_time, min_stake, max_stake)

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return 1.0

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        return True
