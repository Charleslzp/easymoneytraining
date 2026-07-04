"""
LiveTrader - 实盘推理封装
输入: 原始OHLCV序列 + 仓位状态
输出: action/size/stop_loss per symbol

用法:
    trader = LiveTrader(
        checkpoint='checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt',
        config='configs/e2e_ppo_run19_sl15_valrandom.json',
    )
    signals = trader.predict(ohlcv_dict, positions)
"""

from ppo_trader.ppo_trader.trader import (
    LiveTrader,
    Position,
    Signal,
    normalize_ohlcv,
    compute_atr,
    encode_pos_info,
)

__all__ = [
    "LiveTrader",
    "Position",
    "Signal",
    "normalize_ohlcv",
    "compute_atr",
    "encode_pos_info",
]
