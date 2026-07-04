"""交易环境配置"""
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """交易环境配置"""
    transaction_cost: float = 0.001       # 单边手续费 0.1%
    stop_loss: float = -0.04              # 止损 -4%
    take_profit: float = 0.08             # 止盈 8%
    max_holding_steps: int = 49
    max_empty_steps: int = 49
    min_holding_steps: int = 8
    min_empty_steps: int = 3
