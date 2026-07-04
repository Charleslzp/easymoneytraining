"""PPO交易服务 - 封装环境状态管理和模型推理的端到端服务"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .utils import Actions, States
from .config import EnvConfig


@dataclass
class Position:
    """持仓状态"""
    state: int = States.EMPTY  # EMPTY=0, LONG=1, SHORT=2
    size: float = 0.0
    entry_price: float = 0.0
    entry_step: int = 0
    holding_steps: int = 0
    empty_steps: int = 0
    peak_return: float = 0.0
    max_drawdown: float = 0.0
    peak_step: int = 0
    last_unrealized: float = 0.0
    _entry_empty_steps: int = 0
    stop_loss_pct: float = 0.0


class PPOTraderService:
    """PPO交易服务 - 端到端封装

    对外只需要提供OHLCV数据，内部自动管理：
    - 特征归一化（使用训练时的scaler）
    - 历史窗口维护（180根K线）
    - 持仓状态管理（empty_steps、holding_steps等）
    - 模型推理
    - 动作解码和止损计算
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        seq_length: Optional[int] = None,
        env_overrides: Optional[Dict[str, float]] = None,
    ):
        """初始化服务

        Args:
            model_path: 模型文件路径（.pt文件）
            device: 运行设备（cuda或cpu）
            seq_length: 序列长度（可选，如果提供则覆盖配置文件中的值）
        """
        self.device = device
        self.model_path = model_path

        # 加载模型
        from ppo_trader.model import StudentPolicy
        self.model = StudentPolicy.load(model_path, device=device)
        self.model.to(device)  # 确保模型在正确的设备上
        self.model.eval()

        # 加载配置
        model_dir = Path(model_path).parent
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        import json
        with open(config_path) as f:
            config = json.load(f)

        env_overrides = dict(env_overrides or {})
        for key in ("stop_loss", "stop_loss_vol_multiple", "take_profit"):
            value = env_overrides.get(key)
            if value is not None:
                config["env"][key] = float(value)

        # 过滤掉EnvConfig不支持的参数
        env_config_dict = config['env'].copy()

        # 先保存需要的参数（在过滤之前）
        self.entry_threshold = config['env'].get('entry_threshold', 0.2)
        self.static_stop_loss = config['env'].get('stop_loss', -0.10)
        self.stop_loss_vol_multiple = config['env'].get('stop_loss_vol_multiple', 0.0)
        self.reward_vol_floor = config['env'].get('reward_vol_floor', 0.008)
        self.position_scale = config['env'].get('position_scale', 1.0)

        # 移除不在EnvConfig中的参数
        unsupported_keys = [
            'stop_loss_vol_multiple', 'reward_vol_window', 'reward_vol_floor',
            'reward_hold_norm_denom', 'position_scale', 'long_only_market_groups',
            'potential_drawdown_weight', 'empty_reward_steps', 'empty_penalty_start',
            'entry_threshold', 'open_fee_penalty', 'potential_scale',
            'empty_first_step_bonus', 'empty_step_penalty', 'empty_late_penalty'
        ]
        for key in unsupported_keys:
            env_config_dict.pop(key, None)

        self.env_config = EnvConfig(**env_config_dict)
        # 如果显式传入seq_length参数，覆盖配置文件中的值
        if seq_length is not None:
            self.seq_length = seq_length
        else:
            self.seq_length = config['data']['seq_length']
        self.normalize_method = config['data'].get('normalize_method', 'returns')

        # 状态管理
        self.position = Position()
        self.step_count = 0
        self.feature_buffer = []  # 存储归一化后的特征
        self.price_buffer = []    # 存储价格（用于计算收益）

        # 最小K线数量要求
        self.min_bars = self.seq_length

    def reset(self):
        """重置状态（用于新的交易序列）"""
        self.position = Position()
        self.step_count = 0
        self.feature_buffer = []
        self.price_buffer = []

    def force_close_position(self):
        """
        强制关闭仓位 - 用于外部平仓时同步模型状态

        当外部系统关闭仓位时（止损、手动平仓、清算等），
        调用此方法将模型内部状态重置为空仓

        注意：这不会执行实际的交易操作，只是同步模型内部状态
        """
        if self.position.state != States.EMPTY:
            # 重置为空仓状态（重置Position的所有字段）
            self.position.state = States.EMPTY
            self.position.size = 0.0
            self.position.entry_price = 0.0
            self.position.entry_step = 0  # 补充：重置开仓步数
            self.position.holding_steps = 0
            self.position.empty_steps = 0
            self.position.peak_return = 0.0
            self.position.max_drawdown = 0.0
            self.position.peak_step = 0
            self.position.last_unrealized = 0.0
            self.position._entry_empty_steps = 0  # 补充：重置开仓前空仓步数
            self.position.stop_loss_pct = 0.0

            return True  # 返回True表示状态已更新
        return False  # 返回False表示本来就是空仓

    def _extract_features(self, ohlcv: np.ndarray) -> np.ndarray:
        """从OHLCV提取特征（与训练时一致）

        Args:
            ohlcv: shape (n, 6) - [timestamp, open, high, low, close, volume]

        Returns:
            features: shape (n, 5) - [open, high, low, close, volume]
        """
        # 只取OHLCV，去掉timestamp
        return ohlcv[:, 1:6].astype(np.float32)

    def _normalize_features(self, ohlcv: np.ndarray) -> np.ndarray:
        """归一化特征 - 使用与训练时相同的returns方法"""
        if self.normalize_method == 'returns':
            return self._normalize_returns(ohlcv)
        else:
            return self._normalize_zscore(ohlcv)

    def _normalize_returns(self, ohlcv: np.ndarray) -> np.ndarray:
        """Returns归一化 - 与data_loader.py中的实现一致"""
        normalized = np.zeros_like(ohlcv, dtype=np.float32)

        # OHLC: 计算returns
        for i in range(4):
            col = ohlcv[:, i]
            returns = np.zeros_like(col)
            returns[1:] = (col[1:] - col[:-1]) / (col[:-1] + 1e-8)
            returns[0] = 0.0
            normalized[:, i] = returns

        # Volume: log归一化
        volume = ohlcv[:, 4]
        log_volume = np.log1p(volume)
        mean_vol = log_volume.mean()
        std_vol = log_volume.std() + 1e-8
        normalized[:, 4] = (log_volume - mean_vol) / std_vol

        return normalized

    def _normalize_zscore(self, ohlcv: np.ndarray) -> np.ndarray:
        """Z-score归一化"""
        normalized = np.zeros_like(ohlcv, dtype=np.float32)

        for i in range(ohlcv.shape[1]):
            col = ohlcv[:, i]
            if i < 4:  # OHLC
                mean = col.mean()
                std = col.std() + 1e-8
            else:  # Volume
                mean = col.mean()
                std = col.std() + 1e-8
            normalized[:, i] = (col - mean) / std

        return normalized

    def _calculate_atr(self, ohlcv: np.ndarray, period: int = 14) -> float:
        """计算ATR（平均真实波幅）

        Args:
            ohlcv: shape (n, 5) - [open, high, low, close, volume]
            period: ATR周期，默认14

        Returns:
            ATR值（如果数据不足返回0）
        """
        if len(ohlcv) < 2:
            return 0.0

        # Native TradingEnv._atr(window=14) uses raw_ohlcva[start - 1:end + 1],
        # which yields up to 15 true-range samples for a 14-window setting.
        bars = ohlcv[-min(len(ohlcv), period + 2):]
        highs = bars[1:, 1]
        lows = bars[1:, 2]
        prev_closes = bars[:-1, 3]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
        )
        if tr.size == 0:
            return 0.0
        return float(np.mean(tr))

    def _calculate_dynamic_stop_loss(self, current_price: float) -> float:
        """计算动态止损百分比

        使用和 native TradingEnv 一致的 ATR 止损：
        stop_loss_vol_multiple > 0 时返回 -multiple * ATR%，如配置了负数
        static stop，则取两者中更接近 0 的值。

        Args:
            current_price: 当前价格

        Returns:
            止损百分比（负数，如-0.10表示-10%）
        """
        if self.stop_loss_vol_multiple <= 0:
            return float(self.static_stop_loss)

        # 从feature_buffer获取原始OHLCV数据
        ohlcv = np.array(self.feature_buffer)

        # 计算ATR
        atr = self._calculate_atr(ohlcv, period=14)

        if atr <= 0 or current_price <= 0:
            static_stop = float(self.static_stop_loss)
            if static_stop < 0.0:
                return static_stop
            return -self.stop_loss_vol_multiple * max(float(self.reward_vol_floor), 1e-4)

        atr_pct = max(atr / current_price, max(float(self.reward_vol_floor), 1e-4))
        atr_stop = -self.stop_loss_vol_multiple * atr_pct
        static_stop = float(self.static_stop_loss)
        if static_stop < 0.0:
            return max(static_stop, atr_stop)

        return atr_stop

    def _get_market_features(self) -> np.ndarray:
        """获取市场特征（最近seq_length根K线）

        Returns:
            shape (seq_length, 5) - 归一化后的OHLCV特征
        """
        if len(self.feature_buffer) < self.seq_length:
            # 不足时用0填充
            pad_len = self.seq_length - len(self.feature_buffer)
            padding = np.zeros((pad_len, 5), dtype=np.float32)
            features = np.vstack([padding, np.array(self.feature_buffer)])
        else:
            features = np.array(self.feature_buffer[-self.seq_length:])

        # 对整个序列进行归一化
        return self._normalize_features(features)

    def _get_position_info(self) -> np.ndarray:
        """获取持仓信息（6维向量）"""
        info = np.zeros(6, dtype=np.float32)

        if self.position.state == States.EMPTY:
            # 空仓：只有empty_steps归一化值
            info[4] = min(self.position.empty_steps / max(self.env_config.max_empty_steps, 1), 1.0)
            return info

        # 持仓：计算收益率等信息
        info[0] = float(self.position.size)

        # price_buffer存储的是close价格（浮点数）
        cur_price = self.price_buffer[-1]
        if self.position.state == States.LONG:
            raw_unrealized = (cur_price - self.position.entry_price) / self.position.entry_price
        else:
            raw_unrealized = (self.position.entry_price - cur_price) / self.position.entry_price
        unrealized = abs(float(self.position.size)) * raw_unrealized

        # 使用tanh软缩放
        info[1] = np.tanh(unrealized * 20.0)
        info[2] = np.tanh(self.position.peak_return * 20.0)
        info[3] = np.tanh(max(self.position.peak_return - unrealized, 0.0) * 20.0)
        info[4] = min(self.position.holding_steps / max(self.env_config.max_holding_steps, 1), 1.0)

        steps_since_peak = max(self.step_count - self.position.peak_step, 0)
        info[5] = min(steps_since_peak / max(self.env_config.max_holding_steps, 1), 1.0)

        return info

    def _get_action_mask(self) -> np.ndarray:
        """获取动作掩码（2维向量，用于FlatActions: WAIT=0, OPEN=1）"""
        mask = np.zeros(2, dtype=np.float32)

        if self.position.state == States.EMPTY:
            # 空仓时，检查是否可以开仓
            mask[0] = 1.0  # WAIT总是可用
            mask[1] = 1.0  # OPEN默认可用

            # MIN_EMPTY约束：空仓步数不足时不能开仓
            if self.position.empty_steps < self.env_config.min_empty_steps:
                mask[1] = 0.0
        else:
            mask[0] = 1.0
            mask[1] = 1.0
            if self.position.holding_steps < self.env_config.min_holding_steps:
                mask[0] = 0.0
            if self.position.holding_steps >= self.env_config.max_holding_steps:
                mask[0] = 1.0
                mask[1] = 0.0

        return mask

    def _get_current_return(self) -> float:
        """计算当前收益率"""
        if self.position.state == States.EMPTY:
            return 0.0

        # price_buffer存储的是close价格（浮点数）
        cur_price = self.price_buffer[-1]
        if self.position.state == States.LONG:
            raw_ret = (cur_price - self.position.entry_price) / self.position.entry_price
        else:
            raw_ret = (self.position.entry_price - cur_price) / self.position.entry_price
        return abs(float(self.position.size)) * raw_ret

    def _update_position_stats(self):
        """更新持仓统计（peak_return、drawdown等）"""
        if self.position.state == States.EMPTY:
            return

        cur_ret = self._get_current_return()

        if cur_ret > self.position.peak_return:
            self.position.peak_return = cur_ret
            self.position.peak_step = self.step_count

        dd = max(self.position.peak_return - cur_ret, 0.0)
        if dd > self.position.max_drawdown:
            self.position.max_drawdown = dd

    @staticmethod
    def _bar_ohlc(bar_ohlcv: Optional[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
        if bar_ohlcv is None:
            return None
        arr = np.asarray(bar_ohlcv, dtype=np.float64).reshape(-1)
        if arr.size >= 6:
            return float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])
        if arr.size >= 4:
            return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        raise ValueError("bar_ohlcv must be [ts, open, high, low, close, volume] or [open, high, low, close, ...]")

    def _position_return_at_price(self, price: float) -> float:
        if self.position.state == States.EMPTY or self.position.entry_price <= 0.0:
            return 0.0
        if self.position.state == States.LONG:
            raw_ret = (float(price) - self.position.entry_price) / self.position.entry_price
        else:
            raw_ret = (self.position.entry_price - float(price)) / self.position.entry_price
        return abs(float(self.position.size)) * raw_ret

    @staticmethod
    def _stop_price(entry_price: float, stop_loss_pct: float, state: int) -> float:
        entry = max(float(entry_price), 1e-8)
        stop = float(stop_loss_pct)
        if state == States.SHORT:
            return entry * (1.0 - stop)
        return entry * (1.0 + stop)

    def _reset_position(self) -> None:
        self.position.state = States.EMPTY
        self.position.size = 0.0
        self.position.entry_price = 0.0
        self.position.entry_step = 0
        self.position.holding_steps = 0
        self.position.empty_steps = 0
        self.position.peak_return = 0.0
        self.position.max_drawdown = 0.0
        self.position.peak_step = 0
        self.position.last_unrealized = 0.0
        self.position._entry_empty_steps = 0
        self.position.stop_loss_pct = 0.0

    def _close_position_at_price(
        self,
        price: float,
        exit_reason: str = "",
        force_close: bool = False,
        stop_loss_pct: Optional[float] = None,
    ) -> Dict:
        closing_state = int(self.position.state)
        closing_size = float(self.position.size)
        closing_entry = float(self.position.entry_price)
        closing_stop = (
            float(stop_loss_pct)
            if stop_loss_pct is not None and np.isfinite(float(stop_loss_pct))
            else float(self.position.stop_loss_pct)
        )
        pnl = self._position_return_at_price(price)
        pnl -= self.env_config.transaction_cost * 2.0 * abs(float(self.position.size))
        self._reset_position()
        return {
            'executed': True,
            'opened': False,
            'closed': True,
            'position_state': States.EMPTY,
            'entry_price': 0.0,
            'position_size': closing_size,
            'entry_state': closing_state,
            'opened_entry_price': closing_entry,
            'stop_loss_pct': closing_stop,
            'stop_price': self._stop_price(closing_entry, closing_stop, closing_state) if closing_entry > 0 else 0.0,
            'exit_price': float(price),
            'pnl': float(pnl),
            'force_close': bool(force_close),
            'exit_reason': str(exit_reason or ""),
        }

    def _update_position_stats_from_price(self, price: float) -> None:
        cur_ret = self._position_return_at_price(price)
        if cur_ret > self.position.peak_return:
            self.position.peak_return = cur_ret
            self.position.peak_step = self.step_count
        dd = max(self.position.peak_return - cur_ret, 0.0)
        if dd > self.position.max_drawdown:
            self.position.max_drawdown = dd
        self.position.last_unrealized = cur_ret

    def _process_open_position_bar(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        force_close_end: bool = False,
    ) -> Dict:
        self.position.holding_steps += 1
        stop_loss_pct = self._calculate_dynamic_stop_loss(bar_close)
        if not np.isfinite(stop_loss_pct):
            stop_loss_pct = float(self.position.stop_loss_pct)

        take_profit = float(getattr(self.env_config, "take_profit", 0.0))
        entry = max(float(self.position.entry_price), 1e-8)

        if self.position.state == States.LONG:
            if stop_loss_pct < 0.0:
                stop_px = entry * (1.0 + stop_loss_pct)
                if bar_open <= stop_px:
                    return self._close_position_at_price(bar_open, exit_reason="stop_loss", stop_loss_pct=stop_loss_pct)
                if bar_low <= stop_px:
                    return self._close_position_at_price(stop_px, exit_reason="stop_loss", stop_loss_pct=stop_loss_pct)
            if take_profit > 0.0:
                take_px = entry * (1.0 + take_profit)
                if bar_open >= take_px:
                    return self._close_position_at_price(bar_open, exit_reason="take_profit")
                if bar_high >= take_px:
                    return self._close_position_at_price(take_px, exit_reason="take_profit")
        else:
            if stop_loss_pct < 0.0:
                stop_px = entry * (1.0 - stop_loss_pct)
                if bar_open >= stop_px:
                    return self._close_position_at_price(bar_open, exit_reason="stop_loss", stop_loss_pct=stop_loss_pct)
                if bar_high >= stop_px:
                    return self._close_position_at_price(stop_px, exit_reason="stop_loss", stop_loss_pct=stop_loss_pct)
            if take_profit > 0.0:
                take_px = entry * (1.0 - take_profit)
                if bar_open <= take_px:
                    return self._close_position_at_price(bar_open, exit_reason="take_profit")
                if bar_low <= take_px:
                    return self._close_position_at_price(take_px, exit_reason="take_profit")

        self._update_position_stats_from_price(bar_close)
        if force_close_end:
            return self._close_position_at_price(bar_close, exit_reason="force_close", force_close=True)

        return {
            'executed': False,
            'position_state': self.position.state,
            'entry_price': self.position.entry_price,
            'pnl': 0.0,
            'stop_loss_pct': float(stop_loss_pct),
            'stop_price': self._stop_price(
                self.position.entry_price,
                stop_loss_pct,
                self.position.state,
            ),
            'force_close': False,
            'exit_reason': "",
        }

    def add_bar(self, ohlcv: np.ndarray) -> bool:
        """添加一根新K线

        Args:
            ohlcv: shape (6,) - [timestamp, open, high, low, close, volume]

        Returns:
            是否已准备好进行推理（至少有min_bars根K线）
        """
        # 提取特征（OHLCV，不归一化）
        features = self._extract_features(ohlcv.reshape(1, -1))[0]

        # 添加到缓冲区（存储原始特征）
        self.feature_buffer.append(features)
        # 只存储close价格（与工作版本一致）
        self.price_buffer.append(float(features[3]))  # close price

        # 严格的队列形式：始终保持seq_length根K线
        # 满300后，每进1根新的就去掉1根旧的
        if len(self.feature_buffer) > self.seq_length:
            self.feature_buffer.pop(0)  # 去掉最旧的
            self.price_buffer.pop(0)    # 同步去掉最旧的价格

        return len(self.feature_buffer) >= self.min_bars

    def predict(self) -> Tuple[int, float, float]:
        """执行推理并返回交易信号

        Returns:
            action: 动作（0=HOLD, 1=OPEN_LONG, 2=OPEN_SHORT, 3=CLOSE, 4=DO_NOTHING）
            position_size: 仓位大小（0.0-1.0）
            stop_loss_pct: 止损百分比（负数，如-0.02表示2%止损）
        """
        if len(self.feature_buffer) < self.min_bars:
            # 数据不足，返回DO_NOTHING
            return Actions.DO_NOTHING, 0.0, 0.0

        # 准备输入
        market_features = self._get_market_features()
        position_info = self._get_position_info()
        action_mask = self._get_action_mask()

        # 转换为tensor (确保float32类型)
        market_tensor = torch.from_numpy(market_features).float().unsqueeze(0).to(self.device)
        position_tensor = torch.from_numpy(position_info).float().unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(action_mask).float().unsqueeze(0).to(self.device)

        # position_states: 当前持仓状态 (0=EMPTY, 1=LONG, 2=SHORT)
        state_tensor = torch.tensor([self.position.state], dtype=torch.long, device=self.device)

        # 使用act_deterministic进行推理
        with torch.no_grad():
            output = self.model.act_deterministic(
                market_tensor,
                position_tensor,
                state_tensor,
                mask_tensor
            )  # (1, 2): [gate_action, size]

            gate_action = int(output[0, 0].item())
            size = float(output[0, 1].item())

        # 将gate_action转换为full action
        # gate_action的含义取决于当前持仓状态
        if self.position.state == States.EMPTY:
            # FlatActions: 0=WAIT, 1=OPEN
            if gate_action == 0:  # WAIT
                action = Actions.DO_NOTHING
            else:  # OPEN
                # Native two-head TradingEnv does not apply entry_threshold here.
                # It opens on gate=OPEN unless the signed size is exactly zero.
                if size > 0:
                    action = Actions.OPEN_LONG
                    size = abs(size)
                elif size < 0:
                    action = Actions.OPEN_SHORT
                    size = abs(size)
                else:
                    action = Actions.DO_NOTHING
                    size = 0.0
        else:
            # HoldActions: 0=CLOSE, 1=HOLD
            if gate_action == 0:  # CLOSE
                action = Actions.CLOSE
            else:  # HOLD
                action = Actions.HOLD
            size = 0.0  # 持仓时不需要size

        # 调试日志（每100步打印一次）
        if self.step_count % 100 == 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[PPO Service] Step {self.step_count}: state={self.position.state}, "
                       f"empty_steps={self.position.empty_steps}, gate={gate_action}, action={action}, size={size:.3f}, "
                       f"mask={action_mask}")

        # 计算动态止损
        current_price = self.price_buffer[-1] if len(self.price_buffer) > 0 else 0.0
        stop_loss_pct = self._calculate_dynamic_stop_loss(current_price)

        return action, size, stop_loss_pct

    def execute_action(
        self,
        action: int,
        size: float = 0.0,
        entry_price: Optional[float] = None,
        bar_ohlcv: Optional[np.ndarray] = None,
        stop_loss_pct: Optional[float] = None,
        force_close_end: bool = False,
    ) -> Dict:
        """执行动作并更新状态

        Args:
            action: 动作ID

        Returns:
            执行结果字典，包含：
            - executed: 是否成功执行
            - position_state: 当前持仓状态
            - entry_price: 开仓价格（如果是开仓动作）
            - pnl: 收益率（如果是平仓动作）
        """
        result = {
            'executed': False,
            'opened': False,
            'closed': False,
            'position_state': self.position.state,
            'entry_price': 0.0,
            'position_size': float(self.position.size),
            'entry_state': int(self.position.state),
            'opened_entry_price': 0.0,
            'stop_loss_pct': float(self.position.stop_loss_pct),
            'stop_price': 0.0,
            'exit_price': 0.0,
            'pnl': 0.0,
            'force_close': False,
            'exit_reason': "",
        }

        bar = self._bar_ohlc(bar_ohlcv)
        cur_price = bar[3] if bar is not None else (self.price_buffer[-1] if self.price_buffer else 0.0)

        if self.position.state == States.EMPTY:
            if action == Actions.OPEN_LONG or action == Actions.OPEN_SHORT:
                # 开仓
                scaled_size = abs(float(size)) * max(float(self.position_scale), 0.0)
                if not np.isfinite(scaled_size) or scaled_size == 0.0:
                    self.position.empty_steps += 1
                    result['position_size'] = float(self.position.size)
                    self.step_count += 1
                    return result

                self.position.state = States.LONG if action == Actions.OPEN_LONG else States.SHORT
                self.position.size = scaled_size if action == Actions.OPEN_LONG else -scaled_size
                self.position.entry_price = float(entry_price) if entry_price and entry_price > 0 else cur_price
                self.position.entry_step = self.step_count
                self.position.holding_steps = 0
                self.position._entry_empty_steps = self.position.empty_steps
                self.position.empty_steps = 0
                self.position.peak_return = 0.0
                self.position.max_drawdown = 0.0
                self.position.peak_step = self.step_count
                self.position.last_unrealized = 0.0
                self.position.stop_loss_pct = (
                    float(stop_loss_pct)
                    if stop_loss_pct is not None and np.isfinite(float(stop_loss_pct))
                    else self._calculate_dynamic_stop_loss(cur_price)
                )

                result['executed'] = True
                result['opened'] = True
                result['position_state'] = self.position.state
                result['entry_price'] = self.position.entry_price
                result['position_size'] = float(self.position.size)
                result['entry_state'] = int(self.position.state)
                result['opened_entry_price'] = self.position.entry_price
                result['stop_loss_pct'] = float(self.position.stop_loss_pct)
                result['stop_price'] = self._stop_price(
                    self.position.entry_price,
                    self.position.stop_loss_pct,
                    self.position.state,
                )
                opened_entry_price = self.position.entry_price
                opened_size = float(self.position.size)
                opened_state = int(self.position.state)

                if bar is not None:
                    bar_result = self._process_open_position_bar(
                        bar_open=bar[0],
                        bar_high=bar[1],
                        bar_low=bar[2],
                        bar_close=bar[3],
                        force_close_end=force_close_end,
                    )
                    if bar_result.get('executed'):
                        result.update(bar_result)
                        result['entry_price'] = opened_entry_price
                        result['opened'] = True
                        result['opened_entry_price'] = opened_entry_price
                        result['position_size'] = opened_size
                        result['entry_state'] = opened_state
                    else:
                        result['position_state'] = self.position.state
                        result['position_size'] = float(self.position.size)
                        result['stop_loss_pct'] = float(bar_result.get('stop_loss_pct', result['stop_loss_pct']))
                        result['stop_price'] = float(bar_result.get('stop_price', result['stop_price']))
            else:
                # DO_NOTHING
                self.position.empty_steps += 1
                result['position_size'] = float(self.position.size)
        else:
            # 强制平仓：超过最大持仓时间
            force_close = self.position.holding_steps >= self.env_config.max_holding_steps

            if action == Actions.CLOSE or force_close:
                # 平仓
                close_price = bar[0] if bar is not None else cur_price
                close_reason = "close_signal"
                result = self._close_position_at_price(close_price, exit_reason=close_reason, force_close=force_close)
                result['position_state'] = States.EMPTY
            else:
                # HOLD
                if bar is not None:
                    result = self._process_open_position_bar(
                        bar_open=bar[0],
                        bar_high=bar[1],
                        bar_low=bar[2],
                        bar_close=bar[3],
                        force_close_end=force_close_end,
                    )
                    if not result.get('executed'):
                        result['position_size'] = float(self.position.size)
                else:
                    self.position.holding_steps += 1
                    self._update_position_stats()
                    result['position_size'] = float(self.position.size)
                    result['stop_loss_pct'] = float(self.position.stop_loss_pct)
                    result['stop_price'] = self._stop_price(
                        self.position.entry_price,
                        self.position.stop_loss_pct,
                        self.position.state,
                    )

        self.step_count += 1
        return result

    def get_position_state(self) -> Dict:
        """获取当前持仓状态（用于监控和日志）"""
        return {
            'state': self.position.state,
            'state_name': ['EMPTY', 'LONG', 'SHORT'][self.position.state],
            'size': self.position.size,
            'entry_price': self.position.entry_price,
            'holding_steps': self.position.holding_steps,
            'empty_steps': self.position.empty_steps,
            'unrealized_pnl': self._get_current_return() if self.position.state != States.EMPTY else 0.0,
            'peak_return': self.position.peak_return,
            'max_drawdown': self.position.max_drawdown,
        }
