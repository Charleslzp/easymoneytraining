"""
交易会话管理器
每个交易对一个独立会话，管理滑动窗口、状态和推理
"""
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import logging

from .service import PPOTraderService, Position, States
from .utils import Actions

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """会话状态快照（可序列化）"""
    session_id: str
    pair: str
    created_at: str
    last_update: str

    # 持仓状态
    position_state: int  # 0=EMPTY, 1=LONG, 2=SHORT
    entry_price: float
    entry_step: int
    holding_steps: int
    empty_steps: int
    peak_return: float
    max_drawdown: float
    peak_step: int
    last_unrealized: float
    _entry_empty_steps: int
    stop_loss_pct: float

    # 统计信息
    total_bars: int
    total_trades: int
    is_active: bool

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'SessionState':
        """从字典创建"""
        data = dict(data)
        data.setdefault("stop_loss_pct", 0.0)
        return cls(**data)


class TradingSession:
    """
    单个交易对的会话

    功能：
    - 滑动窗口管理（自动维护观察窗口）
    - 状态管理（持仓、步数等）
    - 推理接口
    - 状态快照和恢复
    """

    def __init__(self, session_id: str, pair: str, service: PPOTraderService):
        """
        Args:
            session_id: 会话ID
            pair: 交易对名称
            service: PPO交易服务实例
        """
        self.session_id = session_id
        self.pair = pair
        self.service = service
        self.created_at = datetime.now().isoformat()
        self.last_update = self.created_at
        self.is_active = True

        # 统计信息
        self.total_bars = 0
        self.total_trades = 0

        logger.info(f"[Session {session_id}] Created for {pair}")

    def add_bar(self, ohlcv: np.ndarray) -> bool:
        """
        添加新K线到滑动窗口

        Args:
            ohlcv: shape (6,) - [timestamp, open, high, low, close, volume]

        Returns:
            是否已准备好推理
        """
        ready = self.service.add_bar(ohlcv)
        self.total_bars += 1
        self.last_update = datetime.now().isoformat()
        return ready

    def is_ready(self) -> bool:
        """
        检查是否准备好进行推理

        Returns:
            是否准备好
        """
        return len(self.service.feature_buffer) >= self.service.min_bars

    def predict(self) -> Tuple[int, float, float]:
        """
        执行推理

        Returns:
            (action, size, stop_loss_pct)
        """
        if not self.is_active:
            raise RuntimeError(f"Session {self.session_id} is not active")

        return self.service.predict()

    def execute_action(
        self,
        action: int,
        size: float = 0.0,
        entry_price: Optional[float] = None,
        bar_ohlcv: Optional[np.ndarray] = None,
        stop_loss_pct: Optional[float] = None,
        force_close_end: bool = False,
    ) -> dict:
        """
        执行动作并更新状态

        Args:
            action: 动作ID

        Returns:
            执行结果
        """
        result = self.service.execute_action(
            action,
            size=size,
            entry_price=entry_price,
            bar_ohlcv=bar_ohlcv,
            stop_loss_pct=stop_loss_pct,
            force_close_end=force_close_end,
        )

        # 统计交易次数
        if result.get('opened', False) or (
            result.get('executed', False) and int(action) in [int(Actions.OPEN_LONG), int(Actions.OPEN_SHORT)]
        ):
            self.total_trades += 1

        self.last_update = datetime.now().isoformat()
        return result

    def force_close(self) -> bool:
        """
        外部强制平仓（同步状态）

        Returns:
            是否更新了状态
        """
        updated = self.service.force_close_position()
        if updated:
            logger.info(f"[Session {self.session_id}] Force closed position for {self.pair}")
            self.last_update = datetime.now().isoformat()
        return updated

    def get_position_state(self) -> Dict:
        """获取当前持仓状态"""
        return self.service.get_position_state()

    def get_session_state(self) -> SessionState:
        """
        获取会话状态快照

        Returns:
            可序列化的状态快照
        """
        pos = self.service.position

        return SessionState(
            session_id=self.session_id,
            pair=self.pair,
            created_at=self.created_at,
            last_update=self.last_update,
            position_state=pos.state,
            entry_price=pos.entry_price,
            entry_step=pos.entry_step,
            holding_steps=pos.holding_steps,
            empty_steps=pos.empty_steps,
            peak_return=pos.peak_return,
            max_drawdown=pos.max_drawdown,
            peak_step=pos.peak_step,
            last_unrealized=pos.last_unrealized,
            _entry_empty_steps=pos._entry_empty_steps,
            stop_loss_pct=pos.stop_loss_pct,
            total_bars=self.total_bars,
            total_trades=self.total_trades,
            is_active=self.is_active
        )

    def terminate(self):
        """终止会话"""
        self.is_active = False
        logger.info(f"[Session {self.session_id}] Terminated for {self.pair}")


class SessionManager:
    """
    会话管理器

    功能：
    - 创建/销毁会话
    - 管理多个交易对的会话
    - 批量操作
    - 状态持久化
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        seq_length: Optional[int] = None,
        env_overrides: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            model_path: 模型路径
            device: 设备（cuda/cpu）
            seq_length: 观察窗口长度；传入后覆盖checkpoint config中的data.seq_length
        """
        self.model_path = model_path
        self.device = device
        self.seq_length = seq_length
        self.env_overrides = dict(env_overrides or {})
        self.sessions: Dict[str, TradingSession] = {}

        logger.info(f"[SessionManager] Initialized with model: {model_path}")
        logger.info(f"[SessionManager] Device: {device}")
        if seq_length is not None:
            logger.info(f"[SessionManager] Seq length override: {seq_length}")

    def create_session(self, pair: str, session_id: Optional[str] = None) -> str:
        """
        创建新会话

        Args:
            pair: 交易对名称
            session_id: 可选的会话ID，不提供则自动生成

        Returns:
            会话ID
        """
        if session_id is None:
            session_id = f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        # 创建独立的服务实例
        service = PPOTraderService(
            self.model_path,
            device=self.device,
            seq_length=self.seq_length,
            env_overrides=self.env_overrides,
        )

        # 创建会话
        session = TradingSession(session_id, pair, service)
        self.sessions[session_id] = session

        logger.info(f"[SessionManager] Created session {session_id} for {pair}")
        return session_id

    def get_or_create_session(self, pair: str, session_id: Optional[str] = None) -> TradingSession:
        """获取交易对活跃会话，不存在则创建。"""
        session = self.get_session_by_pair(pair)
        if session is not None:
            return session
        sid = self.create_session(pair, session_id=session_id)
        return self.get_session(sid)

    def get_session(self, session_id: str) -> TradingSession:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            交易会话
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        return self.sessions[session_id]

    def get_session_by_pair(self, pair: str) -> Optional[TradingSession]:
        """
        根据交易对获取活跃会话

        Args:
            pair: 交易对名称

        Returns:
            交易会话或None
        """
        for session in self.sessions.values():
            if session.pair == pair and session.is_active:
                return session
        return None

    def has_session(self, pair: str) -> bool:
        """
        检查交易对是否有活跃会话

        Args:
            pair: 交易对名称

        Returns:
            是否存在活跃会话
        """
        return self.get_session_by_pair(pair) is not None

    def get_session_id(self, pair: str) -> Optional[str]:
        """
        获取交易对的会话ID

        Args:
            pair: 交易对名称

        Returns:
            会话ID或None
        """
        session = self.get_session_by_pair(pair)
        return session.session_id if session else None

    def terminate_session(self, session_id: str):
        """
        终止会话

        Args:
            session_id: 会话ID
        """
        session = self.get_session(session_id)
        session.terminate()
        logger.info(f"[SessionManager] Terminated session {session_id}")

    def remove_session(self, session_id: str):
        """
        移除会话（从内存中删除）

        Args:
            session_id: 会话ID
        """
        if session_id in self.sessions:
            self.sessions[session_id].terminate()
            del self.sessions[session_id]
            logger.info(f"[SessionManager] Removed session {session_id}")

    def list_sessions(self, active_only: bool = False) -> Dict[str, SessionState]:
        """
        列出所有会话

        Args:
            active_only: 是否只列出活跃会话

        Returns:
            会话ID -> 会话状态的字典
        """
        result = {}
        for session_id, session in self.sessions.items():
            if active_only and not session.is_active:
                continue
            result[session_id] = session.get_session_state()
        return result

    def export_states(self, filepath: str):
        """
        导出所有会话状态到文件

        Args:
            filepath: 文件路径
        """
        states = {}
        for session_id, session in self.sessions.items():
            states[session_id] = session.get_session_state().to_dict()

        with open(filepath, 'w') as f:
            json.dump(states, f, indent=2)

        logger.info(f"[SessionManager] Exported {len(states)} session states to {filepath}")

    def get_active_count(self) -> int:
        """获取活跃会话数量"""
        return sum(1 for s in self.sessions.values() if s.is_active)

    def terminate_all(self):
        """终止所有会话"""
        for session in self.sessions.values():
            session.terminate()
        logger.info(f"[SessionManager] Terminated all {len(self.sessions)} sessions")

    def clear_all(self):
        """清空所有会话"""
        count = len(self.sessions)
        self.sessions.clear()
        logger.info(f"[SessionManager] Cleared all {count} sessions")

    # 便捷方法：直接通过交易对操作
    def add_bar(self, pair: str, bar: np.ndarray):
        """
        向交易对的会话添加K线

        Args:
            pair: 交易对名称
            bar: K线数据
        """
        session = self.get_session_by_pair(pair)
        if session is None:
            raise ValueError(f"No active session for {pair}")
        session.add_bar(bar)

    def is_ready(self, pair: str) -> bool:
        """
        检查交易对的会话是否准备好

        Args:
            pair: 交易对名称

        Returns:
            是否准备好
        """
        session = self.get_session_by_pair(pair)
        if session is None:
            return False
        return session.is_ready()

    def predict(self, pair: str) -> dict:
        """
        对交易对执行推理

        Args:
            pair: 交易对名称

        Returns:
            推理结果 {'action': int, 'size': float, 'stop_loss': float}
        """
        session = self.get_session_by_pair(pair)
        if session is None:
            raise ValueError(f"No active session for {pair}")

        action, size, stop_loss = session.predict()
        return {
            'action': action,
            'size': size,
            'stop_loss': stop_loss
        }

    def execute_action(
        self,
        pair: str,
        action: int,
        size: float = 0.0,
        entry_price: Optional[float] = None,
        bar_ohlcv: Optional[np.ndarray] = None,
        stop_loss_pct: Optional[float] = None,
        force_close_end: bool = False,
    ) -> dict:
        """
        对交易对执行动作

        Args:
            pair: 交易对名称
            action: 动作ID

        Returns:
            执行结果
        """
        session = self.get_session_by_pair(pair)
        if session is None:
            raise ValueError(f"No active session for {pair}")
        return session.execute_action(
            action,
            size=size,
            entry_price=entry_price,
            bar_ohlcv=bar_ohlcv,
            stop_loss_pct=stop_loss_pct,
            force_close_end=force_close_end,
        )

    def get_state(self, pair: str) -> dict:
        """
        获取交易对的状态

        Args:
            pair: 交易对名称

        Returns:
            状态字典
        """
        session = self.get_session_by_pair(pair)
        if session is None:
            raise ValueError(f"No active session for {pair}")

        state = session.get_session_state()
        # 转换为嵌套结构，与旧版本兼容
        return {
            'session_id': state.session_id,
            'pair': state.pair,
            'created_at': state.created_at,
            'last_update': state.last_update,
            'position': {
                'state': state.position_state,
                'entry_price': state.entry_price,
                'entry_step': state.entry_step,
                'holding_steps': state.holding_steps,
                'empty_steps': state.empty_steps,
                'peak_return': state.peak_return,
                'max_drawdown': state.max_drawdown,
                'peak_step': state.peak_step,
                'last_unrealized': state.last_unrealized,
                '_entry_empty_steps': state._entry_empty_steps,
                'stop_loss_pct': state.stop_loss_pct,
            },
            'total_bars': state.total_bars,
            'total_trades': state.total_trades,
            'is_active': state.is_active
        }

    def force_close(self, pair: str) -> bool:
        """
        强制平仓交易对

        Args:
            pair: 交易对名称

        Returns:
            是否更新了状态
        """
        session = self.get_session_by_pair(pair)
        if session is None:
            return False
        return session.force_close()
