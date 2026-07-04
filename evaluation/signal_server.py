"""
PPO 模型信号服务 - FastAPI
freqtrade 策略通过 HTTP 调用此服务获取交易信号

启动: source ~/freqenv/bin/activate && python signal_server.py
默认端口: 8765
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import threading
import time
import logging

from live_trader import LiveTrader, Position

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPO Signal Server")

# ------------------------------------------------------------------
# 全局模型实例（单例）
# ------------------------------------------------------------------
trader: Optional[LiveTrader] = None
_lock = threading.Lock()

def get_trader() -> LiveTrader:
    global trader
    if trader is None:
        trader = LiveTrader(
            checkpoint='checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt',
            config='configs/e2e_ppo_run19_sl15_valrandom.json',
        )
    return trader


# ------------------------------------------------------------------
# 请求/响应模型
# ------------------------------------------------------------------
class OHLCVBar(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class PositionInfo(BaseModel):
    state: str = "empty"        # "empty" / "long" / "short"
    entry_price: float = 0.0
    size: float = 0.0
    holding_steps: int = 0
    empty_steps: int = 0
    peak_return: float = 0.0
    peak_step: int = 0
    unrealized_pnl: float = 0.0

class SymbolRequest(BaseModel):
    symbol: str
    ohlcv: List[OHLCVBar]       # 至少 181 根，按时间升序
    position: PositionInfo = PositionInfo()

class BatchRequest(BaseModel):
    symbols: List[SymbolRequest]

class SignalResponse(BaseModel):
    symbol: str
    action: str                 # wait / long / short / close / hold
    size: float                 # 0~1
    stop_pct: float             # 止损比例，如 -0.033
    stop_price: float           # 动态止损价（0表示空仓）
    raw_size: float             # 模型原始输出

class BatchResponse(BaseModel):
    signals: List[SignalResponse]
    timestamp: float


# ------------------------------------------------------------------
# 接口
# ------------------------------------------------------------------
@app.on_event("startup")
def startup():
    logger.info("Loading model...")
    get_trader()
    logger.info("Model ready.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": trader is not None}

@app.post("/predict", response_model=BatchResponse)
def predict(req: BatchRequest):
    t = get_trader()

    ohlcv_dict: Dict[str, np.ndarray] = {}
    positions: Dict[str, Position] = {}

    for item in req.symbols:
        if len(item.ohlcv) < t.seq_len + 1:
            raise HTTPException(
                status_code=400,
                detail=f"{item.symbol}: need >= {t.seq_len+1} bars, got {len(item.ohlcv)}"
            )
        arr = np.array(
            [[b.open, b.high, b.low, b.close, b.volume] for b in item.ohlcv],
            dtype=np.float32
        )
        ohlcv_dict[item.symbol] = arr
        p = item.position
        positions[item.symbol] = Position(
            state=p.state,
            entry_price=p.entry_price,
            size=p.size,
            holding_steps=p.holding_steps,
            empty_steps=p.empty_steps,
            peak_return=p.peak_return,
            peak_step=p.peak_step,
            unrealized_pnl=p.unrealized_pnl,
        )

    with _lock:
        signals = t.predict(ohlcv_dict, positions)

    result = []
    for sym, sig in signals.items():
        result.append(SignalResponse(
            symbol=sym,
            action=sig.action,
            size=sig.size,
            stop_pct=sig.stop_pct,
            stop_price=sig.stop_price,
            raw_size=sig.raw_size,
        ))

    return BatchResponse(signals=result, timestamp=time.time())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
