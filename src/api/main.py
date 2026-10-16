import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel

from src.database.session import engine, Base, get_db
from src.database.models import PredictionRecord, AlertRecord, TradeRecord, PositionRecord
from src.config.settings import settings
from src.api.price_stream import binance_price_generator, refresh_usdt_rate

logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Bitcoin AI Market Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    id: int
    symbol: str
    timestamp: str
    horizon: str
    price: float
    probability_down: float
    probability_neutral: float
    probability_up: float
    predicted_class: str
    confidence: float
    model_version: str
    actual_outcome: Optional[str]
    actual_return: Optional[float]
    
    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    id: int
    timestamp: str
    action: str
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float
    status: str
    
    class Config:
        from_attributes = True


class TradeResponse(BaseModel):
    id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity_btc: float
    position_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    balance_after: Optional[float] = None

    class Config:
        from_attributes = True


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/v1/prediction/latest", response_model=PredictionResponse)
def get_latest_prediction(db: Session = Depends(get_db)):
    prediction = db.query(PredictionRecord).order_by(desc(PredictionRecord.timestamp)).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions found")
    # format timestamp manually for pydantic, appending 'Z' since SQLite loses the timezone
    res = prediction.__dict__
    ts = res["timestamp"].isoformat()
    res["timestamp"] = ts if ts.endswith(('Z', '+00:00')) else f"{ts}Z"
    return res


@app.get("/api/v1/prediction/history", response_model=List[PredictionResponse])
def get_prediction_history(limit: int = 100, db: Session = Depends(get_db)):
    predictions = db.query(PredictionRecord).order_by(desc(PredictionRecord.timestamp)).limit(limit).all()
    results = []
    for p in predictions:
        res = p.__dict__
        ts = res["timestamp"].isoformat()
        res["timestamp"] = ts if ts.endswith(('Z', '+00:00')) else f"{ts}Z"
        results.append(res)
    return results


@app.get("/api/v1/alerts/latest", response_model=Optional[AlertResponse])
def get_latest_alert(db: Session = Depends(get_db)):
    alert = db.query(AlertRecord).order_by(desc(AlertRecord.timestamp)).first()
    if not alert:
        return None
    res = alert.__dict__
    ts = res["timestamp"].isoformat()
    res["timestamp"] = ts if ts.endswith(('Z', '+00:00')) else f"{ts}Z"
    return res


@app.get("/api/v1/alerts/history", response_model=List[AlertResponse])
def get_alert_history(limit: int = 50, db: Session = Depends(get_db)):
    alerts = db.query(AlertRecord).order_by(desc(AlertRecord.timestamp)).limit(limit).all()
    results = []
    for a in alerts:
        res = a.__dict__
        ts = res["timestamp"].isoformat()
        res["timestamp"] = ts if ts.endswith(('Z', '+00:00')) else f"{ts}Z"
        results.append(res)
    return results


@app.get("/api/v1/trades/history", response_model=List[TradeResponse])
def get_trades_history(limit: int = 50, db: Session = Depends(get_db)):
    trades = db.query(TradeRecord).order_by(desc(TradeRecord.closed_at)).limit(limit).all()
    results = []
    for t in trades:
        res = {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity_btc": t.quantity_btc,
            "position_usd": t.position_usd or 0.0,
            "pnl_usd": t.pnl_usd,
            "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            "balance_after": t.balance_after,
        }
        results.append(res)
    return results


@app.post("/api/v1/trading/emergency-stop")
def emergency_stop(db: Session = Depends(get_db)):
    """Emergency kill-switch to liquidate open paper position immediately."""
    from src.prediction.worker import _worker_instance
    price = _last_tick["price_usd"] if _last_tick else 0.0

    if _worker_instance is not None:
        return _worker_instance.trader.emergency_close(current_price=price)

    # Process-isolated fallback: close open position directly from DB
    open_pos = db.query(PositionRecord).first()
    if not open_pos:
        return {"status": "no_open_position"}

    exit_price = price if price > 0 else open_pos.entry_price
    if open_pos.side == "BUY":
        pnl_usd = (exit_price - open_pos.entry_price) * open_pos.quantity_btc
    else:
        pnl_usd = (open_pos.entry_price - exit_price) * open_pos.quantity_btc
    pnl_pct = (pnl_usd / open_pos.usd_value * 100) if open_pos.usd_value > 0 else 0.0

    latest_trade = db.query(TradeRecord).order_by(desc(TradeRecord.closed_at)).first()
    prev_balance = latest_trade.balance_after if latest_trade and latest_trade.balance_after else 1000.0
    balance_after = round(prev_balance + pnl_usd, 2)

    closed_record = TradeRecord(
        symbol=open_pos.symbol,
        side=open_pos.side,
        entry_price=open_pos.entry_price,
        exit_price=exit_price,
        quantity_btc=open_pos.quantity_btc,
        position_usd=open_pos.usd_value,
        pnl_usd=round(pnl_usd, 4),
        pnl_pct=round(pnl_pct, 3),
        exit_reason="emergency_stop",
        opened_at=open_pos.opened_at,
        closed_at=datetime.now(timezone.utc),
        balance_after=balance_after,
    )
    db.add(closed_record)
    db.delete(open_pos)
    db.commit()
    return {"status": "closed", "trade": {"exit_price": exit_price, "pnl_usd": pnl_usd, "reason": "emergency_stop"}}


@app.get("/api/v1/model/status")
def get_model_status():
    from src.models.registry import ModelRegistry
    registry = ModelRegistry()
    return registry.get_model(settings.model.version) or {"status": "not_found"}


@app.get("/api/v1/portfolio")
def get_portfolio(db: Session = Depends(get_db)):
    """Returns the live paper trading portfolio — balance, P&L, open position, trade history."""
    try:
        from src.prediction.worker import _worker_instance
        latest_price = _last_tick["price_usd"] if _last_tick else None
        if _worker_instance is not None:
            return _worker_instance.trader.portfolio_summary(current_price=latest_price)
    except Exception:
        pass

    # Fallback / process-isolated read from DB
    trades = db.query(TradeRecord).order_by(desc(TradeRecord.closed_at)).limit(20).all()
    initial_cap = 1000.0
    latest_trade = trades[0] if trades else None
    current_balance = latest_trade.balance_after if latest_trade and latest_trade.balance_after else initial_cap
    total_trades = db.query(TradeRecord).count()
    winning_trades = db.query(TradeRecord).filter(TradeRecord.pnl_usd > 0).count()
    total_profit = sum(t.pnl_usd for t in db.query(TradeRecord).filter(TradeRecord.pnl_usd > 0).all())
    total_loss = sum(abs(t.pnl_usd) for t in db.query(TradeRecord).filter(TradeRecord.pnl_usd < 0).all())
    pf = (total_profit / total_loss) if total_loss > 0 else (999.0 if total_profit > 0 else 0.0)

    trade_dicts = []
    for t in trades:
        trade_dicts.append({
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity_btc": t.quantity_btc,
            "pnl_usd": t.pnl_usd,
            "pnl_pct": t.pnl_pct,
            "reason": t.exit_reason,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            "balance_after": t.balance_after,
        })

    peak_balance = max([initial_cap] + [t.balance_after for t in trades if t.balance_after]) if trades else initial_cap
    max_dd = ((peak_balance - current_balance) / peak_balance * 100) if peak_balance > 0 else 0.0

    # Query active position from DB
    open_pos = db.query(PositionRecord).first()
    open_pos_dict = None
    if open_pos:
        cur_price = latest_price if latest_price else open_pos.entry_price
        if open_pos.side == "BUY":
            unrealized = (cur_price - open_pos.entry_price) * open_pos.quantity_btc
        else:
            unrealized = (open_pos.entry_price - cur_price) * open_pos.quantity_btc
        unrealized_pct = (unrealized / open_pos.usd_value * 100) if open_pos.usd_value > 0 else 0.0
        open_pos_dict = {
            "side": open_pos.side,
            "entry_price": open_pos.entry_price,
            "quantity_btc": open_pos.quantity_btc,
            "usd_value": open_pos.usd_value,
            "stop_loss_price": open_pos.stop_loss_price,
            "opened_at": open_pos.opened_at.isoformat() if open_pos.opened_at else datetime.now(timezone.utc).isoformat(),
            "unrealized_pnl_usd": round(unrealized, 2),
            "unrealized_pnl_pct": round(unrealized_pct, 2),
        }

    return {
        "mode": "PAPER_TRADING",
        "initial_capital_usd": initial_cap,
        "balance_usd": round(current_balance, 2),
        "peak_balance_usd": round(peak_balance, 2),
        "total_pnl_usd": round(current_balance - initial_cap, 2),
        "total_return_pct": round((current_balance / initial_cap - 1) * 100, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate_pct": round((winning_trades / total_trades * 100), 1) if total_trades > 0 else 0.0,
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max(0.0, max_dd), 2),
        "open_position": open_pos_dict,
        "recent_trades": trade_dicts,
    }

# Register alias routes without '/api' prefix so mounting at '/api' works seamlessly
for _r in list(app.routes):
    if hasattr(_r, 'path') and _r.path.startswith('/api/v1/'):
        _subpath = _r.path[len('/api'):]
        app.add_api_route(_subpath, _r.endpoint, methods=_r.methods, response_model=getattr(_r, 'response_model', None))

# Live Price WebSocket
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages multiple connected dashboard WebSocket clients."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        logger.info(f"WS client connected. Total: {len(self.active_connections)}")

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)
        logger.info(f"WS client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections.remove(ws)


manager = ConnectionManager()

# Background task: single Binance connection, shared across all WS clients
_price_stream_task: asyncio.Task | None = None
_last_tick: dict | None = None


async def _run_price_stream():
    """Connects once to Binance and fans out to all dashboard clients."""
    global _last_tick
    async for tick in binance_price_generator():
        _last_tick = tick
        payload = json.dumps({
            "price_usdt": tick["price_usdt"],
            "price_usd": tick["price_usd"],
            "usdt_usd_rate": tick["usdt_usd_rate"],
            "timestamp": tick["timestamp"],
            "is_buyer_maker": tick["is_buyer_maker"],
        })
        if manager.active_connections:
            await manager.broadcast(payload)


@app.on_event("startup")
async def startup_event():
    global _price_stream_task
    # Start USDT→USD rate refresh (every 30 seconds)
    asyncio.create_task(refresh_usdt_rate())
    # Start Binance price stream
    _price_stream_task = asyncio.create_task(_run_price_stream())
    logger.info("Binance live price stream and USDT rate refresh started.")


@app.websocket("/ws/price")
async def websocket_price(ws: WebSocket):
    # Accept unconditionally — CORS middleware does not cover WebSocket upgrades
    await manager.connect(ws)
    # Send last known price immediately so client doesn't wait for next trade
    if _last_tick:
        await ws.send_text(json.dumps({
            "price_usdt": _last_tick["price_usdt"],
            "price_usd": _last_tick["price_usd"],
            "usdt_usd_rate": _last_tick["usdt_usd_rate"],
            "timestamp": _last_tick["timestamp"],
            "is_buyer_maker": _last_tick["is_buyer_maker"],
        }))
    try:
        while True:
            # Keep alive; data is pushed from the price stream background task
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        manager.disconnect(ws)
