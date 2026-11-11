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


@app.api_route("/", methods=["GET", "HEAD"])
def root_check():
    return {"status": "ok", "app": "Bitcoin AI Market Prediction API"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}


def _format_iso_utc(dt) -> str:
    if dt is None:
        return datetime.now(timezone.utc).isoformat()
    ts = dt.isoformat()
    return ts if ts.endswith(('Z', '+00:00')) else f"{ts}Z"


def _serialize_prediction(p: PredictionRecord) -> dict:
    return {
        "id": p.id,
        "symbol": p.symbol,
        "timestamp": _format_iso_utc(p.timestamp),
        "horizon": p.horizon,
        "price": p.price,
        "probability_down": p.probability_down,
        "probability_neutral": p.probability_neutral,
        "probability_up": p.probability_up,
        "predicted_class": p.predicted_class,
        "confidence": p.confidence,
        "model_version": p.model_version,
        "actual_outcome": p.actual_outcome,
        "actual_return": p.actual_return,
    }


def _serialize_alert(a: AlertRecord) -> dict:
    return {
        "id": a.id,
        "timestamp": _format_iso_utc(a.timestamp),
        "action": a.action,
        "entry_price": a.entry_price,
        "take_profit": a.take_profit,
        "stop_loss": a.stop_loss,
        "confidence": a.confidence,
        "status": a.status,
    }


@app.get("/api/v1/prediction/latest", response_model=PredictionResponse)
def get_latest_prediction(db: Session = Depends(get_db)):
    prediction = db.query(PredictionRecord).order_by(desc(PredictionRecord.timestamp)).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions found")
    return _serialize_prediction(prediction)


@app.get("/api/v1/prediction/history", response_model=List[PredictionResponse])
def get_prediction_history(limit: int = 100, db: Session = Depends(get_db)):
    predictions = db.query(PredictionRecord).order_by(desc(PredictionRecord.timestamp)).limit(limit).all()
    return [_serialize_prediction(p) for p in predictions]


@app.get("/api/v1/alerts/latest", response_model=Optional[AlertResponse])
def get_latest_alert(db: Session = Depends(get_db)):
    alert = db.query(AlertRecord).order_by(desc(AlertRecord.timestamp)).first()
    if not alert:
        return None
    return _serialize_alert(alert)


@app.get("/api/v1/alerts/history", response_model=List[AlertResponse])
def get_alert_history(limit: int = 50, db: Session = Depends(get_db)):
    alerts = db.query(AlertRecord).order_by(desc(AlertRecord.timestamp)).limit(limit).all()
    return [_serialize_alert(a) for a in alerts]


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
    prev_balance = latest_trade.balance_after if latest_trade and latest_trade.balance_after else 100000.0
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


@app.api_route("/api/v1/notification/test-telegram", methods=["GET", "POST"])
def test_telegram_notification():
    """
    Test endpoint to verify Telegram Bot configuration and deliver an instant test alert.
    Returns exact diagnostics: whether TOKEN/CHAT_ID are present and what Telegram's API responds.
    """
    from src.notification.notifier import get_telegram_creds, send_telegram_alert_sync
    token, chat_id = get_telegram_creds()

    token_preview = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else ("SET" if token else "MISSING")
    chat_id_preview = f"{chat_id[:4]}***" if len(chat_id) > 4 else ("SET" if chat_id else "MISSING")

    diag = {
        "TELEGRAM_BOT_TOKEN_present": bool(token),
        "TELEGRAM_BOT_TOKEN_preview": token_preview,
        "TELEGRAM_CHAT_ID_present": bool(chat_id),
        "TELEGRAM_CHAT_ID_preview": chat_id_preview,
    }

    if not token:
        return {
            "success": False,
            "error": "TELEGRAM_BOT_TOKEN is not set in Render Environment Variables.",
            "diagnostics": diag,
            "action_required": "In Render Dashboard -> Environment -> add variable TELEGRAM_BOT_TOKEN with the token from @BotFather, then click 'Save, rebuild, and deploy'."
        }

    if not chat_id:
        return {
            "success": False,
            "error": "TELEGRAM_CHAT_ID is not set in Render Environment Variables.",
            "diagnostics": diag,
            "action_required": "In Telegram: 1) Open @userinfobot and tap Start to get your numeric ID. 2) Open your trading bot and tap Start. 3) In Render Dashboard -> Environment -> add variable TELEGRAM_CHAT_ID with your numeric ID, then click 'Save, rebuild, and deploy'."
        }

    test_msg = (
        "🤖 *Bitcoin Autonomous Trading Bot*\n\n"
        "✅ *Telegram alerts are successfully connected!*\n"
        "You will now receive instant push notifications whenever trades are opened, "
        "trailing stops adjust, or profits are harvested."
    )
    result = send_telegram_alert_sync(test_msg)
    return {
        "success": result.get("success", False),
        "diagnostics": diag,
        "telegram_api_result": result,
    }


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
    initial_cap = 100000.0
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
            "id": t.id,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity_btc": t.quantity_btc,
            "position_usd": t.position_usd or round((t.entry_price or 0.0) * (t.quantity_btc or 0.0), 2),
            "pnl_usd": t.pnl_usd,
            "pnl_pct": t.pnl_pct,
            "reason": t.exit_reason,
            "exit_reason": t.exit_reason,
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
            tp_price = open_pos.entry_price * 1.010
        else:
            unrealized = (open_pos.entry_price - cur_price) * open_pos.quantity_btc
            tp_price = open_pos.entry_price * 0.990
        unrealized_pct = (unrealized / open_pos.usd_value * 100) if open_pos.usd_value > 0 else 0.0
        open_pos_dict = {
            "side": open_pos.side,
            "entry_price": open_pos.entry_price,
            "quantity_btc": open_pos.quantity_btc,
            "usd_value": open_pos.usd_value,
            "stop_loss_price": open_pos.stop_loss_price,
            "take_profit_price": round(tp_price, 2),
            "highest_price": open_pos.highest_price,
            "lowest_price": open_pos.lowest_price,
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


class ManualTradeRequest(BaseModel):
    side: str = "BUY"
    price: Optional[float] = None
    reason: str = "manual_execution"


class ManualCloseRequest(BaseModel):
    exit_price: Optional[float] = None
    reason: str = "manual_withdrawal"


@app.post("/api/v1/trade/execute")
def execute_manual_trade(req: ManualTradeRequest, db: Session = Depends(get_db)):
    """Opens a 1.0 BTC paper position cleanly on the Render backend and persists to Neon DB."""
    from src.prediction.worker import _worker_instance
    side = "SELL" if req.side.upper() == "SELL" else "BUY"
    price = req.price or (_last_tick["price_usd"] if _last_tick else 0.0)
    if not price or price <= 0:
        raise HTTPException(status_code=400, detail="Live market price unavailable")

    if _worker_instance is not None:
        if _worker_instance.trader.risk_manager.open_position is not None:
            raise HTTPException(status_code=400, detail="Position already open")
        signal = "UP" if side == "BUY" else "DOWN"
        res = _worker_instance.trader.execute_signal(signal=signal, confidence=0.80, current_price=float(price))
        if _worker_instance.trader.risk_manager.open_position is not None:
            _worker_instance.sync_open_position_to_db()
            try:
                import threading
                from src.notification.notifier import notify_trade_event_sync
                pos = _worker_instance.trader.risk_manager.open_position
                threading.Thread(
                    target=notify_trade_event_sync,
                    args=("OPEN", {
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "quantity_btc": pos.quantity_btc,
                        "reason": req.reason,
                        "stop_loss": pos.stop_loss_price,
                        "take_profit": pos.entry_price * (1.010 if pos.side == "BUY" else 0.990),
                    }),
                    daemon=True,
                ).start()
            except Exception:
                pass
        return {"success": True, "result": res, "portfolio": _worker_instance.trader.portfolio_summary(current_price=float(price))}

    raise HTTPException(status_code=503, detail="Trading worker initializing")


@app.post("/api/v1/trade/close")
def close_manual_trade(req: ManualCloseRequest, db: Session = Depends(get_db)):
    """Closes the currently open position at live price and saves to Neon DB."""
    from src.prediction.worker import _worker_instance
    price = req.exit_price or (_last_tick["price_usd"] if _last_tick else 0.0)
    if _worker_instance is not None:
        pos = _worker_instance.trader.risk_manager.open_position
        if not pos:
            return {"success": False, "status": "no_open_position"}
        exit_p = float(price) if price and price > 0 else pos.entry_price
        closed = _worker_instance.trader._record_closed_trade(req.reason, exit_p)
        _worker_instance.sync_open_position_to_db()
        return {"success": True, "trade": closed, "portfolio": _worker_instance.trader.portfolio_summary(current_price=exit_p)}

    return emergency_stop(db=db)

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
        try:
            from src.prediction.worker import _worker_instance
            if _worker_instance is not None:
                _worker_instance.on_price_tick(tick["price_usd"])
        except Exception as e:
            logger.error(f"Error evaluating live tick: {e}")

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

    # Start Binance Live Prediction & Paper Trading Worker
    def _run_worker():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from src.prediction.worker import LiveWorker
            import src.prediction.worker as _worker_mod
            worker = LiveWorker()
            _worker_mod._worker_instance = worker
            logger.info("LiveWorker initialized and running Binance analysis.")
            loop.run_until_complete(worker.run())
        except Exception as e:
            logger.error(f"LiveWorker startup error: {e}")

    import threading
    worker_thread = threading.Thread(target=_run_worker, daemon=True)
    worker_thread.start()


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
