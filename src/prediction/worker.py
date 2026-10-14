import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd

from src.data.collectors.live import BinanceLiveCollector
from src.data.collectors.historical import BinanceHistoricalCollector
from src.prediction.predictor import LivePredictor
from src.database.session import SessionLocal
from src.database.models import PredictionRecord, AlertRecord, TradeRecord, PositionRecord
from src.config.settings import settings
from src.trading.paper_trader import PaperTrader
from src.trading.risk_manager import RiskManager, Position

logger = logging.getLogger(__name__)

# Module-level reference so the API can read portfolio state
_worker_instance: "LiveWorker | None" = None


class LiveWorker:
    def __init__(self):
        self.collector = BinanceLiveCollector()
        self.historical = BinanceHistoricalCollector()
        self.predictor = LivePredictor()

        # Paper trading engine (PAPER_TRADE mode — no real money)
        risk_mgr = RiskManager(
            max_position_usd=100.0,   # Max $100 per trade
            stop_loss_pct=0.01,       # 1% stop-loss
            trailing_stop_pct=0.005,  # 0.5% dynamic trailing stop
            max_daily_loss_usd=50.0,  # Stop trading if $50 lost in a day
            min_confidence=0.60,      # Only trade when model ≥60% confident
        )
        self.trader = PaperTrader(
            initial_capital_usd=1000.0,
            risk_manager=risk_mgr,
            reversal_confidence_threshold=0.50, # Withdraw immediately on loss anticipation (reversal)
            on_trade_closed=self.save_trade,
        )

        # Restore previous trade state if any exist
        self._load_historical_trades()

        # Load initial recent data
        logger.info("Preloading recent historical data for feature generation...")
        start_date = (datetime.now(timezone.utc) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        self.recent_data = self.historical.download_historical_data(start_date=start_date)
        if self.recent_data.empty:
            logger.warning("Recent data fetch returned empty!")

    def _load_historical_trades(self):
        db = SessionLocal()
        try:
            records = db.query(TradeRecord).order_by(TradeRecord.closed_at.asc()).all()
            trades = []
            for r in records:
                trades.append({
                    "symbol": r.symbol,
                    "side": r.side,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "quantity_btc": r.quantity_btc,
                    "pnl_usd": r.pnl_usd,
                    "pnl_pct": r.pnl_pct,
                    "reason": r.exit_reason,
                    "opened_at": r.opened_at.isoformat() if r.opened_at else None,
                    "closed_at": r.closed_at.isoformat() if r.closed_at else None,
                    "balance_after": r.balance_after,
                })
            if trades:
                self.trader.restore_from_trades(trades)

            # Restore open position if one was active in DB
            open_rec = db.query(PositionRecord).first()
            if open_rec:
                pos = Position(
                    side=open_rec.side,
                    entry_price=open_rec.entry_price,
                    quantity_btc=open_rec.quantity_btc,
                    usd_value=open_rec.usd_value,
                    stop_loss_price=open_rec.stop_loss_price,
                    opened_at=open_rec.opened_at or datetime.now(timezone.utc),
                )
                pos.highest_price = open_rec.highest_price or open_rec.entry_price
                pos.lowest_price = open_rec.lowest_price or open_rec.entry_price
                self.trader.risk_manager.open_position = pos
                logger.info(f"Restored open {pos.side} position from DB @ ${pos.entry_price:,.2f}")
        except Exception as e:
            logger.error(f"Error loading historical trades from DB: {e}")
        finally:
            db.close()

    def sync_open_position_to_db(self):
        db = SessionLocal()
        try:
            pos = self.trader.risk_manager.open_position
            db.query(PositionRecord).delete()
            if pos:
                rec = PositionRecord(
                    symbol=settings.market.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    quantity_btc=pos.quantity_btc,
                    usd_value=pos.usd_value,
                    stop_loss_price=pos.stop_loss_price,
                    highest_price=pos.highest_price,
                    lowest_price=pos.lowest_price,
                    opened_at=pos.opened_at,
                )
                db.add(rec)
            db.commit()
        except Exception as e:
            logger.error(f"Error syncing open position to DB: {e}")
        finally:
            db.close()

    def save_trade(self, closed_trade: dict):
        db = SessionLocal()
        try:
            # Clear active open position
            db.query(PositionRecord).delete()

            opened_at_val = None
            if closed_trade.get("opened_at"):
                try:
                    opened_at_val = pd.to_datetime(closed_trade["opened_at"], utc=True)
                except Exception:
                    opened_at_val = datetime.now(timezone.utc)

            closed_at_val = None
            if closed_trade.get("closed_at"):
                try:
                    closed_at_val = pd.to_datetime(closed_trade["closed_at"], utc=True)
                except Exception:
                    closed_at_val = datetime.now(timezone.utc)

            record = TradeRecord(
                symbol=settings.market.symbol,
                side=closed_trade.get("side", "BUY"),
                entry_price=closed_trade.get("entry_price", 0.0),
                exit_price=closed_trade.get("exit_price", 0.0),
                quantity_btc=closed_trade.get("quantity_btc", 0.0),
                position_usd=round((closed_trade.get("entry_price", 0.0) or 0.0) * (closed_trade.get("quantity_btc", 0.0) or 0.0), 2),
                pnl_usd=closed_trade.get("pnl_usd", 0.0),
                pnl_pct=closed_trade.get("pnl_pct", 0.0),
                exit_reason=closed_trade.get("reason", "unknown"),
                opened_at=opened_at_val,
                closed_at=closed_at_val,
                balance_after=closed_trade.get("balance_after", self.trader.balance_usd),
            )
            db.add(record)
            db.commit()
            logger.info(f"Trade successfully saved to DB: #{record.id} {record.side} PnL=${record.pnl_usd:+.2f} ({record.exit_reason})")
        except Exception as e:
            logger.error(f"Failed to save trade to DB: {e}")
        finally:
            db.close()

    async def on_candle_closed(self, candle_data: dict):
        logger.info(f"Candle closed at {candle_data['close_time']}. Price: {candle_data['close']}")
        
        # 1. Update our recent data state
        new_row = pd.DataFrame([candle_data])
        new_row["open_time"] = pd.to_datetime(new_row["open_time"], unit="ms", utc=True)
        new_row["close_time"] = pd.to_datetime(new_row["close_time"], unit="ms", utc=True)
        
        self.recent_data = pd.concat([self.recent_data, new_row]).drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        
        # 2. Keep only last 100 rows to save memory
        self.recent_data = self.recent_data.tail(100).reset_index(drop=True)
        
        try:
            # 3. Generate Features
            features_df = self.predictor.generate_features(self.recent_data)

            # 4. Predict
            prediction = self.predictor.predict(features_df)
            signal = prediction["predicted_class"]    # "UP", "DOWN", or "NEUTRAL"
            confidence = prediction["confidence"]

            logger.info(f"Prediction: {signal} (Conf: {confidence:.2f})")

            # 5. Execute paper trade based on signal (handles dynamic entry, predictive withdrawal, trailing stop)
            trade_result = self.trader.execute_signal(
                signal=signal,
                confidence=confidence,
                current_price=candle_data["close"],
            )
            action = trade_result.get("action", "HOLD")
            if action != "HOLD":
                self.sync_open_position_to_db()
                portfolio = self.trader.portfolio_summary()
                logger.info(
                    f"[TRADE] Action={action} | Reason={trade_result.get('reason')} | "
                    f"Balance=${portfolio['balance_usd']:.2f} | "
                    f"P&L=${portfolio['total_pnl_usd']:+.2f} | "
                    f"Win rate={portfolio['win_rate_pct']:.0f}%"
                )

            # 6. Save prediction to database
            self.save_prediction(candle_data, prediction)
            
            # 7. Save alert if a new trade is opened or closed
            if action.startswith("OPEN_"):
                self.save_alert(candle_data, action, candle_data["close"], confidence)
            elif action.startswith("CLOSE_"):
                self.save_alert(candle_data, action, candle_data["close"], confidence, status="CLOSED")

        except Exception as e:
            logger.error(f"Failed to process prediction: {e}")

    def save_prediction(self, candle_data: dict, prediction: dict):
        db = SessionLocal()
        try:
            record = PredictionRecord(
                symbol=settings.market.symbol,
                timestamp=pd.to_datetime(candle_data["open_time"], unit="ms", utc=True),
                horizon=settings.target.horizon,
                price=candle_data["close"],
                probability_down=prediction["probability_down"],
                probability_neutral=prediction["probability_neutral"],
                probability_up=prediction["probability_up"],
                predicted_class=prediction["predicted_class"],
                confidence=prediction["confidence"],
                model_version=prediction["model_version"]
            )
            db.add(record)
            db.commit()
            logger.info("Prediction saved to database.")
        finally:
            db.close()

    def save_alert(self, candle_data: dict, action: str, current_price: float, confidence: float, status: str = "ACTIVE"):
        db = SessionLocal()
        try:
            entry = current_price
            if "BUY" in action:
                tp = entry * 1.025  # 2.5% take profit
                sl = entry * 0.985  # 1.5% stop loss
            else:
                tp = entry * 0.975
                sl = entry * 1.015
                
            record = AlertRecord(
                timestamp=pd.to_datetime(candle_data["open_time"], unit="ms", utc=True),
                action=action,
                entry_price=entry,
                take_profit=tp,
                stop_loss=sl,
                confidence=confidence,
                status=status
            )
            db.add(record)
            db.commit()
            logger.info(f"Alert saved to database: {record.action} @ {record.entry_price}")
        finally:
            db.close()

    async def run(self):
        await self.collector.start(self.on_candle_closed)


if __name__ == "__main__":
    import src.prediction.worker as _self_module
    logging.basicConfig(level=logging.INFO)
    worker = LiveWorker()
    _self_module._worker_instance = worker
    asyncio.run(worker.run())

