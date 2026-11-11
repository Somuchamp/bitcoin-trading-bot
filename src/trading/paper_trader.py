"""
Paper Trader
------------
Simulates buy/sell orders with real live prices but ZERO real money.
Tracks simulated P&L, trade history, and portfolio balance.

Set PAPER_TRADE=true in .env (default) to use this instead of real orders.
When ready for live trading, set PAPER_TRADE=false and configure Binance API keys.
"""
import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

from src.trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Simulates trading with real market prices but no real orders.
    Tracks every trade decision in memory and notifies listeners (DB).
    """

    def __init__(
        self,
        initial_capital_usd: float = 1000.0,
        risk_manager: Optional[RiskManager] = None,
        reversal_confidence_threshold: float = 0.50,
        on_trade_closed: Optional[Callable[[dict], None]] = None,
        fixed_btc_size: Optional[float] = None,
        take_profit_pct: float = 0.010,
    ):
        self.balance_usd = initial_capital_usd
        self.initial_capital = initial_capital_usd
        self.peak_balance = initial_capital_usd
        self.risk_manager = risk_manager or RiskManager()
        self.reversal_confidence_threshold = reversal_confidence_threshold
        self.on_trade_closed = on_trade_closed
        self.fixed_btc_size = fixed_btc_size
        self.take_profit_pct = take_profit_pct
        self.trade_history: List[dict] = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit_usd = 0.0
        self.total_loss_usd = 0.0

        logger.info(
            f"PaperTrader initialized. Capital: ${initial_capital_usd:.2f} | "
            f"Max per trade: ${self.risk_manager.max_position_usd:.2f} | "
            f"Stop-loss: {self.risk_manager.stop_loss_pct:.1%} | "
            f"Trailing stop: {self.risk_manager.trailing_stop_pct:.1%} | "
            f"Reversal exit threshold: {self.reversal_confidence_threshold:.0%}"
        )

    def restore_from_trades(self, trades: List[dict]):
        """Restores balance and trade statistics from previously saved DB records."""
        if not trades:
            return
        self.trade_history = list(trades)
        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if (t.get("pnl_usd") or 0) > 0)
        self.total_profit_usd = sum(t.get("pnl_usd", 0) for t in trades if (t.get("pnl_usd") or 0) > 0)
        self.total_loss_usd = sum(abs(t.get("pnl_usd", 0)) for t in trades if (t.get("pnl_usd") or 0) < 0)
        
        # Latest balance
        if trades[-1].get("balance_after"):
            self.balance_usd = trades[-1]["balance_after"]
            self.peak_balance = max([self.initial_capital] + [t.get("balance_after", self.initial_capital) for t in trades])
        logger.info(f"PaperTrader restored {self.total_trades} trades from history. Current balance: ${self.balance_usd:.2f}")

    def execute_signal(self, signal: str, confidence: float, current_price: float) -> dict:
        """
        Called after each model prediction.
        signal: 'UP', 'DOWN', or 'NEUTRAL'
        confidence: float 0-1 (e.g. 0.65)
        current_price: live BTC price in USD
        Returns a dict describing the action taken.
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "confidence": confidence,
            "price": current_price,
            "action": "HOLD",
            "reason": "",
            "pnl_usd": None,
        }

        # 1. Check stop-loss / trailing stop on any open position first
        sl_triggered, sl_reason = self.risk_manager.check_stop_loss(current_price)
        if sl_triggered:
            closed = self._record_closed_trade(sl_reason, current_price)
            result["action"] = f"CLOSE_{sl_reason.upper()}"
            result["pnl_usd"] = closed.get("pnl_usd")
            result["reason"] = f"Protective stop hit: {sl_reason}"
            return result

        # 2. Predictive loss anticipation & withdrawal
        pos = self.risk_manager.open_position
        if pos is not None:
            # If LONG and model predicts DOWN with high confidence: exit before further drop
            if pos.side == "BUY" and signal == "DOWN" and confidence >= self.reversal_confidence_threshold:
                closed = self._record_closed_trade("predictive_reversal", current_price)
                result["action"] = "CLOSE_PREDICTIVE_REVERSAL"
                result["pnl_usd"] = closed.get("pnl_usd")
                result["reason"] = f"Loss anticipated: Model flipped to DOWN ({confidence:.0%} confidence)"
                return result

            # If SHORT and model predicts UP: exit before further rise
            elif pos.side == "SELL" and signal == "UP" and confidence >= self.reversal_confidence_threshold:
                closed = self._record_closed_trade("predictive_reversal", current_price)
                result["action"] = "CLOSE_PREDICTIVE_REVERSAL"
                result["pnl_usd"] = closed.get("pnl_usd")
                result["reason"] = f"Loss anticipated: Model flipped to UP ({confidence:.0%} confidence)"
                return result

            # Still holding current position
            result["action"] = "HOLD"
            result["reason"] = f"Holding {pos.side} position (Entry: ${pos.entry_price:,.2f}, SL: ${pos.stop_loss_price:,.2f})"
            return result

        # 3. No position open: check if model signals a new trade
        can_trade, reason = self.risk_manager.can_open_trade(confidence, signal)
        if not can_trade:
            result["reason"] = result["reason"] or reason
            return result

        # Position sizing (exact fixed BTC size when requested, capped by risk limits & balance)
        target_usd = (
            current_price * self.fixed_btc_size
            if self.fixed_btc_size and self.fixed_btc_size > 0
            else self.risk_manager.max_position_usd
        )
        position_usd = min(target_usd, self.risk_manager.max_position_usd, self.balance_usd * 0.95)
        if position_usd < 10:
            result["reason"] = f"Insufficient balance (${self.balance_usd:.2f})"
            return result

        quantity_btc = position_usd / current_price
        side = "BUY" if signal == "UP" else "SELL"

        self.risk_manager.open_position_for(side, current_price, quantity_btc)

        result["action"] = f"OPEN_{side}"
        result["reason"] = f"Model: {signal} @ {confidence:.0%} confidence"
        result["quantity_btc"] = quantity_btc
        result["position_usd"] = position_usd

        logger.info(
            f"[PAPER] {side} {quantity_btc:.6f} BTC @ ${current_price:,.2f} "
            f"| Signal: {signal} ({confidence:.0%}) | Size: ${position_usd:.2f}"
        )
        return result

    def on_price_tick(self, current_price: float) -> Optional[dict]:
        """Evaluates protective exits (take profit / trailing stop) in real time on every price tick."""
        if not self.risk_manager.open_position:
            return None

        pos = self.risk_manager.open_position
        is_buy = pos.side == "BUY"
        # Dynamic take-profit target (default 1.0%)
        take_profit_price = (
            pos.entry_price * (1 + self.take_profit_pct)
            if is_buy
            else pos.entry_price * (1 - self.take_profit_pct)
        )
        take_profit_hit = (current_price >= take_profit_price) if is_buy else (current_price <= take_profit_price)
        if take_profit_hit:
            closed = self._record_closed_trade("take_profit", current_price)
            logger.info(f"TAKE PROFIT harvested at ${current_price:,.2f} | PnL: ${closed.get('pnl_usd', 0):+.2f}")
            return {"action": "CLOSE_TAKE_PROFIT", "trade": closed}

        # Check dynamic stop-loss / trailing stop
        sl_triggered, sl_reason = self.risk_manager.check_stop_loss(current_price)
        if sl_triggered:
            closed = self._record_closed_trade(sl_reason, current_price)
            logger.info(f"{sl_reason.upper()} hit at ${current_price:,.2f} | PnL: ${closed.get('pnl_usd', 0):+.2f}")
            return {"action": f"CLOSE_{sl_reason.upper()}", "trade": closed}

        return None

    def emergency_close(self, current_price: float) -> dict:
        """Forces closure of any active position immediately."""
        if self.risk_manager.open_position:
            closed = self._record_closed_trade("emergency_stop", current_price)
            logger.warning(f"EMERGENCY CLOSE executed at ${current_price:,.2f}")
            return {"status": "closed", "trade": closed}
        return {"status": "no_open_position"}

    def _record_trade(self, closed: dict):
        if not closed:
            return
        self.total_trades += 1
        pnl = closed.get("pnl_usd", 0) or 0
        self.balance_usd += pnl
        if self.balance_usd > self.peak_balance:
            self.peak_balance = self.balance_usd

        if pnl > 0:
            self.winning_trades += 1
            self.total_profit_usd += pnl
        else:
            self.total_loss_usd += abs(pnl)

        closed["balance_after"] = round(self.balance_usd, 2)
        self.trade_history.append(closed)
        logger.info(
            f"Trade #{self.total_trades} recorded. Reason: {closed.get('reason')} | "
            f"P&L: ${pnl:+.2f} | Balance: ${self.balance_usd:.2f} | "
            f"Win rate: {self.win_rate:.0%}"
        )

        if self.on_trade_closed:
            try:
                self.on_trade_closed(closed)
            except Exception as e:
                logger.error(f"Failed in on_trade_closed callback: {e}")

    def _record_closed_trade(self, reason: str, exit_price: float) -> dict:
        closed = self.risk_manager.close_position(exit_price, reason=reason)
        self._record_trade(closed)
        return closed

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.total_loss_usd == 0:
            return float("inf") if self.total_profit_usd > 0 else 0.0
        return self.total_profit_usd / self.total_loss_usd

    @property
    def max_drawdown_pct(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        dd = (self.peak_balance - self.balance_usd) / self.peak_balance * 100
        return max(0.0, dd)

    @property
    def total_pnl(self) -> float:
        return self.balance_usd - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        return (self.balance_usd / self.initial_capital - 1) * 100

    def portfolio_summary(self, current_price: Optional[float] = None) -> dict:
        pos = self.risk_manager.open_position
        open_pos_dict = None
        if pos:
            unrealized = pos.unrealized_pnl(current_price) if current_price else 0.0
            unrealized_pct = pos.pnl_pct(current_price) * 100 if current_price else 0.0
            tp_price = (
                pos.entry_price * (1 + self.take_profit_pct)
                if pos.side == "BUY"
                else pos.entry_price * (1 - self.take_profit_pct)
            )
            open_pos_dict = {
                "side": pos.side,
                "entry_price": pos.entry_price,
                "quantity_btc": pos.quantity_btc,
                "usd_value": round(pos.usd_value, 2),
                "stop_loss_price": round(pos.stop_loss_price, 2),
                "take_profit_price": round(tp_price, 2),
                "highest_price": round(pos.highest_price, 2),
                "lowest_price": round(pos.lowest_price, 2),
                "opened_at": pos.opened_at.isoformat(),
                "unrealized_pnl_usd": round(unrealized, 2),
                "unrealized_pnl_pct": round(unrealized_pct, 2),
            }

        recent_normalized = []
        for t in reversed(self.trade_history[-50:]):
            item = dict(t)
            if "exit_reason" not in item and "reason" in item:
                item["exit_reason"] = item["reason"]
            if "position_usd" not in item:
                item["position_usd"] = round((item.get("entry_price") or 0.0) * (item.get("quantity_btc") or 0.0), 2)
            recent_normalized.append(item)

        return {
            "mode": "PAPER_TRADING",
            "initial_capital_usd": self.initial_capital,
            "balance_usd": round(self.balance_usd, 2),
            "peak_balance_usd": round(self.peak_balance, 2),
            "total_pnl_usd": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate_pct": round(self.win_rate * 100, 1),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor != float("inf") else 999.0,
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "open_position": open_pos_dict,
            "recent_trades": recent_normalized,
        }

