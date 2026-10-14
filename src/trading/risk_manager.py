"""
Risk Manager
------------
Enforces hard limits on every trade:
- Max position size in USD
- Stop-loss %: auto-close if price moves against us
- Max daily loss: disables trading for the rest of the day if hit
- Only one open position at a time
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    side: str           # "BUY" or "SELL"
    entry_price: float
    quantity_btc: float
    usd_value: float
    stop_loss_price: float
    highest_price: float = field(init=False)
    lowest_price: float = field(init=False)
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "BUY":
            return (current_price - self.entry_price) * self.quantity_btc
        else:  # SELL / short simulation
            return (self.entry_price - current_price) * self.quantity_btc

    def is_stop_loss_hit(self, current_price: float) -> bool:
        if self.side == "BUY":
            return current_price <= self.stop_loss_price
        else:
            return current_price >= self.stop_loss_price

    def pnl_pct(self, current_price: float) -> float:
        return self.unrealized_pnl(current_price) / self.usd_value


class RiskManager:
    def __init__(
        self,
        max_position_usd: float = 100.0,
        stop_loss_pct: float = 0.01,       # 1% initial stop-loss
        trailing_stop_pct: float = 0.005,  # 0.5% trailing stop
        max_daily_loss_usd: float = 50.0,  # Stop trading if we lose $50 in a day
        min_confidence: float = 0.60,      # Only trade when model is ≥60% confident
    ):
        self.max_position_usd = max_position_usd
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_daily_loss_usd = max_daily_loss_usd
        self.min_confidence = min_confidence

        self.open_position: Optional[Position] = None
        self._daily_loss: float = 0.0
        self._last_reset_date: date = date.today()

    def _reset_daily_loss_if_new_day(self):
        today = date.today()
        if today != self._last_reset_date:
            logger.info(f"New trading day. Resetting daily loss (was ${self._daily_loss:.2f})")
            self._daily_loss = 0.0
            self._last_reset_date = today

    @property
    def daily_loss_limit_hit(self) -> bool:
        self._reset_daily_loss_if_new_day()
        return self._daily_loss >= self.max_daily_loss_usd

    def can_open_trade(self, confidence: float, signal: str) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        self._reset_daily_loss_if_new_day()

        if self.open_position is not None:
            return False, f"Position already open ({self.open_position.side})"
        if signal == "NEUTRAL":
            return False, "Signal is NEUTRAL — no trade"
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence:.0%} below minimum {self.min_confidence:.0%}"
        if self.daily_loss_limit_hit:
            return False, f"Daily loss limit hit (${self._daily_loss:.2f} lost today)"
        return True, "OK"

    def open_position_for(self, side: str, entry_price: float, quantity_btc: float) -> Position:
        usd_value = quantity_btc * entry_price
        if side == "BUY":
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        else:
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)

        pos = Position(
            side=side,
            entry_price=entry_price,
            quantity_btc=quantity_btc,
            usd_value=usd_value,
            stop_loss_price=stop_loss_price,
        )
        self.open_position = pos
        logger.info(
            f"Position OPENED: {side} {quantity_btc:.6f} BTC @ ${entry_price:,.2f} "
            f"| Stop-loss: ${stop_loss_price:,.2f} | Value: ${usd_value:.2f}"
        )
        return pos

    def close_position(self, exit_price: float, reason: str = "signal") -> dict:
        if self.open_position is None:
            return {}
        pos = self.open_position
        pnl = pos.unrealized_pnl(exit_price)
        if pnl < 0:
            self._daily_loss += abs(pnl)
        result = {
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity_btc": pos.quantity_btc,
            "pnl_usd": round(pnl, 4),
            "pnl_pct": round(pos.pnl_pct(exit_price) * 100, 3),
            "reason": reason,
            "opened_at": pos.opened_at.isoformat(),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"Position CLOSED: {pos.side} @ ${exit_price:,.2f} | "
            f"P&L: ${pnl:+.2f} ({result['pnl_pct']:+.2f}%) | Reason: {reason}"
        )
        self.open_position = None
        return result

    def check_stop_loss(self, current_price: float) -> tuple[bool, str]:
        """Updates trailing stops and returns (is_triggered, reason)."""
        if not self.open_position:
            return False, ""
            
        pos = self.open_position
        initial_stop = (pos.entry_price * (1 - self.stop_loss_pct)) if pos.side == "BUY" else (pos.entry_price * (1 + self.stop_loss_pct))
        
        # Update trailing stop dynamically
        if pos.side == "BUY":
            if current_price > pos.highest_price:
                pos.highest_price = current_price
                new_stop = current_price * (1 - self.trailing_stop_pct)
                if new_stop > pos.stop_loss_price:
                    pos.stop_loss_price = new_stop
        else:
            if current_price < pos.lowest_price:
                pos.lowest_price = current_price
                new_stop = current_price * (1 + self.trailing_stop_pct)
                if new_stop < pos.stop_loss_price:
                    pos.stop_loss_price = new_stop

        # Check if the dynamic stop was hit
        if pos.is_stop_loss_hit(current_price):
            reason = "trailing_stop" if abs(pos.stop_loss_price - initial_stop) > 1e-4 else "stop_loss"
            logger.warning(
                f"{reason.upper()} triggered at ${current_price:,.2f} "
                f"(entry: ${pos.entry_price:,.2f}, peak: ${pos.highest_price if pos.side == 'BUY' else pos.lowest_price:,.2f})"
            )
            return True, reason
        return False, ""

    def status(self) -> dict:
        self._reset_daily_loss_if_new_day()
        pos = self.open_position
        return {
            "has_open_position": pos is not None,
            "open_position": {
                "side": pos.side,
                "entry_price": pos.entry_price,
                "quantity_btc": pos.quantity_btc,
                "stop_loss_price": pos.stop_loss_price,
                "usd_value": pos.usd_value,
                "opened_at": pos.opened_at.isoformat(),
            } if pos else None,
            "daily_loss_usd": round(self._daily_loss, 4),
            "daily_loss_limit_hit": self.daily_loss_limit_hit,
            "max_position_usd": self.max_position_usd,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "min_confidence": self.min_confidence,
        }
