"""
Interactive & Automated Testing Tool for Autonomous Trading Bot
---------------------------------------------------------------
Use this script to test trade execution, live dashboard telemetry,
trailing stops, and predictive loss-anticipation exits without waiting
hours for real candles.

Usage:
  python test_trading.py                    # Run full interactive test suite
  python test_trading.py --simulate         # Run automated 10-second end-to-end trade cycle
  python test_trading.py --open BUY         # Open a live position right now at market price
  python test_trading.py --close            # Close current open position at market price
  python test_trading.py --status           # Check current portfolio & position status
  python test_trading.py --reset            # Reset balance back to $1,000 & clear test history
"""

import sys
import time
import argparse
import logging
from datetime import datetime, timezone
import requests

from src.database.session import SessionLocal, engine, Base
from src.database.models import TradeRecord, PositionRecord, AlertRecord, PredictionRecord
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestTrading")

# Ensure all DB tables are created
Base.metadata.create_all(bind=engine)


def get_current_btc_price() -> float:
    """Fetches real-time BTC price from Binance or local API."""
    try:
        res = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
        if res.status_code == 200:
            return float(res.json()["price"])
    except Exception:
        pass
    # Fallback to local prediction
    db = SessionLocal()
    try:
        last = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).first()
        if last:
            return float(last.price)
    finally:
        db.close()
    return 83500.0


def get_portfolio_state():
    db = SessionLocal()
    try:
        trades = db.query(TradeRecord).order_by(TradeRecord.closed_at.asc()).all()
        open_pos = db.query(PositionRecord).first()
        initial_cap = 1000.0
        cur_balance = trades[-1].balance_after if trades and trades[-1].balance_after else initial_cap
        wins = sum(1 for t in trades if (t.pnl_usd or 0) > 0)
        return {
            "balance": cur_balance,
            "total_trades": len(trades),
            "wins": wins,
            "win_rate": (wins / len(trades) * 100) if trades else 0.0,
            "open_position": open_pos,
        }
    finally:
        db.close()


def open_test_position(side: str = "BUY", price: float = None, size_usd: float = 100.0):
    if price is None:
        price = get_current_btc_price()

    db = SessionLocal()
    try:
        existing = db.query(PositionRecord).first()
        if existing:
            print(f"[!] A position is ALREADY OPEN: {existing.side} @ ${existing.entry_price:,.2f}")
            return existing

        quantity_btc = round(size_usd / price, 6)
        stop_loss = round(price * 0.99 if side == "BUY" else price * 1.01, 2)
        pos = PositionRecord(
            symbol=settings.market.symbol,
            side=side,
            entry_price=price,
            quantity_btc=quantity_btc,
            usd_value=size_usd,
            stop_loss_price=stop_loss,
            highest_price=price,
            lowest_price=price,
            opened_at=datetime.now(timezone.utc),
        )
        db.add(pos)
        
        # Also log alert
        alert = AlertRecord(
            timestamp=datetime.now(timezone.utc),
            action=f"OPEN_{side}",
            entry_price=price,
            take_profit=price * 1.02,
            stop_loss=stop_loss,
            confidence=0.72,
            status="ACTIVE",
        )
        db.add(alert)
        db.commit()

        print(f"\n=======================================================")
        print(f" [SUCCESS] Position OPENED in Database:")
        print(f"   Side:         {side}")
        print(f"   Entry Price:  ${price:,.2f}")
        print(f"   Position Size:${size_usd:.2f} ({quantity_btc} BTC)")
        print(f"   Stop-Loss:    ${stop_loss:,.2f}")
        print(f"=======================================================")
        print(f"-> Check your Dashboard (http://localhost:3000) now!")
        print(f"   You should see: ACTIVE POSITION: {side} @ ${price:,.2f}\n")
        return pos
    finally:
        db.close()


def close_test_position(exit_price: float = None, reason: str = "predictive_reversal"):
    if exit_price is None:
        exit_price = get_current_btc_price()

    db = SessionLocal()
    try:
        pos = db.query(PositionRecord).first()
        if not pos:
            print("[!] No active open position found to close.")
            return None

        if pos.side == "BUY":
            pnl_usd = (exit_price - pos.entry_price) * pos.quantity_btc
        else:
            pnl_usd = (pos.entry_price - exit_price) * pos.quantity_btc
        pnl_pct = (pnl_usd / pos.usd_value) * 100

        trades = db.query(TradeRecord).order_by(TradeRecord.closed_at.desc()).first()
        prev_balance = trades.balance_after if trades and trades.balance_after else 1000.0
        new_balance = round(prev_balance + pnl_usd, 2)

        trade = TradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity_btc=pos.quantity_btc,
            position_usd=pos.usd_value,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=reason,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc),
            balance_after=new_balance,
        )
        db.add(trade)
        db.delete(pos)

        # Log alert
        alert = AlertRecord(
            timestamp=datetime.now(timezone.utc),
            action=f"CLOSE_{reason.upper()}",
            entry_price=exit_price,
            take_profit=exit_price,
            stop_loss=exit_price,
            confidence=0.65,
            status="CLOSED",
        )
        db.add(alert)
        db.commit()

        print(f"\n=======================================================")
        print(f" [SUCCESS] Position CLOSED & Logged to Trade History:")
        print(f"   Side:         {pos.side}")
        print(f"   Entry:        ${pos.entry_price:,.2f}")
        print(f"   Exit Price:   ${exit_price:,.2f}")
        print(f"   P&L USD:      {'+' if pnl_usd >= 0 else ''}${pnl_usd:.2f}")
        print(f"   P&L %:        {'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%")
        print(f"   Exit Reason:  {reason.upper()}")
        print(f"   New Balance:  ${new_balance:.2f}")
        print(f"=======================================================")
        print(f"-> Check your Dashboard (http://localhost:3000) now!")
        print(f"   The trade will now appear in 'Paper Trade Execution History'!\n")
        return trade
    finally:
        db.close()


def simulate_full_cycle():
    """Runs a complete end-to-end demonstration in 5 seconds."""
    print("\n--- STARTING AUTOMATED TEST SIMULATION ---")
    live_price = get_current_btc_price()
    print(f"1. Fetching Live Bitcoin Market Price: ${live_price:,.2f}")

    print("2. Simulating AI Model Signal: UP (Bullish) with 75% Confidence...")
    open_test_position("BUY", price=live_price, size_usd=100.0)

    print("Waiting 4 seconds... (Check your dashboard: Position should be ACTIVE)")
    time.sleep(4)

    # Simulate price moving up by +$350 then model anticipating loss
    exit_price = round(live_price + 350.0, 2)
    print(f"3. Simulating Market Pump to ${exit_price:,.2f} followed by Model PREDICTIVE LOSS REVERSAL...")
    close_test_position(exit_price=exit_price, reason="predictive_reversal")

    print("[DONE] Full trade lifecycle completed and recorded in database!")


def reset_database():
    confirm = input("Are you sure you want to clear test trades and reset balance to $1,000? (y/n): ")
    if confirm.lower() == "y":
        db = SessionLocal()
        try:
            db.query(TradeRecord).delete()
            db.query(PositionRecord).delete()
            db.query(AlertRecord).delete()
            db.commit()
            print("[SUCCESS] All test trades cleared. Balance reset to $1,000.00.")
        finally:
            db.close()


def show_status():
    p = get_portfolio_state()
    print("\n================ PORTFOLIO STATUS ================")
    print(f" Current Balance:   ${p['balance']:,.2f}")
    print(f" Total Trades:      {p['total_trades']}")
    print(f" Winning Trades:    {p['wins']}")
    print(f" Win Rate:          {p['win_rate']:.1f}%")
    if p['open_position']:
        pos = p['open_position']
        print(f" Open Position:     {pos.side} @ ${pos.entry_price:,.2f} (${pos.usd_value:.2f})")
        print(f" Stop-Loss:         ${pos.stop_loss_price:,.2f}")
    else:
        print(f" Open Position:     None (Holding Cash)")
    print("===================================================\n")


def interactive_menu():
    while True:
        print("\n================ BITCOIN TRADING TEST MENU ================")
        print("  1. Run Full Simulated Trade (Auto-Open & Auto-Exit in 5s)")
        print("  2. Open Simulated BUY Position at Current Live Price")
        print("  3. Open Simulated SELL Position at Current Live Price")
        print("  4. Close Active Position (Simulate Predictive Exit)")
        print("  5. Close Active Position (Simulate Trailing Stop Exit)")
        print("  6. View Portfolio & Position Status")
        print("  7. Reset Portfolio to $1,000 (Clear History)")
        print("  0. Exit")
        print("===========================================================")
        choice = input("Enter choice (0-7): ").strip()

        if choice == "1":
            simulate_full_cycle()
        elif choice == "2":
            open_test_position("BUY")
        elif choice == "3":
            open_test_position("SELL")
        elif choice == "4":
            close_test_position(reason="predictive_reversal")
        elif choice == "5":
            close_test_position(reason="trailing_stop")
        elif choice == "6":
            show_status()
        elif choice == "7":
            reset_database()
        elif choice == "0":
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot Test Utility")
    parser.add_argument("--simulate", action="store_true", help="Run automated trade lifecycle test")
    parser.add_argument("--open", choices=["BUY", "SELL"], help="Open a position")
    parser.add_argument("--close", action="store_true", help="Close active position")
    parser.add_argument("--reason", default="predictive_reversal", help="Exit reason")
    parser.add_argument("--status", action="store_true", help="View current portfolio status")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio back to $1,000")

    args = parser.parse_args()

    if args.simulate:
        simulate_full_cycle()
    elif args.open:
        open_test_position(args.open)
    elif args.close:
        close_test_position(reason=args.reason)
    elif args.status:
        show_status()
    elif args.reset:
        reset_database()
    else:
        interactive_menu()
