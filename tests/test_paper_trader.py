import pytest
from src.trading.paper_trader import PaperTrader
from src.trading.risk_manager import RiskManager


def test_predictive_reversal_exit():
    risk_mgr = RiskManager(
        max_position_usd=100.0,
        stop_loss_pct=0.05,        # 5% stop loss so SL won't trigger immediately
        trailing_stop_pct=0.03,    # 3% trailing stop
        min_confidence=0.60,
    )
    saved_trades = []
    trader = PaperTrader(
        initial_capital_usd=1000.0,
        risk_manager=risk_mgr,
        reversal_confidence_threshold=0.50,
        on_trade_closed=lambda t: saved_trades.append(t),
    )

    # 1. Open a BUY position
    res_open = trader.execute_signal("UP", confidence=0.75, current_price=50000.0)
    assert res_open["action"] == "OPEN_BUY"
    assert trader.risk_manager.open_position is not None
    assert trader.risk_manager.open_position.side == "BUY"

    # 2. Price fluctuates slightly, model stays UP or NEUTRAL -> HOLD
    res_hold = trader.execute_signal("NEUTRAL", confidence=0.40, current_price=50100.0)
    assert res_hold["action"] == "HOLD"
    assert trader.risk_manager.open_position is not None

    # 3. Model detects loss is about to begin: DOWN signal with high confidence -> PREDICTIVE REVERSAL EXIT
    res_exit = trader.execute_signal("DOWN", confidence=0.65, current_price=50200.0)
    assert res_exit["action"] == "CLOSE_PREDICTIVE_REVERSAL"
    assert trader.risk_manager.open_position is None
    assert trader.total_trades == 1
    assert len(saved_trades) == 1
    assert saved_trades[0]["reason"] == "predictive_reversal"
    assert saved_trades[0]["pnl_usd"] > 0  # Sold at 50,200 vs bought at 50,000


def test_trailing_stop_loss_exit():
    risk_mgr = RiskManager(
        max_position_usd=100.0,
        stop_loss_pct=0.02,        # 2% initial SL
        trailing_stop_pct=0.01,    # 1% trailing SL
        min_confidence=0.60,
    )
    trader = PaperTrader(
        initial_capital_usd=1000.0,
        risk_manager=risk_mgr,
    )

    # Open BUY at $50,000. Initial stop = $49,000
    trader.execute_signal("UP", confidence=0.70, current_price=50000.0)
    assert trader.risk_manager.open_position.stop_loss_price == 49000.0

    # Price pumps to $52,000. Trailing stop updates to 52000 * 0.99 = $51,480
    trader.execute_signal("UP", confidence=0.70, current_price=52000.0)
    assert trader.risk_manager.open_position.highest_price == 52000.0
    assert trader.risk_manager.open_position.stop_loss_price == 51480.0

    # Price suddenly drops to $51,400 (reversal hits trailing stop!)
    res_exit = trader.execute_signal("HOLD", confidence=0.30, current_price=51400.0)
    assert res_exit["action"] == "CLOSE_TRAILING_STOP"
    assert trader.risk_manager.open_position is None
    assert trader.total_trades == 1
    assert trader.trade_history[0]["reason"] == "trailing_stop"
    assert trader.trade_history[0]["pnl_usd"] > 0  # Still in profit because trailing stop locked in gains!
