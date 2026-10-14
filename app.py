import threading
import time
import logging
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
from datetime import datetime, timezone

from src.database.session import engine, Base
from src.prediction.predictor import LivePredictor
from src.trading.paper_trader import PaperTrader
from src.data.collectors.live import BinanceLiveCollector

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is None:
                return lambda f: f
            return func

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HF-BTC-App")

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

# Shared in-memory state for UI responsiveness
state = {
    "latest_price": 68500.0,
    "last_prediction": {
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "prob_up": 0.33,
        "prob_down": 0.33,
        "prob_neutral": 0.34,
        "timestamp": "Waiting for data..."
    },
    "recent_prices": [],
    "logs": []
}

def log_event(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    state["logs"].append(entry)
    if len(state["logs"]) > 50:
        state["logs"].pop(0)
    logger.info(msg)

# Initialize Predictor and Trader
try:
    predictor = LivePredictor()
    log_event("AI Market Predictor initialized with XGBoost.")
except Exception as e:
    predictor = None
    log_event(f"Predictor initialization warning: {e}")

paper_trader = PaperTrader(initial_capital_usd=10000.0)

# -------------------------------------------------------------
# Background Worker Thread (Continuous Live Market Ingestion)
# -------------------------------------------------------------
def background_trading_loop():
    collector = BinanceLiveCollector(symbol="BTCUSDT")
    log_event("Connected to Binance live data stream.")
    
    last_inference_time = 0
    
    while True:
        try:
            # 1. Fetch current price
            ticker = collector.get_ticker_price()
            price = float(ticker.get("price", 0.0))
            if price > 0:
                state["latest_price"] = price
                state["recent_prices"].append({"time": datetime.now(timezone.utc), "price": price})
                if len(state["recent_prices"]) > 200:
                    state["recent_prices"].pop(0)

                # 2. Check protective stops on any open position
                sl_triggered, sl_reason = paper_trader.risk_manager.check_stop_loss(price)
                if sl_triggered:
                    closed = paper_trader._record_closed_trade(sl_reason, price)
                    log_event(f"🚨 Protective exit triggered at ${price:,.2f} ({sl_reason}) | P&L: ${closed.get('pnl_usd', 0):+.2f}")

            # 3. Periodically run AI prediction every 60 seconds automatically
            now = time.time()
            if now - last_inference_time > 60:
                last_inference_time = now
                run_ai_prediction()

            time.sleep(3)
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            time.sleep(5)

# Start background thread
worker_thread = threading.Thread(target=background_trading_loop, daemon=True)
worker_thread.start()

# -------------------------------------------------------------
# Core Prediction Evaluation
# -------------------------------------------------------------
@spaces.GPU
def run_ai_prediction():
    try:
        collector = BinanceLiveCollector(symbol="BTCUSDT")
        candles = collector.get_recent_klines(limit=100)
        if not candles or len(candles) < 55:
            return "Fetching historical candles from Binance..."

        df_candles = pd.DataFrame(candles)
        df_candles["open_time"] = pd.to_datetime(df_candles["open_time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df_candles[col] = df_candles[col].astype(float)
        df_candles = df_candles.sort_values("open_time").reset_index(drop=True)

        current_price = float(df_candles["close"].iloc[-1])
        state["latest_price"] = current_price

        if predictor:
            features = predictor.generate_features(df_candles)
            pred = predictor.predict(features)
            signal = pred["predicted_class"]
            conf = pred["confidence"]

            state["last_prediction"] = {
                "signal": signal,
                "confidence": conf,
                "prob_up": pred["probability_up"],
                "prob_down": pred["probability_down"],
                "prob_neutral": pred["probability_neutral"],
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }

            log_event(f"AI Model Inference: Signal={signal} ({conf*100:.1f}%), Price=${current_price:,.2f}")

            # Execute automated signal via Paper Trader
            trade_res = paper_trader.execute_signal(signal, conf, current_price)
            if trade_res.get("action") not in ("HOLD", None):
                log_event(f"Trade Action: {trade_res['action']} - {trade_res.get('reason', '')}")
            
            return f"Success: {signal} ({conf*100:.1f}% confidence)"
        else:
            return "Model not loaded."
    except Exception as e:
        log_event(f"Prediction error: {e}")
        return f"Error: {e}"

# -------------------------------------------------------------
# Trade Actions (Manual / Triggers)
# -------------------------------------------------------------
def trigger_manual_buy():
    price = state["latest_price"]
    pos = paper_trader.risk_manager.open_position
    if pos:
        return f"Position already open: {pos.side} @ ${pos.entry_price:,.2f}"
    qty = 1000.0 / price
    paper_trader.risk_manager.open_position_for("BUY", price, qty)
    log_event(f"Manual BUY executed: {qty:.4f} BTC at ${price:,.2f}")
    return f"Opened BUY position for {qty:.4f} BTC"

def trigger_manual_sell():
    price = state["latest_price"]
    pos = paper_trader.risk_manager.open_position
    if pos:
        return f"Position already open: {pos.side} @ ${pos.entry_price:,.2f}"
    qty = 1000.0 / price
    paper_trader.risk_manager.open_position_for("SELL", price, qty)
    log_event(f"Manual SELL executed: {qty:.4f} BTC at ${price:,.2f}")
    return f"Opened SELL position for {qty:.4f} BTC"

def trigger_emergency_close():
    price = state["latest_price"]
    res = paper_trader.emergency_close(price)
    if res.get("status") == "closed":
        trade = res["trade"]
        log_event(f"Emergency closed position at ${price:,.2f} | P&L: ${trade.get('pnl_usd', 0):+.2f}")
        return f"Closed position at ${price:,.2f} (P&L: ${trade.get('pnl_usd', 0):+.2f})"
    return "No active position to close."

def trigger_reset_portfolio():
    paper_trader.balance_usd = 10000.0
    paper_trader.trade_history.clear()
    paper_trader.total_trades = 0
    paper_trader.winning_trades = 0
    paper_trader.total_profit_usd = 0.0
    paper_trader.total_loss_usd = 0.0
    paper_trader.risk_manager.open_position = None
    log_event("Portfolio balance reset to $10,000.00 dummy capital.")
    return "Portfolio reset successfully to $10,000.00."

# -------------------------------------------------------------
# UI State Refresh Functions
# -------------------------------------------------------------
def get_metrics():
    price = state["latest_price"]
    summary = paper_trader.portfolio_summary(price)
    pos = summary.get("open_position")
    pred = state["last_prediction"]

    price_str = f"${price:,.2f}"
    balance_str = f"${summary['balance_usd']:,.2f}"
    
    pnl = summary["total_pnl_usd"]
    pnl_str = f"{'+' if pnl >= 0 else ''}${pnl:,.2f} ({summary['total_return_pct']:+.2f}%)"

    sig = pred["signal"]
    sig_badge = f"{'🟢' if sig == 'UP' else ('🔴' if sig == 'DOWN' else '⚪')} {sig} ({pred['confidence']*100:.1f}%)"

    if pos:
        unr = pos.get('unrealized_pnl_usd', 0.0)
        pos_str = f"{pos['side']} {pos['quantity_btc']:.4f} BTC @ ${pos['entry_price']:,.2f} | Float P&L: {'+' if unr >= 0 else ''}${unr:,.2f} (SL: ${pos.get('stop_loss_price', 0):,.2f})"
    else:
        pos_str = "No active position (100% Cash Idle)"

    return price_str, balance_str, pnl_str, sig_badge, pos_str

def get_trades_table():
    trades = paper_trader.trade_history
    if not trades:
        return pd.DataFrame(columns=["#", "Timestamp", "Side", "Entry Price", "Exit Price", "Size (BTC)", "P&L ($)", "Return (%)", "Exit Reason"])
    
    rows = []
    for i, t in enumerate(trades, 1):
        pnl = t.get("pnl_usd", 0.0)
        ret = t.get("return_pct", 0.0)
        rows.append({
            "#": i,
            "Timestamp": t.get("exit_time", "") or t.get("timestamp", ""),
            "Side": t.get("side", ""),
            "Entry Price": f"${t.get('entry_price', 0.0):,.2f}",
            "Exit Price": f"${t.get('exit_price', 0.0):,.2f}",
            "Size (BTC)": f"{t.get('quantity_btc', 0.0):.4f}",
            "P&L ($)": f"{'+' if pnl >= 0 else ''}${pnl:,.2f}",
            "Return (%)": f"{ret:+.2f}%",
            "Exit Reason": t.get("reason", "")
        })
    return pd.DataFrame(rows)

def get_price_chart():
    pts = state["recent_prices"]
    if not pts:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Connecting to Binance real-time stream...")
        return fig

    df_p = pd.DataFrame(pts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_p["time"],
        y=df_p["price"],
        mode="lines",
        name="BTC/USDT",
        line=dict(color="#3b82f6", width=2)
    ))
    
    # Add active position line if open
    pos = paper_trader.risk_manager.open_position
    if pos:
        fig.add_hline(
            y=pos.entry_price,
            line_dash="dash",
            line_color="#10b981" if pos.side == "BUY" else "#ef4444",
            annotation_text=f"Entry ({pos.side}): ${pos.entry_price:,.2f}"
        )
        if pos.stop_loss_price:
            fig.add_hline(
                y=pos.stop_loss_price,
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text=f"Stop Level: ${pos.stop_loss_price:,.2f}"
            )

    fig.update_layout(
        template="plotly_dark",
        title="Live BTC/USDT Price & Dynamic Risk Protection Levels",
        xaxis_title="Time (UTC)",
        yaxis_title="Price (USD)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380
    )
    return fig

def get_logs_text():
    return "\n".join(reversed(state["logs"]))

# -------------------------------------------------------------
# Gradio Dashboard Layout
# -------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, title="Bitcoin AI Trading Platform") as demo:
    gr.Markdown("""
    # 🤖 Autonomous Bitcoin AI Trading Platform
    ### *Real-Time Market Ingestion · XGBoost Prediction · 3-Layer Predictive Risk Protection · 24/7 Paper Trading*
    """)

    with gr.Row():
        live_price_box = gr.Textbox(label="Live BTC/USDT Price", interactive=False)
        balance_box = gr.Textbox(label="Total Balance (Paper)", interactive=False)
        pnl_box = gr.Textbox(label="Realized P&L", interactive=False)
        signal_box = gr.Textbox(label="AI Model Signal", interactive=False)

    with gr.Row():
        position_box = gr.Textbox(label="Active Position Status", interactive=False, scale=3)
        refresh_btn = gr.Button("⚡ Force AI Prediction", variant="primary", scale=1)

    with gr.Row():
        chart_view = gr.Plot(label="Live Market Chart", value=get_price_chart)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Manual Trade Controls & Simulator")
            with gr.Row():
                buy_btn = gr.Button("🟢 Simulate BUY ($1k)", variant="secondary")
                sell_btn = gr.Button("🔴 Simulate SELL ($1k)", variant="secondary")
            with gr.Row():
                close_btn = gr.Button("🚨 Emergency Close", variant="stop")
                reset_btn = gr.Button("🔄 Reset Account ($10k)", variant="secondary")
            action_status = gr.Textbox(label="Action Feedback", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📜 Real-Time AI System Logs")
            log_box = gr.Textbox(label="Telemetry Log Stream", value=get_logs_text, lines=8, interactive=False)

    with gr.Row():
        gr.Markdown("### 📊 Trade Execution History")
    with gr.Row():
        trades_table = gr.Dataframe(value=get_trades_table, interactive=False)

    # Auto refresh timer (every 3 seconds)
    timer = gr.Timer(3)
    timer.tick(
        fn=lambda: (
            *get_metrics(),
            get_price_chart(),
            get_trades_table(),
            get_logs_text()
        ),
        outputs=[
            live_price_box,
            balance_box,
            pnl_box,
            signal_box,
            position_box,
            chart_view,
            trades_table,
            log_box
        ]
    )

    # Button Event Handlers
    refresh_btn.click(fn=run_ai_prediction, outputs=[action_status])
    buy_btn.click(fn=trigger_manual_buy, outputs=[action_status])
    sell_btn.click(fn=trigger_manual_sell, outputs=[action_status])
    close_btn.click(fn=trigger_emergency_close, outputs=[action_status])
    reset_btn.click(fn=trigger_reset_portfolio, outputs=[action_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
