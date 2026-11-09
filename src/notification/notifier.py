"""
Notification Dispatcher
-----------------------
Sends instant real-time alerts to Telegram and Discord on:
- Trade entries (Side, Price, Size, Reason)
- Dynamic trailing stop adjustments
- Trade profit harvests / stop-loss exits
- Emergency kill-switch triggers
"""
import os
import logging
import httpx
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")


async def send_telegram_alert(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.post(url, json=payload)
            return res.is_success
    except Exception as e:
        logger.warning(f"Telegram notification error: {e}")
        return False


async def send_discord_alert(title: str, description: str, color: int = 0x00FF00) -> bool:
    if not DISCORD_WEBHOOK_URL:
        return False
    payload = {
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.post(DISCORD_WEBHOOK_URL, json=payload)
            return res.is_success
    except Exception as e:
        logger.warning(f"Discord notification error: {e}")
        return False


async def notify_trade_event(event_type: str, details: dict):
    """
    Unified dispatcher called on all trade events.
    """
    side = details.get("side", "BUY")
    price = details.get("price", details.get("entry_price", details.get("exit_price", 0.0)))
    pnl = details.get("pnl_usd")

    if event_type.startswith("OPEN"):
        emoji = "🟢" if side == "BUY" else "🔴"
        title = f"{emoji} BTC TRADE OPENED: {side}"
        msg = (
            f"*{title}*\n"
            f"• *Entry Price*: ${price:,.2f}\n"
            f"• *Quantity*: {details.get('quantity_btc', 1.0)} BTC\n"
            f"• *Reason*: {details.get('reason', 'AI+TA Convergence')}\n"
            f"• *Stop-Loss*: ${details.get('stop_loss', 0.0):,.2f}\n"
            f"• *Target*: ${details.get('take_profit', 0.0):,.2f}"
        )
        color = 0x00FF7F if side == "BUY" else 0xFF4500
    elif event_type.startswith("CLOSE"):
        is_profit = (pnl or 0) > 0
        emoji = "💰" if is_profit else "🛑"
        title = f"{emoji} BTC TRADE CLOSED ({details.get('reason', 'exit')})"
        msg = (
            f"*{title}*\n"
            f"• *Side*: {side}\n"
            f"• *Exit Price*: ${price:,.2f}\n"
            f"• *P&L*: ${pnl:+,.2f} ({details.get('pnl_pct', 0):+.2f}%)\n"
            f"• *Balance After*: ${details.get('balance_after', 0):,.2f}"
        )
        color = 0x00FF00 if is_profit else 0xFF0000
    else:
        title = f"ℹ️ BTC BOT: {event_type}"
        msg = f"*{title}*\n{details}"
        color = 0x1E90FF

    logger.info(f"[NOTIFIER] {title}")
    await send_telegram_alert(msg)
    await send_discord_alert(title, msg, color=color)
