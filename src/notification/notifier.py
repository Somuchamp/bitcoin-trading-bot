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
import requests
import httpx
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_telegram_creds():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return token, chat_id


def get_discord_webhook():
    return os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def send_telegram_alert_sync(message: str) -> dict:
    """
    Synchronous Telegram delivery using requests.
    Thread-safe and loop-independent.
    """
    token, chat_id = get_telegram_creds()
    if not token or not chat_id:
        msg = f"Telegram skipped: token_set={bool(token)}, chat_id_set={bool(chat_id)}"
        logger.warning(msg)
        return {"success": False, "error": msg, "token_set": bool(token), "chat_id_set": bool(chat_id)}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        res = requests.post(url, json=payload, timeout=8)
        data = res.json()
        if res.ok and data.get("ok"):
            logger.info("Telegram alert delivered successfully.")
            return {"success": True, "data": data}
        
        # If Markdown entity parsing failed, retry with plain text
        if "can't parse entities" in str(data.get("description", "")):
            payload.pop("parse_mode", None)
            res2 = requests.post(url, json=payload, timeout=8)
            data2 = res2.json()
            if res2.ok and data2.get("ok"):
                logger.info("Telegram alert delivered via plain text fallback.")
                return {"success": True, "data": data2}

        logger.error(f"Telegram API error {res.status_code}: {res.text}")
        return {"success": False, "status_code": res.status_code, "error": data.get("description", res.text)}
    except Exception as e:
        logger.error(f"Telegram notification exception: {e}")
        return {"success": False, "error": str(e)}


async def send_telegram_alert(message: str) -> bool:
    """Async wrapper for send_telegram_alert_sync."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, send_telegram_alert_sync, message)
        return res.get("success", False)
    except RuntimeError:
        res = send_telegram_alert_sync(message)
        return res.get("success", False)


async def send_discord_alert(title: str, description: str, color: int = 0x00FF00) -> bool:
    webhook = get_discord_webhook()
    if not webhook:
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
        async with httpx.AsyncClient(timeout=8) as client:
            res = await client.post(webhook, json=payload)
            return res.is_success
    except Exception as e:
        logger.warning(f"Discord notification error: {e}")
        return False


def notify_trade_event_sync(event_type: str, details: dict):
    """
    Synchronous notification dispatcher safe to call from any thread or callback.
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
            f"• *Quantity*: {details.get('quantity_btc', 1.0):.4f} BTC\n"
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
    send_telegram_alert_sync(msg)


async def notify_trade_event(event_type: str, details: dict):
    """Async wrapper for notify_trade_event_sync."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, notify_trade_event_sync, event_type, details)
    except RuntimeError:
        notify_trade_event_sync(event_type, details)
