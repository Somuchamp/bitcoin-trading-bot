import asyncio
import json
import logging

import httpx
import websockets

logger = logging.getLogger(__name__)

BINANCE_AGG_TRADE_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"

# Shared rate — updated every 30 seconds by a background task
_usdt_usd_rate: float = 1.0  # 1 USDT in USD terms (via USDC proxy)


async def refresh_usdt_rate():
    """
    Fetches USDCUSDT price every 30 seconds.

    USDCUSDT = how many USDT are needed to buy 1 USDC
      e.g. USDCUSDT = 1.0008  →  1 USDC costs 1.0008 USDT
                               →  1 USDT = 1/1.0008 USD ≈ $0.9992

    Therefore: usdt_usd_rate = 1 / USDCUSDT_price
    And:       btc_usd = btc_usdt × usdt_usd_rate
    """
    global _usdt_usd_rate
    async with httpx.AsyncClient() as client:
        while True:
            try:
                res = await client.get(
                    BINANCE_TICKER_URL,
                    params={"symbol": "USDCUSDT"},
                    timeout=5,
                )
                data = res.json()
                usdcusdt = float(data["price"])  # e.g. 1.0008
                _usdt_usd_rate = 1.0 / usdcusdt  # e.g. 0.9992 USD per USDT
                logger.debug(f"USDCUSDT={usdcusdt:.6f}  →  1 USDT = ${_usdt_usd_rate:.6f}")
            except Exception as e:
                logger.warning(f"Failed to fetch USDT rate: {e}")
            await asyncio.sleep(30)


async def binance_price_generator():
    """
    Async generator that yields live BTC price ticks from Binance aggTrade stream.
    Each tick includes both the raw USDT price and the USD-equivalent price.
    Reconnects automatically on disconnect.
    """
    reconnect_delay = 1
    while True:
        try:
            async with websockets.connect(
                BINANCE_AGG_TRADE_URL,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                logger.info("Binance aggTrade stream connected.")
                reconnect_delay = 1
                async for message in ws:
                    data = json.loads(message)
                    price_usdt = float(data["p"])
                    price_usd = price_usdt * _usdt_usd_rate
                    yield {
                        "price_usdt": price_usdt,
                        "price_usd": price_usd,
                        "usdt_usd_rate": _usdt_usd_rate,
                        "quantity": float(data["q"]),
                        "timestamp": data["T"],
                        "is_buyer_maker": data["m"],
                    }
        except Exception as e:
            logger.warning(f"Binance stream error: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

