import asyncio
import json
import logging
from typing import Callable, Coroutine

import websockets

from src.config.settings import settings

logger = logging.getLogger(__name__)


class BinanceLiveCollector:
    """
    Connects to Binance WebSocket for live kline (candlestick) updates.
    """

    def __init__(
        self,
        symbol: str = settings.market.symbol,
        interval: str = settings.market.timeframe,
    ):
        self.symbol = symbol.lower()
        self.interval = interval
        self.stream_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"
        self.keep_running = True
        
        # Prevent duplicate events from same candle
        self.last_candle_close_time = None

    async def start(self, on_candle_closed: Callable[[dict], Coroutine]):
        """
        Connects to the websocket and calls the callback when a candle closes.
        """
        logger.info(f"Connecting to Binance WS: {self.stream_url}")
        
        reconnect_delay = 1
        
        while self.keep_running:
            try:
                async with websockets.connect(self.stream_url, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("Connected to Binance WebSocket.")
                    reconnect_delay = 1
                    
                    async for message in ws:
                        if not self.keep_running:
                            break
                            
                        data = json.loads(message)
                        kline = data.get("k")
                        if not kline:
                            continue
                        
                        is_closed = kline.get("x", False)
                        close_time = kline.get("T")
                        
                        if is_closed and close_time != self.last_candle_close_time:
                            self.last_candle_close_time = close_time
                            
                            candle_data = {
                                "open_time": kline["t"],
                                "open": float(kline["o"]),
                                "high": float(kline["h"]),
                                "low": float(kline["l"]),
                                "close": float(kline["c"]),
                                "volume": float(kline["v"]),
                                "close_time": close_time,
                                "number_of_trades": int(kline["n"])
                            }
                            
                            # Invoke callback
                            try:
                                await on_candle_closed(candle_data)
                            except Exception as e:
                                logger.error(f"Error in on_candle_closed callback: {e}")

            except websockets.exceptions.ConnectionClosedError as e:
                logger.warning(f"WebSocket disconnected: {e}. Reconnecting in {reconnect_delay}s...")
            except Exception as e:
                logger.error(f"Unexpected WebSocket error: {e}. Reconnecting in {reconnect_delay}s...")
            
            if not self.keep_running:
                break
                
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
            
    def stop(self):
        logger.info("Stopping Binance Live Collector...")
        self.keep_running = False
