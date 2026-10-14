import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.config.settings import settings

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


class BinanceHistoricalCollector:
    """
    Downloads historical klines (candlesticks) from Binance.
    """

    COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    def __init__(
        self,
        symbol: str = settings.market.symbol,
        interval: str = settings.market.timeframe,
        output_dir: str = settings.data.raw_dir,
    ):
        self.symbol = symbol
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.symbol.lower()}_{self.interval}.csv"
        
        self.timeout = 30
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        self.limit = 1000

    @staticmethod
    def _date_to_ms(date_string: str) -> int:
        dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def fetch_klines_batch(
        self, start_time: int, end_time: int | None = None
    ) -> List[list]:
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "startTime": start_time,
            "limit": self.limit,
        }
        if end_time is not None:
            params["endTime"] = end_time

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(BINANCE_KLINES_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries:
                    logger.error(f"Failed to fetch batch after {self.max_retries} attempts.")
                    raise
                time.sleep(self.retry_delay)
        return []

    def download_historical_data(
        self, start_date: str, end_date: str | None = None
    ) -> pd.DataFrame:
        start_time = self._date_to_ms(start_date)
        
        if end_date:
            end_time = self._date_to_ms(end_date)
        else:
            # We intentionally stop before the current hour to ensure completed candles.
            now = datetime.now(timezone.utc)
            current_hour_start = now.replace(minute=0, second=0, microsecond=0)
            end_time = int(current_hour_start.timestamp() * 1000) - 1

        all_data = []
        request_count = 0
        
        logger.info(f"Downloading {self.symbol} {self.interval} from {start_date}")

        while start_time < end_time:
            request_count += 1
            if request_count % 10 == 0:
                logger.info(f"Request #{request_count} | Downloaded rows so far: {len(all_data):,}")

            batch = self.fetch_klines_batch(start_time, end_time)
            if not batch:
                break

            all_data.extend(batch)
            last_open_time = batch[-1][0]
            start_time = last_open_time + 1
            
            # Rate limiting
            time.sleep(0.25)
            if len(batch) < self.limit:
                break

        if not all_data:
            logger.warning("No data was downloaded.")
            return pd.DataFrame(columns=self.COLUMNS)

        df = self._process_and_save(all_data)
        logger.info(f"Download complete! Saved {len(df):,} rows to {self.output_file}")
        return df

    def _process_and_save(self, all_data: List[list]) -> pd.DataFrame:
        df = pd.DataFrame(all_data, columns=self.COLUMNS)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["open_time"])
        
        # Convert timestamps
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        numeric_columns = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep only the fields we currently need
        df = df[["open_time", "open", "high", "low", "close", "volume", "close_time", "number_of_trades"]]
        
        # Merge with existing file if it exists (append instead of override entirely)
        if self.output_file.exists():
            try:
                existing_df = pd.read_csv(self.output_file)
                existing_df["open_time"] = pd.to_datetime(existing_df["open_time"], utc=True)
                existing_df["close_time"] = pd.to_datetime(existing_df["close_time"], utc=True)
                df = pd.concat([existing_df, df]).drop_duplicates(subset=["open_time"])
            except Exception as e:
                logger.warning(f"Could not read existing file: {e}. Overwriting.")

        # Sort and save
        df = df.sort_values("open_time").reset_index(drop=True)
        df.to_csv(self.output_file, index=False)
        return df
