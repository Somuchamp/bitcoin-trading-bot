import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.binance.com/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "1h"

# Start with 2021.
# This gives us multiple market regimes.
START_DATE = "2021-01-01"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "btc_usdt_1h.csv"


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


def date_to_ms(date_string: str) -> int:
    """
    Convert YYYY-MM-DD to UTC milliseconds.
    """
    dt = datetime.strptime(date_string, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp() * 1000)


def get_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int | None = None,
    limit: int = 1000,
) -> list:

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit,
    }

    if end_time is not None:
        params["endTime"] = end_time

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def main() -> None:

    start_time = date_to_ms(START_DATE)

    # Current UTC time
    now = datetime.now(timezone.utc)

    # We intentionally stop before the current hour.
    current_hour_start = now.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    end_time = int(current_hour_start.timestamp() * 1000) - 1

    all_data = []

    request_count = 0

    print("=" * 60)
    print("Downloading Binance historical BTC/USDT data")
    print("=" * 60)

    print(f"Symbol      : {SYMBOL}")
    print(f"Interval    : {INTERVAL}")
    print(f"Start       : {START_DATE}")
    print(f"End         : {now.isoformat()}")
    print()

    while start_time < end_time:

        request_count += 1

        print(
            f"Request #{request_count} | "
            f"Downloaded rows so far: {len(all_data):,}"
        )

        try:

            batch = get_klines(
                symbol=SYMBOL,
                interval=INTERVAL,
                start_time=start_time,
                end_time=end_time,
                limit=1000,
            )

        except requests.RequestException as exc:

            print(f"Request failed: {exc}")
            print("Waiting 5 seconds before retrying...")

            time.sleep(5)
            continue

        if not batch:
            break

        all_data.extend(batch)

        # Move to the candle immediately after the last received candle.
        last_open_time = batch[-1][0]

        start_time = last_open_time + 1

        # Be polite to the API.
        time.sleep(0.25)

        # Safety check.
        if len(batch) < 1000:
            break

    if not all_data:
        raise RuntimeError("No data was downloaded.")

    df = pd.DataFrame(
        all_data,
        columns=COLUMNS,
    )

    # Remove duplicates.
    df = df.drop_duplicates(
        subset=["open_time"]
    )

    # Convert timestamps.
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    # Convert numerical columns.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Only retain COMPLETED candles.
    current_time = pd.Timestamp.now(tz="UTC")

    df = df[
        df["close_time"] <= current_time
    ]

    # Sort chronologically.
    df = df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    # Keep only the fields we currently need.
    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "number_of_trades",
        ]
    ]

    # Save.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Rows       : {len(df):,}")
    print(f"Start      : {df['open_time'].min()}")
    print(f"End        : {df['open_time'].max()}")
    print(f"Output     : {OUTPUT_FILE}")
    print()

    print("Missing values:")
    print(df.isna().sum())

    print()
    print("Last 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    main()