from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/btc_usdt_1h.csv")


def main() -> None:
    print("=" * 60)
    print("BTC DATA VALIDATION")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)

    # ---------------------------------------------------------
    # 1. Basic information
    # ---------------------------------------------------------

    print("\n1. BASIC INFORMATION")
    print("-" * 40)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    # ---------------------------------------------------------
    # 2. Convert timestamps
    # ---------------------------------------------------------

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        format="mixed",
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        utc=True,
        format="mixed",
    )

    # ---------------------------------------------------------
    # 3. Missing values
    # ---------------------------------------------------------

    print("\n2. MISSING VALUES")
    print("-" * 40)

    missing = df.isna().sum()

    print(missing)

    if missing.sum() == 0:
        print("\n✓ No missing values")

    # ---------------------------------------------------------
    # 4. Duplicate candles
    # ---------------------------------------------------------

    print("\n3. DUPLICATES")
    print("-" * 40)

    duplicate_count = df["open_time"].duplicated().sum()

    print(f"Duplicate timestamps: {duplicate_count}")

    if duplicate_count == 0:
        print("✓ No duplicate candles")

    # ---------------------------------------------------------
    # 5. Chronological order
    # ---------------------------------------------------------

    print("\n4. CHRONOLOGICAL ORDER")
    print("-" * 40)

    is_sorted = df["open_time"].is_monotonic_increasing

    print(f"Sorted: {is_sorted}")

    # ---------------------------------------------------------
    # 6. Candle interval check
    # ---------------------------------------------------------

    print("\n5. CANDLE INTERVALS")
    print("-" * 40)

    time_diff = df["open_time"].diff()

    expected_interval = pd.Timedelta(hours=1)

    wrong_intervals = (
        time_diff.iloc[1:] != expected_interval
    )

    wrong_count = wrong_intervals.sum()

    print(f"Unexpected intervals: {wrong_count}")

    if wrong_count == 0:
        print("✓ Every candle is exactly 1 hour apart")
    else:
        print("\nUnexpected intervals:")

        problematic = df.loc[
            wrong_intervals.index[wrong_intervals]
        ]

        print(problematic.head(20))

    # ---------------------------------------------------------
    # 7. OHLC validity
    # ---------------------------------------------------------

    print("\n6. OHLC VALIDATION")
    print("-" * 40)

    invalid_ohlc = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )

    invalid_count = invalid_ohlc.sum()

    print(f"Invalid OHLC candles: {invalid_count}")

    if invalid_count == 0:
        print("✓ OHLC values are structurally valid")

    # ---------------------------------------------------------
    # 8. Non-positive prices
    # ---------------------------------------------------------

    print("\n7. PRICE VALIDATION")
    print("-" * 40)

    invalid_prices = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
    )

    print(
        f"Non-positive price rows: {invalid_prices.sum()}"
    )

    # ---------------------------------------------------------
    # 9. Volume validation
    # ---------------------------------------------------------

    print("\n8. VOLUME VALIDATION")
    print("-" * 40)

    negative_volume = (
        df["volume"] < 0
    ).sum()

    print(f"Negative volume rows: {negative_volume}")

    # ---------------------------------------------------------
    # 10. Date range
    # ---------------------------------------------------------

    print("\n9. DATE RANGE")
    print("-" * 40)

    print(f"Start: {df['open_time'].min()}")
    print(f"End  : {df['open_time'].max()}")

    # ---------------------------------------------------------
    # 11. Basic statistics
    # ---------------------------------------------------------

    print("\n10. BASIC STATISTICS")
    print("-" * 40)

    print(
        df[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "number_of_trades",
            ]
        ].describe()
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()