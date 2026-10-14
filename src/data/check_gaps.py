from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/btc_usdt_1h.csv")


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = df.sort_values("open_time").reset_index(drop=True)

    # Difference between consecutive candles
    df["time_diff"] = df["open_time"].diff()

    expected = pd.Timedelta(hours=1)

    gaps = df[
        (df["time_diff"] != expected)
        & df["time_diff"].notna()
    ].copy()

    print("=" * 70)
    print("BTC DATA GAP ANALYSIS")
    print("=" * 70)

    print(f"Total gaps: {len(gaps)}")
    print()

    for index, row in gaps.iterrows():

        previous_time = df.loc[index - 1, "open_time"]
        current_time = row["open_time"]
        gap = row["time_diff"]

        missing_candles = int(
            gap / expected
        ) - 1

        print(f"Gap #{index}")
        print(f"Previous candle : {previous_time}")
        print(f"Current candle  : {current_time}")
        print(f"Time difference : {gap}")
        print(f"Missing candles : {missing_candles}")
        print("-" * 50)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()