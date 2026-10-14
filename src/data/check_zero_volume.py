from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/btc_usdt_1h.csv")


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    zero_volume = df[df["volume"] == 0].copy()
    zero_trades = df[df["number_of_trades"] == 0].copy()

    print("=" * 70)
    print("BTC ZERO-VOLUME / ZERO-TRADE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal rows: {len(df):,}")

    print("\n1. ZERO VOLUME")
    print("-" * 50)

    print(f"Zero-volume candles: {len(zero_volume)}")

    if len(zero_volume) > 0:
        print("\nZero-volume rows:")
        print(
            zero_volume[
                [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "number_of_trades",
                ]
            ].to_string(index=False)
        )
    else:
        print("✓ No zero-volume candles")

    print("\n2. ZERO TRADES")
    print("-" * 50)

    print(f"Zero-trade candles: {len(zero_trades)}")

    if len(zero_trades) > 0:
        print("\nZero-trade rows:")
        print(
            zero_trades[
                [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "number_of_trades",
                ]
            ].to_string(index=False)
        )
    else:
        print("✓ No zero-trade candles")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()