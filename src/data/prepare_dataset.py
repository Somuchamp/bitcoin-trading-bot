from pathlib import Path

import pandas as pd


RAW_FILE = Path("data/raw/btc_usdt_1h.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "btc_usdt_1h_clean.csv"


def main() -> None:
    print("=" * 70)
    print("PREPARING BTC MODELING DATASET")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load raw data
    # ---------------------------------------------------------

    df = pd.read_csv(RAW_FILE)

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

    print(f"\nRaw rows: {len(df):,}")

    # ---------------------------------------------------------
    # Remove duplicate timestamps
    # ---------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["open_time"],
        keep="first",
    )

    duplicates_removed = before - len(df)

    print(
        f"Duplicate rows removed: "
        f"{duplicates_removed}"
    )

    # ---------------------------------------------------------
    # Remove zero-volume / zero-trade candles
    # ---------------------------------------------------------

    zero_volume_mask = (
        (df["volume"] == 0)
        & (df["number_of_trades"] == 0)
    )

    zero_volume_count = zero_volume_mask.sum()

    print(
        f"Zero-volume candles removed: "
        f"{zero_volume_count}"
    )

    df = df[
        ~zero_volume_mask
    ].copy()

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Basic OHLC validation
    # ---------------------------------------------------------

    invalid_ohlc = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    invalid_count = invalid_ohlc.sum()

    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count} invalid OHLC rows."
        )

    # ---------------------------------------------------------
    # Price and volume validation
    # ---------------------------------------------------------

    if (df["close"] <= 0).any():
        raise ValueError(
            "Found non-positive close prices."
        )

    if (df["volume"] < 0).any():
        raise ValueError(
            "Found negative volume."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEAN DATASET CREATED")
    print("=" * 70)

    print(f"Rows:       {len(df):,}")
    print(f"Start:      {df['open_time'].min()}")
    print(f"End:        {df['open_time'].max()}")
    print(f"Output:     {OUTPUT_FILE}")

    print("\nIMPORTANT:")
    print("Missing time gaps were NOT filled.")
    print("The raw dataset remains unchanged.")


if __name__ == "__main__":
    main()