from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/btc_usdt_1h_clean.csv"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "btc_usdt_1h_features.csv"


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def add_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new segment whenever the hourly sequence is broken.

    This prevents rolling calculations from treating large
    missing-data gaps as continuous observations.
    """

    time_diff = df["open_time"].diff()

    gap = time_diff != pd.Timedelta(hours=1)

    df["segment_id"] = gap.cumsum()

    return df


def add_features_to_segment(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    segment = segment.copy()

    close = segment["close"]

    # ---------------------------------------------------------
    # Price returns
    # ---------------------------------------------------------

    segment["return_1h"] = close.pct_change(1)

    segment["return_3h"] = close.pct_change(3)

    segment["return_6h"] = close.pct_change(6)

    segment["return_24h"] = close.pct_change(24)

    # ---------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------

    segment["sma_10"] = close.rolling(
        window=10,
        min_periods=10,
    ).mean()

    segment["sma_20"] = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    segment["sma_50"] = close.rolling(
        window=50,
        min_periods=50,
    ).mean()

    segment["ema_10"] = close.ewm(
        span=10,
        adjust=False,
        min_periods=10,
    ).mean()

    segment["ema_20"] = close.ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    # ---------------------------------------------------------
    # Momentum
    # ---------------------------------------------------------

    segment["rsi_14"] = calculate_rsi(
        close,
        period=14,
    )

    ema_12 = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    segment["macd"] = ema_12 - ema_26

    segment["macd_signal"] = segment["macd"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    segment["macd_histogram"] = (
        segment["macd"]
        - segment["macd_signal"]
    )

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    segment["rolling_volatility_24h"] = (
        segment["return_1h"]
        .rolling(
            window=24,
            min_periods=24,
        )
        .std()
    )

    segment["atr_14"] = calculate_atr(
        segment,
        period=14,
    )

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    segment["volume_change"] = (
        segment["volume"].pct_change(1)
    )

    volume_ma_24 = segment["volume"].rolling(
        window=24,
        min_periods=24,
    ).mean()

    segment["volume_ratio"] = (
        segment["volume"] / volume_ma_24
    )

    # ---------------------------------------------------------
    # Relative position to moving averages
    # ---------------------------------------------------------

    segment["price_vs_sma_20"] = (
        close / segment["sma_20"] - 1
    )

    segment["price_vs_sma_50"] = (
        close / segment["sma_50"] - 1
    )

    # ---------------------------------------------------------
    # Future target
    # ---------------------------------------------------------

    # IMPORTANT:
    # This uses future data ONLY to create the target.
    # We will NEVER use this column as an input feature.

    segment["future_1h_return"] = (
        close.shift(-1) / close - 1
    )

    segment["target"] = (
        segment["future_1h_return"] > 0
    ).astype("Int64")

    return segment


def main() -> None:

    print("=" * 70)
    print("BUILDING BTC FEATURE DATASET")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        utc=True,
        format="mixed",
    )

    df = df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    print(f"\nInput rows: {len(df):,}")

    # ---------------------------------------------------------
    # Segment data by missing-hour gaps
    # ---------------------------------------------------------

    df = add_segment_id(df)

    segment_count = df["segment_id"].nunique()

    print(f"Continuous segments: {segment_count}")

    # ---------------------------------------------------------
    # Calculate features separately per segment
    # ---------------------------------------------------------

    df = (
        df.groupby(
            "segment_id",
            group_keys=False,
        )
        .apply(
            add_features_to_segment,
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Remove rows without enough history / future target
    # ---------------------------------------------------------

    feature_columns = [
        "return_1h",
        "return_3h",
        "return_6h",
        "return_24h",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_10",
        "ema_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "rolling_volatility_24h",
        "atr_14",
        "volume_change",
        "volume_ratio",
        "price_vs_sma_20",
        "price_vs_sma_50",
    ]

    before_drop = len(df)

    df = df.dropna(
        subset=feature_columns + [
            "future_1h_return",
            "target",
        ]
    ).copy()

    removed = before_drop - len(df)

    print(
        f"Rows removed due to insufficient "
        f"history/future target: {removed:,}"
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
    print("FEATURE DATASET CREATED")
    print("=" * 70)

    print(f"Rows:       {len(df):,}")
    print(f"Columns:    {len(df.columns)}")
    print(f"Output:     {OUTPUT_FILE}")

    print("\nTarget distribution:")

    print(
        df["target"]
        .value_counts(normalize=True)
        .sort_index()
        .rename(
            {
                0: "DOWN",
                1: "UP",
            }
        )
    )

    print("\nFeature columns:")

    for column in feature_columns:
        print(f"  - {column}")

    print("\nLast 5 rows:")
    print(
        df[
            [
                "open_time",
                "close",
                "rsi_14",
                "macd",
                "volume_ratio",
                "future_1h_return",
                "target",
            ]
        ].tail()
    )


if __name__ == "__main__":
    main()