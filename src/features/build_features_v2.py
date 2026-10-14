from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/btc_usdt_1h_clean.csv"
)

OUTPUT_FILE = Path(
    "data/processed/btc_usdt_1h_features_v2.csv"
)


def calculate_rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

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

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_segment_id(
    df: pd.DataFrame,
) -> pd.DataFrame:

    time_diff = df["open_time"].diff()

    new_segment = (
        time_diff != pd.Timedelta(hours=1)
    )

    df["segment_id"] = new_segment.cumsum()

    return df


def build_segment_features(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    segment = segment.copy()

    close = segment["close"]

    # ========================================================
    # Returns
    # ========================================================

    segment["return_1h"] = close.pct_change(1)
    segment["return_3h"] = close.pct_change(3)
    segment["return_6h"] = close.pct_change(6)
    segment["return_12h"] = close.pct_change(12)
    segment["return_24h"] = close.pct_change(24)

    # ========================================================
    # Moving averages
    # ========================================================

    sma_10 = close.rolling(
        10,
        min_periods=10,
    ).mean()

    sma_20 = close.rolling(
        20,
        min_periods=20,
    ).mean()

    sma_50 = close.rolling(
        50,
        min_periods=50,
    ).mean()

    ema_10 = close.ewm(
        span=10,
        adjust=False,
        min_periods=10,
    ).mean()

    ema_20 = close.ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    # ========================================================
    # Normalized trend features
    # ========================================================

    segment["price_vs_sma_10"] = (
        close / sma_10 - 1
    )

    segment["price_vs_sma_20"] = (
        close / sma_20 - 1
    )

    segment["price_vs_sma_50"] = (
        close / sma_50 - 1
    )

    segment["price_vs_ema_10"] = (
        close / ema_10 - 1
    )

    segment["price_vs_ema_20"] = (
        close / ema_20 - 1
    )

    # Distance between averages

    segment["sma_10_vs_sma_50"] = (
        sma_10 / sma_50 - 1
    )

    segment["ema_10_vs_ema_20"] = (
        ema_10 / ema_20 - 1
    )

    # ========================================================
    # RSI
    # ========================================================

    segment["rsi_14"] = calculate_rsi(
        close,
        14,
    )

    # ========================================================
    # MACD
    # ========================================================

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

    macd = ema_12 - ema_26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    segment["macd_pct"] = (
        macd / close
    )

    segment["macd_signal_pct"] = (
        macd_signal / close
    )

    segment["macd_histogram_pct"] = (
        (macd - macd_signal) / close
    )

    # ========================================================
    # Volatility
    # ========================================================

    segment["volatility_24h"] = (
        segment["return_1h"]
        .rolling(
            24,
            min_periods=24,
        )
        .std()
    )

    atr = calculate_atr(
        segment,
        14,
    )

    segment["atr_pct"] = (
        atr / close
    )

    # ========================================================
    # Volume
    # ========================================================

    volume_ma_24 = segment[
        "volume"
    ].rolling(
        24,
        min_periods=24,
    ).mean()

    segment["volume_ratio"] = (
        segment["volume"]
        / volume_ma_24
    )

    segment["volume_change"] = (
        segment["volume"].pct_change()
    )

    # ========================================================
    # Candle structure
    # ========================================================

    segment["high_low_range_pct"] = (
        (segment["high"] - segment["low"])
        / close
    )

    segment["body_pct"] = (
        (segment["close"] - segment["open"])
        / segment["open"]
    )

    segment["upper_wick_pct"] = (
        (
            segment["high"]
            - segment[
                ["open", "close"]
            ].max(axis=1)
        )
        / close
    )

    segment["lower_wick_pct"] = (
        (
            segment[
                ["open", "close"]
            ].min(axis=1)
            - segment["low"]
        )
        / close
    )

    # ========================================================
    # Future return
    # ========================================================

    segment["future_1h_return"] = (
        close.shift(-1) / close - 1
    )

    # ========================================================
    # 3-class target
    #
    # ±0.15% threshold
    # ========================================================

    threshold = 0.0015

    segment["target"] = np.select(
        [
            segment["future_1h_return"] < -threshold,
            segment["future_1h_return"] > threshold,
        ],
        [
            0,  # DOWN
            2,  # UP
        ],
        default=1,  # NEUTRAL
    )

    return segment


def main() -> None:

    print("=" * 70)
    print("BUILDING BTC V2 FEATURE DATASET")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

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

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    print(
        f"\nInput rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Segment by missing candles
    # --------------------------------------------------------

    df = add_segment_id(df)

    print(
        f"Continuous segments: "
        f"{df['segment_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Feature generation
    # --------------------------------------------------------

    df = (
        df.groupby(
            "segment_id",
            group_keys=False,
        )
        .apply(
            build_segment_features,
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Features used by model
    # --------------------------------------------------------

    feature_columns = [
        "return_1h",
        "return_3h",
        "return_6h",
        "return_12h",
        "return_24h",

        "price_vs_sma_10",
        "price_vs_sma_20",
        "price_vs_sma_50",
        "price_vs_ema_10",
        "price_vs_ema_20",

        "sma_10_vs_sma_50",
        "ema_10_vs_ema_20",

        "rsi_14",

        "macd_pct",
        "macd_signal_pct",
        "macd_histogram_pct",

        "volatility_24h",
        "atr_pct",

        "volume_ratio",
        "volume_change",

        "high_low_range_pct",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
    ]

    # --------------------------------------------------------
    # Remove insufficient rows
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=feature_columns
        + [
            "future_1h_return",
            "target",
        ]
    ).copy()

    removed = before - len(df)

    print(
        f"Rows removed: {removed:,}"
    )

    # --------------------------------------------------------
    # Ensure target is integer
    # --------------------------------------------------------

    df["target"] = (
        df["target"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Save only useful columns + metadata
    # --------------------------------------------------------

    output_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "future_1h_return",
        "target",
        *feature_columns,
    ]

    df = df[output_columns]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("V2 FEATURE DATASET CREATED")
    print("=" * 70)

    print(
        f"Rows:       {len(df):,}"
    )

    print(
        f"Columns:    {len(df.columns)}"
    )

    print(
        f"Output:     {OUTPUT_FILE}"
    )

    print("\nTarget distribution:")

    target_names = {
        0: "DOWN",
        1: "NEUTRAL",
        2: "UP",
    }

    counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    for value, count in counts.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{target_names[value]:8s}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    print("\nFeature columns:")

    for column in feature_columns:
        print(f"  - {column}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
    