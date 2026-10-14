from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = Path(
    "data/processed/btc_usdt_1h_features_v2.csv"
)

FEATURE_COLUMNS = [
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


TARGET_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


def main() -> None:

    print("=" * 70)
    print("BTC V2 FEATURE DATASET INSPECTION")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # 1. Missing values
    # ---------------------------------------------------------

    print("\n1. MISSING VALUES")
    print("-" * 50)

    missing = df[FEATURE_COLUMNS].isna().sum()

    if missing.sum() == 0:
        print("✓ No missing feature values")
    else:
        print(missing[missing > 0])

    # ---------------------------------------------------------
    # 2. Infinite values
    # ---------------------------------------------------------

    print("\n2. INFINITE VALUES")
    print("-" * 50)

    numeric = df[
        FEATURE_COLUMNS
    ].select_dtypes(
        include=np.number
    )

    inf_counts = np.isinf(numeric).sum()

    if inf_counts.sum() == 0:
        print("✓ No infinite values")
    else:
        print(inf_counts[inf_counts > 0])

    # ---------------------------------------------------------
    # 3. Target distribution
    # ---------------------------------------------------------

    print("\n3. TARGET DISTRIBUTION")
    print("-" * 50)

    counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    for target, count in counts.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{TARGET_NAMES[target]:8s}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # 4. Feature statistics
    # ---------------------------------------------------------

    print("\n4. FEATURE STATISTICS")
    print("-" * 50)

    stats = (
        df[FEATURE_COLUMNS]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
    )

    print(stats)

    # ---------------------------------------------------------
    # 5. Feature correlations
    # ---------------------------------------------------------

    print("\n5. HIGH FEATURE CORRELATIONS")
    print("-" * 50)

    correlation = (
        df[FEATURE_COLUMNS]
        .corr()
        .abs()
    )

    upper = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1,
        ).astype(bool)
    )

    high_corr = (
        upper
        .stack()
        .sort_values(
            ascending=False
        )
    )

    if len(high_corr) == 0:
        print("No feature correlations found.")
    else:
        print(
            high_corr.head(20)
        )

    # ---------------------------------------------------------
    # 6. Feature averages by target
    # ---------------------------------------------------------

    print("\n6. FEATURE MEANS BY TARGET")
    print("-" * 50)

    means = (
        df.groupby("target")[
            FEATURE_COLUMNS
        ]
        .mean()
        .T
    )

    means.columns = [
        TARGET_NAMES[int(c)]
        for c in means.columns
    ]

    print(means)

    # ---------------------------------------------------------
    # 7. Future return distribution by class
    # ---------------------------------------------------------

    print("\n7. FUTURE RETURN BY TARGET")
    print("-" * 50)

    grouped = (
        df.groupby("target")[
            "future_1h_return"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
    )

    grouped.index = [
        TARGET_NAMES[int(i)]
        for i in grouped.index
    ]

    print(grouped)

    # ---------------------------------------------------------
    # 8. Check suspicious feature ranges
    # ---------------------------------------------------------

    print("\n8. EXTREME VALUES")
    print("-" * 50)

    for column in FEATURE_COLUMNS:

        maximum = df[column].max()
        minimum = df[column].min()

        if (
            abs(maximum) > 10
            or abs(minimum) > 10
        ):

            print(
                f"{column:25s} "
                f"min={minimum:.6f} "
                f"max={maximum:.6f}"
            )

    print("\n" + "=" * 70)
    print("V2 INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()