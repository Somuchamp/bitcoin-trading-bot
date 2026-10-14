from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = Path(
    "data/processed/btc_usdt_1h_features.csv"
)


FEATURE_COLUMNS = [
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


def main() -> None:

    print("=" * 70)
    print("BTC FEATURE DATASET INSPECTION")
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

    print(missing[missing > 0])

    if missing.sum() == 0:
        print("✓ No missing feature values")

    # ---------------------------------------------------------
    # 2. Infinite values
    # ---------------------------------------------------------

    print("\n2. INFINITE VALUES")
    print("-" * 50)

    numeric_features = df[FEATURE_COLUMNS].select_dtypes(
        include=np.number
    )

    infinite_counts = np.isinf(
        numeric_features
    ).sum()

    print(infinite_counts[infinite_counts > 0])

    if infinite_counts.sum() == 0:
        print("✓ No infinite feature values")

    # ---------------------------------------------------------
    # 3. Feature statistics
    # ---------------------------------------------------------

    print("\n3. FEATURE STATISTICS")
    print("-" * 50)

    print(
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

    # ---------------------------------------------------------
    # 4. Target distribution
    # ---------------------------------------------------------

    print("\n4. TARGET DISTRIBUTION")
    print("-" * 50)

    target_counts = df["target"].value_counts()

    target_percent = (
        df["target"]
        .value_counts(normalize=True)
        .mul(100)
    )

    print(
        pd.DataFrame(
            {
                "count": target_counts,
                "percentage": target_percent,
            }
        ).sort_index()
    )

    # ---------------------------------------------------------
    # 5. Feature correlations
    # ---------------------------------------------------------

    print("\n5. HIGH FEATURE CORRELATIONS")
    print("-" * 50)

    correlation = df[
        FEATURE_COLUMNS
    ].corr().abs()

    upper = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1,
        ).astype(bool)
    )

    highly_correlated = (
        upper
        .stack()
        .sort_values(ascending=False)
    )

    if len(highly_correlated) == 0:
        print("No correlations found.")
    else:
        print(
            highly_correlated.head(20)
        )

    # ---------------------------------------------------------
    # 6. Target relationship
    # ---------------------------------------------------------

    print("\n6. AVERAGE FEATURE VALUES BY TARGET")
    print("-" * 50)

    target_means = (
        df.groupby("target")[FEATURE_COLUMNS]
        .mean()
        .T
    )

    target_means.columns = [
        "DOWN",
        "UP",
    ]

    target_means["difference"] = (
        target_means["UP"]
        - target_means["DOWN"]
    )

    print(
        target_means.sort_values(
            "difference",
            key=lambda x: x.abs(),
            ascending=False,
        ).head(20)
    )

    # ---------------------------------------------------------
    # 7. Future return statistics
    # ---------------------------------------------------------

    print("\n7. FUTURE 1H RETURN")
    print("-" * 50)

    print(
        df["future_1h_return"].describe()
    )

    print("\nQuantiles:")

    print(
        df["future_1h_return"]
        .quantile(
            [
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()