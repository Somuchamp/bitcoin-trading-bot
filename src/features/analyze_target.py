from pathlib import Path

import pandas as pd


DATA_FILE = Path(
    "data/processed/btc_usdt_1h_features.csv"
)


def main() -> None:

    print("=" * 70)
    print("BTC FUTURE RETURN / TARGET ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    returns = df["future_1h_return"].dropna()

    print(f"\nTotal observations: {len(returns):,}")

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    print("\n1. BASIC STATISTICS")
    print("-" * 50)

    print(returns.describe())

    # ---------------------------------------------------------
    # Quantiles
    # ---------------------------------------------------------

    print("\n2. RETURN QUANTILES")
    print("-" * 50)

    quantiles = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.50,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.98,
        0.99,
    ]

    print(
        returns.quantile(quantiles).to_string()
    )

    # ---------------------------------------------------------
    # Candidate thresholds
    # ---------------------------------------------------------

    thresholds = [
        0.0005,   # 0.05%
        0.0010,   # 0.10%
        0.0015,   # 0.15%
        0.0020,   # 0.20%
        0.0025,   # 0.25%
        0.0030,   # 0.30%
        0.0040,   # 0.40%
        0.0050,   # 0.50%
    ]

    print("\n3. CANDIDATE 3-CLASS THRESHOLDS")
    print("-" * 70)

    rows = []

    for threshold in thresholds:

        up = returns > threshold
        down = returns < -threshold
        neutral = ~(up | down)

        up_count = up.sum()
        down_count = down.sum()
        neutral_count = neutral.sum()

        total = len(returns)

        rows.append(
            {
                "threshold": threshold,
                "DOWN_count": down_count,
                "DOWN_pct": down_count / total * 100,
                "NEUTRAL_count": neutral_count,
                "NEUTRAL_pct": neutral_count / total * 100,
                "UP_count": up_count,
                "UP_pct": up_count / total * 100,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "threshold": lambda x: f"{x * 100:.2f}%",
                "DOWN_count": "{:.0f}".format,
                "DOWN_pct": lambda x: f"{x:.2f}%",
                "NEUTRAL_count": "{:.0f}".format,
                "NEUTRAL_pct": lambda x: f"{x:.2f}%",
                "UP_count": "{:.0f}".format,
                "UP_pct": lambda x: f"{x:.2f}%",
            },
        )
    )

    # ---------------------------------------------------------
    # Small-movement analysis
    # ---------------------------------------------------------

    print("\n4. SMALL MOVEMENT ANALYSIS")
    print("-" * 70)

    ranges = [
        0.00025,  # 0.025%
        0.00050,  # 0.05%
        0.00100,  # 0.10%
        0.00150,  # 0.15%
        0.00200,  # 0.20%
        0.00300,  # 0.30%
        0.00500,  # 0.50%
    ]

    rows = []

    for threshold in ranges:

        inside = (
            returns.abs() <= threshold
        )

        outside = ~inside

        rows.append(
            {
                "range": threshold,
                "inside_count": inside.sum(),
                "inside_pct": inside.mean() * 100,
                "outside_count": outside.sum(),
                "outside_pct": outside.mean() * 100,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "range": lambda x: f"±{x * 100:.3f}%",
                "inside_count": "{:.0f}".format,
                "inside_pct": lambda x: f"{x:.2f}%",
                "outside_count": "{:.0f}".format,
                "outside_pct": lambda x: f"{x:.2f}%",
            },
        )
    )

    # ---------------------------------------------------------
    # Positive / negative movement
    # ---------------------------------------------------------

    print("\n5. POSITIVE VS NEGATIVE RETURNS")
    print("-" * 70)

    positive = (
        returns > 0
    ).mean() * 100

    negative = (
        returns < 0
    ).mean() * 100

    zero = (
        returns == 0
    ).mean() * 100

    print(f"Positive returns : {positive:.2f}%")
    print(f"Negative returns : {negative:.2f}%")
    print(f"Zero returns     : {zero:.2f}%")

    print("\n" + "=" * 70)
    print("TARGET ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()