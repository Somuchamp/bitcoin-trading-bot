from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_FILE = Path(
    "data/processed/xgboost_v2_test_predictions.csv"
)


def analyze_up_signals(df: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE UP SIGNALS")
    print("=" * 70)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    rows = []

    for threshold in thresholds:

        mask = (
            df["p_up"] >= threshold
        )

        selected = df.loc[mask]

        if len(selected) == 0:
            continue

        future_returns = (
            selected["future_1h_return"]
        )

        rows.append(
            {
                "threshold": threshold,
                "samples": len(selected),
                "coverage": (
                    len(selected)
                    / len(df)
                    * 100
                ),
                "avg_return": (
                    future_returns.mean()
                    * 100
                ),
                "median_return": (
                    future_returns.median()
                    * 100
                ),
                "positive_rate": (
                    (future_returns > 0).mean()
                    * 100
                ),
                "meaningful_up_rate": (
                    (future_returns > 0.0015).mean()
                    * 100
                ),
                "mean_abs_return": (
                    future_returns.abs().mean()
                    * 100
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.2f}".format,

                "samples":
                    "{:.0f}".format,

                "coverage":
                    lambda x: f"{x:.2f}%",

                "avg_return":
                    lambda x: f"{x:.4f}%",

                "median_return":
                    lambda x: f"{x:.4f}%",

                "positive_rate":
                    lambda x: f"{x:.2f}%",

                "meaningful_up_rate":
                    lambda x: f"{x:.2f}%",

                "mean_abs_return":
                    lambda x: f"{x:.4f}%",
            },
        )
    )


def analyze_down_signals(df: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE DOWN SIGNALS")
    print("=" * 70)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    rows = []

    for threshold in thresholds:

        mask = (
            df["p_down"] >= threshold
        )

        selected = df.loc[mask]

        if len(selected) == 0:
            continue

        future_returns = (
            selected["future_1h_return"]
        )

        rows.append(
            {
                "threshold": threshold,
                "samples": len(selected),
                "coverage": (
                    len(selected)
                    / len(df)
                    * 100
                ),
                "avg_return": (
                    future_returns.mean()
                    * 100
                ),
                "median_return": (
                    future_returns.median()
                    * 100
                ),
                "negative_rate": (
                    (future_returns < 0).mean()
                    * 100
                ),
                "meaningful_down_rate": (
                    (future_returns < -0.0015).mean()
                    * 100
                ),
                "mean_abs_return": (
                    future_returns.abs().mean()
                    * 100
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.2f}".format,

                "samples":
                    "{:.0f}".format,

                "coverage":
                    lambda x: f"{x:.2f}%",

                "avg_return":
                    lambda x: f"{x:.4f}%",

                "median_return":
                    lambda x: f"{x:.4f}%",

                "negative_rate":
                    lambda x: f"{x:.2f}%",

                "meaningful_down_rate":
                    lambda x: f"{x:.2f}%",

                "mean_abs_return":
                    lambda x: f"{x:.4f}%",
            },
        )
    )


def analyze_neutral_signals(df: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE NEUTRAL SIGNALS")
    print("=" * 70)

    thresholds = [0.50, 0.60, 0.70, 0.80]

    rows = []

    for threshold in thresholds:

        mask = (
            df["p_neutral"] >= threshold
        )

        selected = df.loc[mask]

        if len(selected) == 0:
            continue

        future_returns = (
            selected["future_1h_return"]
        )

        rows.append(
            {
                "threshold": threshold,
                "samples": len(selected),
                "coverage": (
                    len(selected)
                    / len(df)
                    * 100
                ),
                "avg_return": (
                    future_returns.mean()
                    * 100
                ),
                "mean_abs_return": (
                    future_returns.abs().mean()
                    * 100
                ),
                "inside_neutral_range": (
                    (
                        future_returns.abs()
                        <= 0.0015
                    ).mean()
                    * 100
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.2f}".format,

                "samples":
                    "{:.0f}".format,

                "coverage":
                    lambda x: f"{x:.2f}%",

                "avg_return":
                    lambda x: f"{x:.4f}%",

                "mean_abs_return":
                    lambda x: f"{x:.4f}%",

                "inside_neutral_range":
                    lambda x: f"{x:.2f}%",
            },
        )
    )


def main() -> None:

    print("=" * 70)
    print("V2 SIGNAL / RETURN ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    analyze_up_signals(df)

    analyze_down_signals(df)

    analyze_neutral_signals(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

