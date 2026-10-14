from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

DATA_FILE = Path(
    "data/processed/btc_usdt_1h_features.csv"
)

MODEL_FILE = Path(
    "models/xgboost_v1.json"
)

OUTPUT_FILE = Path(
    "data/processed/xgboost_v1_test_predictions.csv"
)


# ============================================================
# Features
# ============================================================

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

TARGET_COLUMN = "target"


def load_test_data():
    """Load the dataset and reproduce the exact test split used in training."""

    df = pd.read_csv(DATA_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    test_df = df.iloc[
        validation_end:
    ].copy()

    return test_df


def load_model():
    """Load the trained XGBoost model."""

    model = XGBClassifier()

    model.load_model(MODEL_FILE)

    return model


def threshold_analysis(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:

    print("\n" + "=" * 70)
    print("BULLISH THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    rows = []

    for threshold in thresholds:

        mask = probabilities >= threshold

        count = mask.sum()

        if count == 0:
            rows.append(
                {
                    "threshold": threshold,
                    "signals": 0,
                    "signal_pct": 0.0,
                    "precision": np.nan,
                    "actual_up_rate": np.nan,
                }
            )
            continue

        selected_actual = y_true[mask]

        precision = selected_actual.mean()

        rows.append(
            {
                "threshold": threshold,
                "signals": count,
                "signal_pct": count / len(y_true) * 100,
                "precision": precision,
                "actual_up_rate": precision,
            }
        )

    result = pd.DataFrame(rows)

    result["precision"] = (
        result["precision"] * 100
    )

    result["actual_up_rate"] = (
        result["actual_up_rate"] * 100
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "signals": "{:.0f}".format,
                "signal_pct": "{:.2f}%".format,
                "precision": (
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                ),
                "actual_up_rate": (
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                ),
            },
        )
    )


def bearish_threshold_analysis(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:

    print("\n" + "=" * 70)
    print("BEARISH THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
    ]

    rows = []

    for threshold in thresholds:

        mask = probabilities <= threshold

        count = mask.sum()

        if count == 0:
            rows.append(
                {
                    "threshold": threshold,
                    "signals": 0,
                    "signal_pct": 0.0,
                    "down_rate": np.nan,
                }
            )
            continue

        selected_actual = y_true[mask]

        down_rate = (selected_actual == 0).mean()

        rows.append(
            {
                "threshold": threshold,
                "signals": count,
                "signal_pct": count / len(y_true) * 100,
                "down_rate": down_rate,
            }
        )

    result = pd.DataFrame(rows)

    result["down_rate"] = (
        result["down_rate"] * 100
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "signals": "{:.0f}".format,
                "signal_pct": "{:.2f}%".format,
                "down_rate": (
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                ),
            },
        )
    )


def probability_bucket_analysis(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:

    print("\n" + "=" * 70)
    print("PROBABILITY BUCKET ANALYSIS")
    print("=" * 70)

    result = pd.DataFrame(
        {
            "actual": y_true.to_numpy(),
            "probability_up": probabilities,
        }
    )

    bins = [
        0.0,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        1.00,
    ]

    labels = [
        "<0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-1.00",
    ]

    result["bucket"] = pd.cut(
        result["probability_up"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    grouped = (
        result.groupby(
            "bucket",
            observed=False,
        )
        .agg(
            samples=("actual", "size"),
            avg_predicted_probability=(
                "probability_up",
                "mean",
            ),
            actual_up_rate=(
                "actual",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["avg_predicted_probability"] *= 100
    grouped["actual_up_rate"] *= 100

    print(
        grouped.to_string(
            index=False,
            formatters={
                "avg_predicted_probability": (
                    lambda x: f"{x:.2f}%"
                ),
                "actual_up_rate": (
                    lambda x: f"{x:.2f}%"
                ),
            },
        )
    )


def confidence_analysis(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE PREDICTIONS")
    print("=" * 70)

    confidence = np.maximum(
        probabilities,
        1 - probabilities,
    )

    predicted_class = (
        probabilities >= 0.5
    ).astype(int)

    correct = (
        predicted_class
        == y_true.to_numpy()
    )

    levels = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    rows = []

    for level in levels:

        mask = confidence >= level

        count = mask.sum()

        if count == 0:

            rows.append(
                {
                    "confidence": level,
                    "predictions": 0,
                    "coverage": 0.0,
                    "accuracy": np.nan,
                }
            )

            continue

        accuracy = correct[mask].mean()

        rows.append(
            {
                "confidence": level,
                "predictions": count,
                "coverage": count / len(y_true) * 100,
                "accuracy": accuracy * 100,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "confidence": "{:.2f}".format,
                "predictions": "{:.0f}".format,
                "coverage": "{:.2f}%".format,
                "accuracy": (
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                ),
            },
        )
    )


def save_predictions(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> None:

    output = test_df[
        [
            "open_time",
            "close",
            "future_1h_return",
            "target",
        ]
    ].copy()

    output["probability_up"] = probabilities

    output["probability_down"] = (
        1 - output["probability_up"]
    )

    output["prediction"] = (
        output["probability_up"] >= 0.5
    ).astype(int)

    output["prediction_label"] = np.where(
        output["prediction"] == 1,
        "UP",
        "DOWN",
    )

    output["correct"] = (
        output["prediction"]
        == output["target"]
    )

    output["confidence"] = np.maximum(
        output["probability_up"],
        output["probability_down"],
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nPredictions saved to: {OUTPUT_FILE}"
    )


def main() -> None:

    print("=" * 70)
    print("XGBOOST V1 PREDICTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    test_df = load_test_data()

    model = load_model()

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("OVERALL TEST METRICS")
    print("=" * 70)

    print(f"Samples    : {len(y_test):,}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"ROC AUC    : {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROBABILITY STATISTICS")
    print("=" * 70)

    print(
        pd.Series(probabilities).describe()
    )

    print("\nProbability quantiles:")

    print(
        pd.Series(probabilities).quantile(
            [
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    threshold_analysis(
        y_test.reset_index(drop=True),
        probabilities,
    )

    bearish_threshold_analysis(
        y_test.reset_index(drop=True),
        probabilities,
    )

    probability_bucket_analysis(
        y_test.reset_index(drop=True),
        probabilities,
    )

    confidence_analysis(
        y_test.reset_index(drop=True),
        probabilities,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_predictions(
        test_df,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()