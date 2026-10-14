from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


DATA_FILE = Path(
    "data/processed/btc_usdt_1h_features_v2.csv"
)

MODEL_FILE = Path(
    "models/xgboost_v2.json"
)

OUTPUT_FILE = Path(
    "data/processed/xgboost_v2_test_predictions.csv"
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
    "upper_wick_pct",
    "lower_wick_pct",
]

TARGET_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


def main():

    print("=" * 70)
    print("XGBOOST V2 PROBABILITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    # Same chronological split used during training.
    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    test_df = df.iloc[
        validation_end:
    ].copy()

    print(
        f"\nTest samples: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    from xgboost import XGBClassifier

    model = XGBClassifier()

    model.load_model(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "target"
    ].astype(int)

    probabilities = (
        model.predict_proba(X_test)
    )

    predictions = (
        probabilities.argmax(axis=1)
    )

    test_df["p_down"] = probabilities[:, 0]
    test_df["p_neutral"] = probabilities[:, 1]
    test_df["p_up"] = probabilities[:, 2]

    test_df["prediction"] = predictions

    test_df["prediction_label"] = (
        test_df["prediction"]
        .map(TARGET_NAMES)
    )

    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2],
        )
    )

    # --------------------------------------------------------
    # Predicted class distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTED CLASS DISTRIBUTION")
    print("=" * 70)

    predicted_counts = (
        pd.Series(predictions)
        .value_counts()
        .sort_index()
    )

    for class_id, count in predicted_counts.items():

        print(
            f"{TARGET_NAMES[class_id]:8s}: "
            f"{count:5d} "
            f"({count / len(predictions) * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROBABILITY STATISTICS")
    print("=" * 70)

    for name, values in [
        ("P(DOWN)", test_df["p_down"]),
        ("P(NEUTRAL)", test_df["p_neutral"]),
        ("P(UP)", test_df["p_up"]),
    ]:

        print(f"\n{name}")

        print(
            values.describe()
        )

    # --------------------------------------------------------
    # Directional prediction accuracy
    #
    # Only count cases where model predicts UP or DOWN.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DIRECTIONAL PREDICTIONS ONLY")
    print("=" * 70)

    directional_mask = (
        predictions != 1
    )

    directional_predictions = (
        predictions[directional_mask]
    )

    directional_actual = (
        y_test.to_numpy()[directional_mask]
    )

    if len(directional_predictions) > 0:

        directional_accuracy = (
            directional_predictions
            == directional_actual
        ).mean()

        print(
            f"Directional predictions: "
            f"{len(directional_predictions):,}"
        )

        print(
            f"Coverage: "
            f"{len(directional_predictions) / len(y_test) * 100:.2f}%"
        )

        print(
            f"Directional accuracy: "
            f"{directional_accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Confidence analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    max_probability = probabilities.max(
        axis=1
    )

    predicted_class = predictions

    correct = (
        predicted_class
        == y_test.to_numpy()
    )

    confidence_levels = [
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    rows = []

    for threshold in confidence_levels:

        mask = (
            max_probability >= threshold
        )

        count = mask.sum()

        if count == 0:

            rows.append(
                {
                    "confidence": threshold,
                    "samples": 0,
                    "coverage": 0.0,
                    "accuracy": np.nan,
                }
            )

            continue

        rows.append(
            {
                "confidence": threshold,
                "samples": count,
                "coverage": (
                    count
                    / len(y_test)
                    * 100
                ),
                "accuracy": (
                    correct[mask].mean()
                    * 100
                ),
            }
        )

    confidence_df = pd.DataFrame(
        rows
    )

    print(
        confidence_df.to_string(
            index=False,
            formatters={
                "confidence":
                    lambda x: f"{x:.2f}",

                "samples":
                    "{:.0f}".format,

                "coverage":
                    lambda x: f"{x:.2f}%",

                "accuracy":
                    lambda x:
                    (
                        f"{x:.2f}%"
                        if pd.notna(x)
                        else "N/A"
                    ),
            },
        )
    )

    # --------------------------------------------------------
    # UP confidence analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE UP PREDICTIONS")
    print("=" * 70)

    up_thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    for threshold in up_thresholds:

        mask = (
            test_df["p_up"]
            >= threshold
        )

        count = mask.sum()

        if count == 0:
            print(
                f"P(UP) >= {threshold:.2f}: "
                f"no predictions"
            )
            continue

        actual_up_rate = (
            y_test.to_numpy()[mask] == 2
        ).mean()

        print(
            f"P(UP) >= {threshold:.2f}: "
            f"{count:5d} samples | "
            f"{count / len(y_test) * 100:6.2f}% coverage | "
            f"Actual UP: "
            f"{actual_up_rate * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # DOWN confidence analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE DOWN PREDICTIONS")
    print("=" * 70)

    for threshold in up_thresholds:

        mask = (
            test_df["p_down"]
            >= threshold
        )

        count = mask.sum()

        if count == 0:
            print(
                f"P(DOWN) >= {threshold:.2f}: "
                f"no predictions"
            )
            continue

        actual_down_rate = (
            y_test.to_numpy()[mask] == 0
        ).mean()

        print(
            f"P(DOWN) >= {threshold:.2f}: "
            f"{count:5d} samples | "
            f"{count / len(y_test) * 100:6.2f}% coverage | "
            f"Actual DOWN: "
            f"{actual_down_rate * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output_columns = [
        "open_time",
        "close",
        "future_1h_return",
        "target",
        "prediction",
        "prediction_label",
        "p_down",
        "p_neutral",
        "p_up",
    ]

    test_df[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PREDICTIONS SAVED")
    print("=" * 70)

    print(
        f"Path: {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("V2 ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()