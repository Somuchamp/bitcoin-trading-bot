from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


DATA_FILE = Path(
    "data/processed/btc_usdt_1h_features_v2.csv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_FILE = (
    MODEL_DIR / "xgboost_v2.json"
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


TARGET_COLUMN = "target"


TARGET_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


def print_metrics(
    name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:

    predictions = probabilities.argmax(axis=1)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy: "
        f"{accuracy_score(y_true, predictions):.4f}"
    )

    print(
        f"Macro F1: "
        f"{f1_score(y_true, predictions, average='macro'):.4f}"
    )

    print(
        f"Weighted F1: "
        f"{f1_score(y_true, predictions, average='weighted'):.4f}"
    )

    print(
        f"Log Loss: "
        f"{log_loss(y_true, probabilities):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1, 2],
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "DOWN",
                "NEUTRAL",
                "UP",
            ],
            zero_division=0,
        )
    )


def main() -> None:

    print("=" * 70)
    print("BTC MODEL V2 TRAINING")
    print("=" * 70)

    df = pd.read_csv(
        DATA_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    print(
        f"\nTotal samples: {len(df):,}"
    )

    # ========================================================
    # Chronological split
    # ========================================================

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    print("\nDATA SPLIT")
    print("-" * 50)

    print(
        f"Train      : {len(train_df):,} "
        f"({train_df['open_time'].iloc[0]} → "
        f"{train_df['open_time'].iloc[-1]})"
    )

    print(
        f"Validation : {len(validation_df):,} "
        f"({validation_df['open_time'].iloc[0]} → "
        f"{validation_df['open_time'].iloc[-1]})"
    )

    print(
        f"Test       : {len(test_df):,} "
        f"({test_df['open_time'].iloc[0]} → "
        f"{test_df['open_time'].iloc[-1]})"
    )

    # ========================================================
    # X / y
    # ========================================================

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # ========================================================
    # Baseline
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)

    majority_class = (
        y_train.value_counts()
        .idxmax()
    )

    baseline_predictions = np.full(
        len(y_test),
        majority_class,
    )

    baseline_accuracy = (
        accuracy_score(
            y_test,
            baseline_predictions,
        )
    )

    print(
        f"Majority class: "
        f"{TARGET_NAMES[int(majority_class)]}"
    )

    print(
        f"Test accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    # ========================================================
    # Logistic Regression
    # ========================================================

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    validation_lr = (
        logistic_model.predict_proba(
            X_validation
        )
    )

    print_metrics(
        "LOGISTIC REGRESSION — VALIDATION",
        y_validation,
        validation_lr,
    )

    # ========================================================
    # XGBoost
    # ========================================================

    print(
        "\nTraining XGBoost..."
    )

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    validation_xgb = (
        xgb_model.predict_proba(
            X_validation
        )
    )

    print_metrics(
        "XGBOOST — VALIDATION",
        y_validation,
        validation_xgb,
    )

    # ========================================================
    # Final test
    # ========================================================

    test_xgb = (
        xgb_model.predict_proba(
            X_test
        )
    )

    print_metrics(
        "XGBOOST — TEST",
        y_test,
        test_xgb,
    )

    # ========================================================
    # Feature importance
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = (
        pd.Series(
            xgb_model.feature_importances_,
            index=FEATURE_COLUMNS,
        )
        .sort_values(
            ascending=False
        )
    )

    print(importance)

    # ========================================================
    # Save model
    # ========================================================

    xgb_model.save_model(
        MODEL_FILE
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Path: {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()