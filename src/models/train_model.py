from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


DATA_FILE = Path(
    "data/processed/btc_usdt_1h_features.csv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

TARGET_COLUMN = "target"


def print_metrics(
    name: str,
    y_true: pd.Series,
    probabilities,
) -> None:

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{accuracy_score(y_true, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1       : "
        f"{f1_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"ROC AUC  : "
        f"{roc_auc_score(y_true, probabilities):.4f}"
    )

    print(
        f"Log Loss : "
        f"{log_loss(y_true, probabilities):.4f}"
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "DOWN",
                "UP",
            ],
            zero_division=0,
        )
    )


def main() -> None:

    print("=" * 70)
    print("BTC MODEL V1 TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    df = df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    print(f"\nTotal samples: {len(df):,}")

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # X / y
    # ---------------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    majority_class = int(
        y_train.mean() >= 0.5
    )

    baseline_probability = (
        float(y_train.mean())
        if majority_class == 1
        else 1 - float(y_train.mean())
    )

    baseline_predictions = pd.Series(
        majority_class,
        index=y_test.index,
    )

    print()
    print("=" * 70)
    print("BASELINE")
    print("=" * 70)

    print(
        f"Training UP probability: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Majority class: "
        f"{'UP' if majority_class == 1 else 'DOWN'}"
    )

    print(
        f"Test accuracy: "
        f"{accuracy_score(y_test, baseline_predictions):.4f}"
    )

    # ---------------------------------------------------------
    # Logistic regression
    # ---------------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    validation_probability_lr = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    print_metrics(
        "LOGISTIC REGRESSION — VALIDATION",
        y_validation,
        validation_probability_lr,
    )

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------

    print("\nTraining XGBoost...")

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
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

    validation_probability_xgb = (
        xgb_model.predict_proba(
            X_validation
        )[:, 1]
    )

    print_metrics(
        "XGBOOST — VALIDATION",
        y_validation,
        validation_probability_xgb,
    )

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_probability_xgb = (
        xgb_model.predict_proba(
            X_test
        )[:, 1]
    )

    print_metrics(
        "XGBOOST — TEST",
        y_test,
        test_probability_xgb,
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.Series(
        xgb_model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    print(importance)

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model_path = (
        MODEL_DIR / "xgboost_v1.json"
    )

    xgb_model.save_model(
        model_path
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(f"Path: {model_path}")


if __name__ == "__main__":
    main()