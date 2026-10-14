"""
Optuna Hyperparameter Tuner for XGBoost v4
-------------------------------------------
Runs N trials to find the best hyperparameters, then trains
a final model with:
  - Best Optuna hyperparameters
  - Class imbalance sample weights (forces model to stop ignoring DOWN/UP)
  - Saves model as xgboost_v4
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config.settings import settings
from src.features.pipeline import get_feature_columns
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Keep output clean


def load_data():
    data_path = Path(settings.data.processed_dir) / "btc_usdt_1h_features_v3_new.csv"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def split_data(df):
    n = len(df)
    train_end = int(n * (1 - settings.model.test_size - settings.model.validation_size))
    validation_end = int(n * (1 - settings.model.test_size))
    feature_columns = get_feature_columns()

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return (
        train_df[feature_columns], train_df["target"],
        val_df[feature_columns], val_df["target"],
        test_df[feature_columns], test_df["target"],
        test_df
    )


def objective(trial, X_train, y_train, X_val, y_val, sample_weights):
    """Optuna objective: minimise validation log-loss."""
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": settings.model.random_seed,
        "n_jobs": -1,
        "verbosity": 0,
        # --- Hyperparameters being tuned ---
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "early_stopping_rounds": 30,
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_probs = model.predict_proba(X_val)
    return log_loss(y_val, val_probs)


def train_final_model(best_params, X_train, y_train, X_val, y_val, sample_weights):
    """Train final model with best params + class weights."""
    logger.info(f"Training final model with best params: {best_params}")
    final_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": settings.model.random_seed,
        "n_jobs": -1,
        "early_stopping_rounds": 30,
        **best_params
    }
    model = XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def run_tuning(n_trials: int = 50):
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, test_df = split_data(df)
    feature_columns = get_feature_columns()

    # Class Imbalance Weights
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    class_counts = y_train.value_counts().rename({0: "DOWN", 1: "NEUTRAL", 2: "UP"})
    logger.info(f"Class distribution in training set:\n{class_counts}")
    logger.info("Sample weights applied: balanced")

    # Optuna Study
    logger.info(f"Starting Optuna study with {n_trials} trials...")
    study = optuna.create_study(direction="minimize", study_name="xgboost_v4_tuning")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, sample_weights),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best val log-loss: {study.best_value:.5f}")
    logger.info(f"Best params: {study.best_params}")

    # Final Training
    model = train_final_model(study.best_params, X_train, y_train, X_val, y_val, sample_weights)

    # Evaluate
    evaluator = ModelEvaluator()
    test_probs = model.predict_proba(X_test)
    test_metrics = evaluator.evaluate(y_test, test_probs, prefix="test")
    test_preds = test_probs.argmax(axis=1)
    accuracy = accuracy_score(y_test, test_preds)
    logger.info(f"V4 Test Accuracy: {accuracy*100:.2f}%")

    # Save Model
    version = "xgboost_v4"
    model_dir = Path(settings.data.models_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / f"{version}.json")
    logger.info(f"Model saved to models/{version}.json")

    # Save Predictions CSV
    test_df = test_df.copy()
    test_df["p_down"] = test_probs[:, 0]
    test_df["p_neutral"] = test_probs[:, 1]
    test_df["p_up"] = test_probs[:, 2]
    test_df["prediction"] = test_preds
    test_df["prediction_label"] = test_df["prediction"].map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})
    if "future_return" in test_df.columns:
        test_df = test_df.rename(columns={"future_return": "future_1h_return"})
    preds_file = Path(settings.data.processed_dir) / f"{version}_test_predictions.csv"
    test_df.to_csv(preds_file, index=False)

    # Save Best Params
    params_file = model_dir / f"{version}_best_params.json"
    params_file.write_text(json.dumps(study.best_params, indent=2))
    logger.info(f"Best params saved to {params_file}")

    # Register Model
    importance = dict(zip(feature_columns, map(float, model.feature_importances_)))
    registry = ModelRegistry()
    registry.register_model(
        version=version,
        model_type="xgboost",
        metadata={
            "training_time": datetime.now(timezone.utc).isoformat(),
            "best_optuna_params": study.best_params,
            "best_val_logloss": study.best_value,
            "n_trials": n_trials,
            "class_weighting": "balanced",
            "features": feature_columns,
            "feature_importance": importance,
            "metrics": test_metrics,
        },
        status="candidate"
    )
    evaluator.generate_report(y_test, test_probs, version)

    logger.info("=" * 50)
    logger.info("TUNING COMPLETE")
    logger.info(f"  V4 Test Accuracy: {accuracy*100:.2f}%")
    logger.info("=" * 50)
    return accuracy, study.best_params


if __name__ == "__main__":
    run_tuning(n_trials=50)
