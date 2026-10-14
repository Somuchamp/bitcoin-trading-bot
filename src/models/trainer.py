import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config.settings import settings
from src.features.pipeline import get_feature_columns
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self, 
        data_path: str, 
        model_dir: str = settings.data.models_dir, 
        version: str = settings.model.version
    ):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator()
        
        self.feature_columns = get_feature_columns()

    def train_and_evaluate(self):
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.sort_values("open_time").reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * (1 - settings.model.test_size - settings.model.validation_size))
        validation_end = int(n * (1 - settings.model.test_size))

        train_df = df.iloc[:train_end].copy()
        validation_df = df.iloc[train_end:validation_end].copy()
        test_df = df.iloc[validation_end:].copy()
        
        X_train, y_train = train_df[self.feature_columns], train_df["target"]
        X_validation, y_validation = validation_df[self.feature_columns], validation_df["target"]
        X_test, y_test = test_df[self.feature_columns], test_df["target"]
        
        logger.info(f"Training XGBoost {self.version}...")
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            random_state=settings.model.random_seed,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )

        logger.info("Evaluating on Validation Set...")
        val_probs = model.predict_proba(X_validation)
        val_metrics = self.evaluator.evaluate(y_validation, val_probs, prefix="val")
        
        logger.info("Evaluating on Test Set...")
        test_probs = model.predict_proba(X_test)
        test_metrics = self.evaluator.evaluate(y_test, test_probs, prefix="test")
        
        # Save Model Artifacts
        model_file = self.model_dir / f"{self.version}.json"
        model.save_model(model_file)
        
        # Generate Test Predictions CSV
        test_df["p_down"] = test_probs[:, 0]
        test_df["p_neutral"] = test_probs[:, 1]
        test_df["p_up"] = test_probs[:, 2]
        test_df["prediction"] = test_probs.argmax(axis=1)
        test_df["prediction_label"] = test_df["prediction"].map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})
        
        # Rename future_return column to match expected schema
        if "future_return" in test_df.columns:
            test_df = test_df.rename(columns={"future_return": "future_1h_return"})

        test_preds_file = Path(settings.data.processed_dir) / f"{self.version}_test_predictions.csv"
        test_df.to_csv(test_preds_file, index=False)
        
        # Save feature importance
        importance = dict(zip(self.feature_columns, map(float, model.feature_importances_)))
        
        # Register Model
        metadata = {
            "training_time": datetime.now(timezone.utc).isoformat(),
            "train_start": train_df["open_time"].min().isoformat(),
            "train_end": train_df["open_time"].max().isoformat(),
            "test_start": test_df["open_time"].min().isoformat(),
            "test_end": test_df["open_time"].max().isoformat(),
            "samples": len(df),
            "features": self.feature_columns,
            "feature_importance": importance,
            "metrics": {**val_metrics, **test_metrics}
        }
        
        self.registry.register_model(
            version=self.version,
            model_type="xgboost",
            metadata=metadata,
            status="candidate"
        )
        
        self.evaluator.generate_report(y_test, test_probs, self.version)
        logger.info(f"Model {self.version} training and evaluation complete.")
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    in_path = Path(settings.data.processed_dir) / "btc_usdt_1h_features_v3_new.csv"
    if not in_path.exists():
        logger.error(f"v3 features not found at {in_path}. Please run: python -m src.features.pipeline")
    else:
        trainer = ModelTrainer(str(in_path), version="xgboost_v3")
        trainer.train_and_evaluate()
