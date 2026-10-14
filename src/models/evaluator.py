import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.config.settings import settings


class ModelEvaluator:
    TARGET_NAMES = {
        0: "DOWN",
        1: "NEUTRAL",
        2: "UP",
    }

    def __init__(self, output_dir: str = settings.data.reports_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true: pd.Series, probabilities: np.ndarray, prefix: str = "test") -> Dict[str, Any]:
        predictions = probabilities.argmax(axis=1)

        metrics = {
            f"{prefix}_accuracy": float(accuracy_score(y_true, predictions)),
            f"{prefix}_macro_f1": float(f1_score(y_true, predictions, average='macro')),
            f"{prefix}_weighted_f1": float(f1_score(y_true, predictions, average='weighted')),
            f"{prefix}_log_loss": float(log_loss(y_true, probabilities)),
        }

        # Confusion Matrix
        cm = confusion_matrix(y_true, predictions, labels=[0, 1, 2])
        metrics[f"{prefix}_confusion_matrix"] = cm.tolist()

        return metrics

    def generate_report(self, y_true: pd.Series, probabilities: np.ndarray, model_version: str, report_name: str = "evaluation.json") -> None:
        metrics = self.evaluate(y_true, probabilities, prefix="test")
        
        report_path = self.output_dir / f"{model_version}_{report_name}"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)
