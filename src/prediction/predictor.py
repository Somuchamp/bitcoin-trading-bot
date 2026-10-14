import logging
import pandas as pd
from xgboost import XGBClassifier
from pathlib import Path
from datetime import datetime, timezone

from src.config.settings import settings
from src.features.builder import FeatureBuilder
from src.features.pipeline import get_feature_columns

logger = logging.getLogger(__name__)


class LivePredictor:
    def __init__(self):
        self.model = XGBClassifier()
        model_path = Path(settings.data.models_dir) / f"{settings.model.version}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model.load_model(model_path)
        
        # Dynamically load the exact feature names the model was trained on
        booster_features = self.model.get_booster().feature_names
        self.feature_columns = booster_features if booster_features else get_feature_columns()
        
        self.TARGET_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the recent history dataframe (at least 50 candles required for MA50) 
        and calculates features for the last row.
        """
        if len(df) < 50:
            raise ValueError("Not enough historical data to compute features (need at least 50 candles for SMA50).")
            
        builder = FeatureBuilder(df)
        builder.add_segment_id()  # In live, we can assume continuous if fetched fresh
        features_df = builder._build_features_for_segment(builder.df)
        return features_df.iloc[[-1]]  # Return only the latest row

    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Predicts the market direction for the given features.
        """
        X = features_df[self.feature_columns]
        probabilities = self.model.predict_proba(X)[0]
        
        predicted_class_idx = probabilities.argmax()
        confidence = probabilities[predicted_class_idx]
        
        return {
            "probability_down": float(probabilities[0]),
            "probability_neutral": float(probabilities[1]),
            "probability_up": float(probabilities[2]),
            "predicted_class": self.TARGET_NAMES[predicted_class_idx],
            "confidence": float(confidence),
            "model_version": settings.model.version,
        }
