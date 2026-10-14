import pandas as pd
from typing import Dict, Any


class SelectivePredictionEngine:
    """
    Evaluates different minimum confidence thresholds to identify when to actually trade.
    """
    def __init__(self, backtester):
        self.backtester = backtester

    def evaluate_thresholds(
        self, 
        long_threshold: float, 
        short_threshold: float, 
        confidence_levels: list[float]
    ) -> pd.DataFrame:
        
        results = []
        for conf in confidence_levels:
            res_df = self.backtester.run(
                long_threshold=long_threshold, 
                short_threshold=short_threshold, 
                confidence_min=conf
            )
            metrics = self.backtester.get_metrics(res_df)
            metrics["confidence_threshold"] = conf
            results.append(metrics)
            
        return pd.DataFrame(results)

