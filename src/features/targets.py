import numpy as np
import pandas as pd

from src.config.settings import settings


class TargetBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def add_targets(self) -> pd.DataFrame:
        """
        Creates the future return and configurable target classes.
        Assumes data is sorted chronologically and within continuous segments.
        """
        # Target must be calculated within segments to avoid look-ahead across gaps
        if "segment_id" not in self.df.columns:
            # We assume it's one continuous segment if no segment_id
            self.df["segment_id"] = 0
            
        def _build_target_for_segment(segment: pd.DataFrame) -> pd.DataFrame:
            segment = segment.copy()
            close = segment["close"]
            
           
            segment["future_return"] = close.shift(-1) / close - 1
            
           
            threshold = settings.target.threshold
            
            segment["target"] = np.select(
                [
                    segment["future_return"] < -threshold,
                    segment["future_return"] > threshold,
                ],
                [
                    0,
                    2,  
                ],
                default=1,  
            )
            return segment

        original_segment_id = self.df["segment_id"].copy()
        self.df = (
            self.df.groupby("segment_id", group_keys=False)
            .apply(_build_target_for_segment, include_groups=False)
        )
        self.df["segment_id"] = original_segment_id
        self.df = self.df.reset_index(drop=True)
        
        # Target needs to be integer
        self.df["target"] = self.df["target"].astype(pd.Int64Dtype())
        
        return self.df
