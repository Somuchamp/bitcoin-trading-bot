import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.config.settings import settings
from src.features.builder import FeatureBuilder
from src.features.targets import TargetBuilder

logger = logging.getLogger(__name__)


def get_feature_columns() -> List[str]:
    """Returns the ordered list of expected feature columns."""
    return [
        # --- Original Features ---
        "return_1h", "return_3h", "return_6h", "return_12h", "return_24h",
        "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50",
        "price_vs_ema_10", "price_vs_ema_20",
        "sma_10_vs_sma_50", "ema_10_vs_ema_20",
        "rsi", "macd_pct", "macd_signal_pct", "macd_histogram_pct",
        "volatility", "atr_pct",
        "volume_ratio", "volume_change",
        "high_low_range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
        # --- NEW: Lagged Momentum (RSI, MACD from 1-3 hours ago) ---
        "rsi_lag1", "rsi_lag2", "rsi_lag3",
        "macd_pct_lag1", "macd_pct_lag2", "macd_pct_lag3",
        "macd_hist_lag1", "macd_hist_lag2", "macd_hist_lag3",
        # --- NEW: Lagged Returns ---
        "return_lag2h", "return_lag4h", "return_lag6h", "return_lag12h", "return_lag24h",
        # --- NEW: Time-of-Day & Session Features ---
        "hour_of_day", "day_of_week", "is_weekend",
        "is_asia_session", "is_london_session", "is_us_session",
    ]


def run_feature_pipeline(input_file: Path, output_file: Path) -> pd.DataFrame:
    logger.info(f"Running Feature Pipeline on {input_file}")
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="mixed")
    df = df.sort_values("open_time").reset_index(drop=True)
    
    logger.info(f"Input rows: {len(df):,}")

    # Build features
    logger.info("Building features...")
    feature_builder = FeatureBuilder(df)
    df = feature_builder.build_features()
    
    # Build targets
    logger.info("Building targets...")
    target_builder = TargetBuilder(df)
    df = target_builder.add_targets()
    
    feature_columns = get_feature_columns()
    
    # Drop rows with NA in required columns (due to rolling windows)
    required_cols = feature_columns + ["future_return", "target"]
    before = len(df)
    df = df.dropna(subset=required_cols).copy()
    
    logger.info(f"Rows dropped due to NA (rolling windows or missing future): {before - len(df):,}")
    
    output_columns = [
        "open_time", "open", "high", "low", "close", "volume", 
        "future_return", "target"
    ] + feature_columns

    df = df[output_columns]
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Feature Pipeline complete. Saved {len(df):,} rows to {output_file}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    in_path = Path(settings.data.processed_dir) / "btc_usdt_1h_clean.csv"
    out_path = Path(settings.data.processed_dir) / f"btc_usdt_1h_features_v3_new.csv"
    run_feature_pipeline(in_path, out_path)
