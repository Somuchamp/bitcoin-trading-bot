import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, df: pd.DataFrame, interval: str = settings.market.timeframe):
        self.df = df
        self.interval = interval
        self.report: Dict[str, Any] = {}

    def validate(self) -> Dict[str, Any]:
        logger.info("Running Data Validation Suite...")
        
        df = self.df.copy()
        
        if df.empty:
            return {"status": "error", "message": "DataFrame is empty"}

        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="mixed")
        
        self.report["total_rows"] = len(df)
        self.report["start_time"] = df["open_time"].min().isoformat()
        self.report["end_time"] = df["open_time"].max().isoformat()
        
        # 1. Missing values
        missing = df.isna().sum().to_dict()
        self.report["missing_values"] = {k: int(v) for k, v in missing.items() if v > 0}
        
        # 2. Duplicate timestamps
        duplicates = df.duplicated(subset=["open_time"]).sum()
        self.report["duplicate_timestamps"] = int(duplicates)
        
        # 3. Chronological ordering
        is_sorted = df["open_time"].is_monotonic_increasing
        self.report["is_chronological"] = bool(is_sorted)
        
        # 4. Gaps and missing candles
        # Assuming 1h interval for gap calculation
        if self.interval == "1h":
            expected_diff = pd.Timedelta(hours=1)
            time_diffs = df["open_time"].diff()
            gaps = time_diffs[time_diffs > expected_diff]
            self.report["timestamp_gaps"] = int(len(gaps))
            
            missing_candles = ((time_diffs - expected_diff) / expected_diff).sum()
            self.report["missing_candles"] = int(missing_candles) if pd.notna(missing_candles) else 0
        
        # 5. OHLC consistency (high >= low, high >= open, high >= close, etc)
        invalid_ohlc = ((df["high"] < df["low"]) | 
                        (df["high"] < df["open"]) | 
                        (df["high"] < df["close"]) |
                        (df["low"] > df["open"]) |
                        (df["low"] > df["close"])).sum()
        self.report["invalid_ohlc_rows"] = int(invalid_ohlc)
        
        # 6. Non-positive prices
        non_positive_prices = ((df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)).sum()
        self.report["non_positive_prices"] = int(non_positive_prices)
        
        # 7. Negative volume
        negative_volume = (df["volume"] < 0).sum()
        self.report["negative_volume"] = int(negative_volume)
        
        # 8. Zero volume placeholders
        zero_volume = (df["volume"] == 0).sum()
        self.report["zero_volume"] = int(zero_volume)
        
        status = "failed" if (
            duplicates > 0 or 
            not is_sorted or 
            invalid_ohlc > 0 or 
            non_positive_prices > 0 or 
            negative_volume > 0
        ) else "passed"
        
        self.report["status"] = status
        
        logger.info(f"Validation completed. Status: {status}")
        return self.report

    def save_report(self, filepath: str = f"{settings.data.reports_dir}/data_quality_report.json"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=4)
        logger.info(f"Report saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Use the existing raw file (btc_usdt naming convention)
    input_file = Path(settings.data.raw_dir) / "btc_usdt_1h.csv"
    if not input_file.exists():
        # Fallback to processed clean file
        input_file = Path(settings.data.processed_dir) / "btc_usdt_1h_clean.csv"
    if input_file.exists():
        df = pd.read_csv(input_file)
        validator = DataValidator(df)
        report = validator.validate()
        validator.save_report()
        print(f"Validation Status: {report['status']}")
        print(f"Total rows: {report['total_rows']:,}")
        print(f"Gaps: {report.get('timestamp_gaps', 'N/A')}")
        print(f"Zero-volume rows: {report.get('zero_volume', 'N/A')}")
    else:
        logger.error(f"Input file not found: {input_file}")
