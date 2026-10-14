import pandas as pd
import pytest
from pathlib import Path

from src.config.settings import settings
from src.features.builder import FeatureBuilder
from src.features.targets import TargetBuilder


@pytest.fixture
def mock_data():
    # Create a dummy dataframe
    dates = pd.date_range("2021-01-01", periods=100, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open_time": dates,
        "open": range(100, 200),
        "high": range(105, 205),
        "low": range(95, 195),
        "close": range(102, 202),
        "volume": [1000] * 100,
        "close_time": dates + pd.Timedelta(minutes=59, seconds=59),
        "number_of_trades": [10] * 100
    })
    
    # Introduce a gap
    df.loc[50, "open_time"] += pd.Timedelta(hours=2)
    
    return df

def test_no_future_leakage_in_features(mock_data):
    builder = FeatureBuilder(mock_data)
    features_df = builder.build_features()
    
    # Assert that rolling means for row N only depend on rows <= N
    # We can check the gap boundary. The segment_id should break it.
    
    assert "segment_id" in features_df.columns
    # Row 50 should start a new segment
    assert features_df.loc[50, "segment_id"] > features_df.loc[49, "segment_id"]
    
    # The rolling feature for row 50 (e.g. sma_10) should be NaN because min_periods=10 and it's a new segment
    assert pd.isna(features_df.loc[50, "price_vs_sma_10"])

def test_target_leakage(mock_data):
    # Verify that target at T only uses close at T+1
    builder = FeatureBuilder(mock_data)
    df = builder.build_features()
    
    target_builder = TargetBuilder(df)
    target_df = target_builder.add_targets()
    
    # At row 49 (end of segment 0), future_return should be NaN because row 50 is in a new segment.
    assert pd.isna(target_df.loc[49, "future_return"])
    assert pd.isna(target_df.loc[99, "future_return"])  # Last row
