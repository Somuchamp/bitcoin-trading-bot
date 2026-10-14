import numpy as np
import pandas as pd

from src.config.settings import settings


class FeatureBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
        previous_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def add_segment_id(self) -> None:
        # Segment by missing candles (gaps > 1h)
        time_diff = self.df["open_time"].diff()
        new_segment = time_diff != pd.Timedelta(hours=1)
        self.df["segment_id"] = new_segment.cumsum()

    def _build_features_for_segment(self, segment: pd.DataFrame) -> pd.DataFrame:
        segment = segment.copy()
        close = segment["close"]

        # 1. Price Returns
        for h in [1, 3, 6, 12, 24]:
            segment[f"return_{h}h"] = close.pct_change(h)

        # 2. Trend (Moving Averages)
        sma_10 = close.rolling(10, min_periods=10).mean()
        sma_20 = close.rolling(20, min_periods=20).mean()
        sma_50 = close.rolling(50, min_periods=50).mean()
        ema_10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
        ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()

        segment["price_vs_sma_10"] = close / sma_10 - 1
        segment["price_vs_sma_20"] = close / sma_20 - 1
        segment["price_vs_sma_50"] = close / sma_50 - 1
        segment["price_vs_ema_10"] = close / ema_10 - 1
        segment["price_vs_ema_20"] = close / ema_20 - 1
        segment["sma_10_vs_sma_50"] = sma_10 / sma_50 - 1
        segment["ema_10_vs_ema_20"] = ema_10 / ema_20 - 1

        # 3. Momentum
        segment["rsi"] = self.calculate_rsi(close, settings.features.rsi_period)

        ema_fast = close.ewm(span=settings.features.macd_fast, adjust=False, min_periods=settings.features.macd_fast).mean()
        ema_slow = close.ewm(span=settings.features.macd_slow, adjust=False, min_periods=settings.features.macd_slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=settings.features.macd_signal, adjust=False, min_periods=settings.features.macd_signal).mean()

        segment["macd_pct"] = macd / close
        segment["macd_signal_pct"] = macd_signal / close
        segment["macd_histogram_pct"] = (macd - macd_signal) / close

        # 4. Volatility
        segment["volatility"] = segment["return_1h"].rolling(settings.features.volatility_period, min_periods=settings.features.volatility_period).std()
        atr = self.calculate_atr(segment, settings.features.atr_period)
        segment["atr_pct"] = atr / close

        # 5. Volume
        volume_ma_24 = segment["volume"].rolling(24, min_periods=24).mean()
        segment["volume_ratio"] = segment["volume"] / volume_ma_24
        segment["volume_change"] = segment["volume"].pct_change()

        # 6. Candle Structure
        segment["high_low_range_pct"] = (segment["high"] - segment["low"]) / close
        segment["body_pct"] = (segment["close"] - segment["open"]) / segment["open"]
        segment["upper_wick_pct"] = (segment["high"] - segment[["open", "close"]].max(axis=1)) / close
        segment["lower_wick_pct"] = (segment[["open", "close"]].min(axis=1) - segment["low"]) / close

        # 7. Lagged Momentum Indicators (RSI, MACD from 1, 2, 3 hours ago)
        # These allow the model to see the DIRECTION and SPEED of momentum change
        for lag in [1, 2, 3]:
            segment[f"rsi_lag{lag}"] = segment["rsi"].shift(lag)
            segment[f"macd_pct_lag{lag}"] = segment["macd_pct"].shift(lag)
            segment[f"macd_hist_lag{lag}"] = segment["macd_histogram_pct"].shift(lag)

        # 8. Lagged Returns (what the price did 2, 4, 6, 12, 24 hours ago)
        # Captures short-term mean-reversion and trend-continuation patterns
        for lag in [2, 4, 6, 12, 24]:
            segment[f"return_lag{lag}h"] = segment["return_1h"].shift(lag)

        # 9. Time-of-Day Features (crypto has strong session-based patterns)
        if "open_time" in segment.columns:
            dt = pd.to_datetime(segment["open_time"], utc=True)
            segment["hour_of_day"] = dt.dt.hour
            segment["day_of_week"] = dt.dt.dayofweek
            segment["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            # Asia session: 00:00-08:00 UTC
            segment["is_asia_session"] = ((dt.dt.hour >= 0) & (dt.dt.hour < 8)).astype(int)
            # London session: 08:00-16:00 UTC
            segment["is_london_session"] = ((dt.dt.hour >= 8) & (dt.dt.hour < 16)).astype(int)
            # US session: 13:00-21:00 UTC
            segment["is_us_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 21)).astype(int)

        return segment

    def build_features(self) -> pd.DataFrame:
        self.add_segment_id()
        original_segment_id = self.df["segment_id"].copy()
        self.df = (
            self.df.groupby("segment_id", group_keys=False)
            .apply(self._build_features_for_segment, include_groups=False)
        )
        self.df["segment_id"] = original_segment_id
        self.df = self.df.reset_index(drop=True)
        return self.df
