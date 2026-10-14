import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic_settings import BaseSettings


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Load YAML first, so env variables can override them if necessary via Pydantic
yaml_config = load_yaml_config("configs/config.yaml")


class MarketSettings(BaseSettings):
    symbol: str = yaml_config.get("market", {}).get("symbol", "BTCUSDT")
    timeframe: str = yaml_config.get("market", {}).get("timeframe", "1h")
    exchange: str = yaml_config.get("market", {}).get("exchange", "binance")


class DataSettings(BaseSettings):
    raw_dir: str = yaml_config.get("data", {}).get("raw_dir", "data/raw")
    processed_dir: str = yaml_config.get("data", {}).get("processed_dir", "data/processed")
    models_dir: str = yaml_config.get("data", {}).get("models_dir", "models")
    reports_dir: str = yaml_config.get("data", {}).get("reports_dir", "reports")


class FeatureSettings(BaseSettings):
    version: str = yaml_config.get("features", {}).get("version", "v2")
    rsi_period: int = yaml_config.get("features", {}).get("rsi_period", 14)
    atr_period: int = yaml_config.get("features", {}).get("atr_period", 14)
    macd_fast: int = yaml_config.get("features", {}).get("macd_fast", 12)
    macd_slow: int = yaml_config.get("features", {}).get("macd_slow", 26)
    macd_signal: int = yaml_config.get("features", {}).get("macd_signal", 9)
    volatility_period: int = yaml_config.get("features", {}).get("volatility_period", 24)


class TargetSettings(BaseSettings):
    threshold: float = yaml_config.get("target", {}).get("threshold", 0.0015)
    horizon: str = yaml_config.get("target", {}).get("horizon", "1h")


class ModelSettings(BaseSettings):
    version: str = yaml_config.get("model", {}).get("version", "v2")
    random_seed: int = yaml_config.get("model", {}).get("random_seed", 42)
    test_size: float = yaml_config.get("model", {}).get("test_size", 0.15)
    validation_size: float = yaml_config.get("model", {}).get("validation_size", 0.15)


class BacktestSettings(BaseSettings):
    initial_capital: float = yaml_config.get("backtest", {}).get("initial_capital", 10000.0)
    fee_rate: float = yaml_config.get("backtest", {}).get("fee_rate", 0.001)
    slippage_rate: float = yaml_config.get("backtest", {}).get("slippage_rate", 0.0005)
    long_threshold: float = yaml_config.get("backtest", {}).get("long_threshold", 0.60)
    short_threshold: float = yaml_config.get("backtest", {}).get("short_threshold", 0.40)


class Settings(BaseSettings):
    database_url: str = os.getenv(
        "DATABASE_URL",
        yaml_config.get("data", {}).get("db_path", "sqlite:///./data/bitcoin_platform.db"),
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    market: MarketSettings = MarketSettings()
    data: DataSettings = DataSettings()
    features: FeatureSettings = FeatureSettings()
    target: TargetSettings = TargetSettings()
    model: ModelSettings = ModelSettings()
    backtest: BacktestSettings = BacktestSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
Path(settings.data.raw_dir).mkdir(parents=True, exist_ok=True)
Path(settings.data.processed_dir).mkdir(parents=True, exist_ok=True)
Path(settings.data.models_dir).mkdir(parents=True, exist_ok=True)
Path(settings.data.reports_dir).mkdir(parents=True, exist_ok=True)
