import logging
import numpy as np
import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)


class ExecutionAwareBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = settings.backtest.initial_capital,
        fee_rate: float = settings.backtest.fee_rate,
        slippage_rate: float = settings.backtest.slippage_rate,
    ):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.cost_per_side = fee_rate + slippage_rate
        
        # Make sure df is sorted
        self.df["open_time"] = pd.to_datetime(self.df["open_time"], utc=True)
        self.df = self.df.sort_values("open_time").reset_index(drop=True)

    def generate_signal(self, prob_up: float, prob_down: float, long_threshold: float, short_threshold: float, confidence_min: float = 0.0) -> int:
        if prob_up >= long_threshold and prob_up >= confidence_min:
            return 1
        if prob_down >= short_threshold and prob_down >= confidence_min:
            return -1
        return 0

    def run(self, long_threshold: float, short_threshold: float, confidence_min: float = 0.0) -> pd.DataFrame:
        result = self.df.copy()

        # The model predictions are available at the close of the current candle.
        # We generate a signal based on these predictions.
        result["signal"] = result.apply(
            lambda row: self.generate_signal(
                row.get("probability_up", 0), 
                row.get("probability_down", 0), 
                long_threshold, 
                short_threshold,
                confidence_min
            ),
            axis=1
        )
        
        # Position is adopted for the next hour based on the signal
        result["position"] = result["signal"].astype(int)

        # Position changes and trades
        result["previous_position"] = result["position"].shift(1).fillna(0).astype(int)
        result["position_change"] = result["position"] - result["previous_position"]
        result["trade_sides"] = result["position_change"].abs()

        # Gross return on the future candle
        if "future_return" not in result.columns:
            result["future_return"] = result["close"].shift(-1) / result["close"] - 1
            
        result["gross_return"] = result["position"] * result["future_return"]

        # Cost applies on the candle where the position change is executed
        # Meaning, if we decide to change position at the end of T, the cost is incurred.
        result["trading_cost"] = result["trade_sides"] * self.cost_per_side
        result["net_return"] = result["gross_return"] - result["trading_cost"]

        # Equity
        result["equity"] = self.initial_capital * (1 + result["net_return"]).cumprod()

        return result

    def get_metrics(self, result: pd.DataFrame) -> dict:
        active = result[result["position"] != 0]
        total_return = (result["equity"].iloc[-1] / self.initial_capital - 1) if not result.empty else 0.0
        
        running_max = result["equity"].cummax()
        drawdown = result["equity"] / running_max - 1
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        if len(active) > 0:
            win_rate = (active["net_return"] > 0).mean()
            avg_return = active["net_return"].mean()
        else:
            win_rate = 0.0
            avg_return = 0.0

        profits = active.loc[active["net_return"] > 0, "net_return"].sum()
        losses = active.loc[active["net_return"] < 0, "net_return"].sum()
        profit_factor = (profits / abs(losses)) if losses != 0 else float("inf")

        std = result["net_return"].std()
        sharpe = (result["net_return"].mean() / std * np.sqrt(24 * 365)) if std > 0 else 0.0

        entries = ((result["position"] != 0) & (result["previous_position"] == 0)).sum()
        exits = ((result["position"] == 0) & (result["previous_position"] != 0)).sum()
        reversals = (result["position"] * result["previous_position"] == -1).sum()

        bh_equity = self.initial_capital * (1 + result["future_return"]).cumprod()
        bh_return = (bh_equity.iloc[-1] / self.initial_capital - 1) if not result.empty else 0.0

        return {
            "active_periods": len(active),
            "entries": int(entries),
            "exits": int(exits),
            "reversals": int(reversals),
            "win_rate": float(win_rate),
            "total_return": float(total_return),
            "final_equity": float(result["equity"].iloc[-1] if not result.empty else self.initial_capital),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(profit_factor),
            "avg_return": float(avg_return),
            "sharpe": float(sharpe),
            "exposure": float(len(active) / len(result)) if len(result) > 0 else 0.0,
            "bh_return": float(bh_return),
        }


if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    preds_file = Path(settings.data.processed_dir) / f"{settings.model.version}_test_predictions.csv"
    if preds_file.exists():
        df = pd.read_csv(preds_file)
        backtester = ExecutionAwareBacktester(df)
        res = backtester.run(settings.backtest.long_threshold, settings.backtest.short_threshold)
        metrics = backtester.get_metrics(res)
        
        logger.info(f"Backtest Metrics: {metrics}")
    else:
        logger.error(f"No predictions found at {preds_file}")
