from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_FILE = Path(
    "data/processed/xgboost_v1_test_predictions.csv"
)

INITIAL_CAPITAL = 10_000.0

# Example assumptions.
# We will test sensitivity later.
FEE_RATE = 0.001          # 0.10% per executed side
SLIPPAGE_RATE = 0.0005   # 0.05% per executed side


def run_strategy(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    df = df.copy()

    # ---------------------------------------------------------
    # Signal
    # ---------------------------------------------------------

    df["position"] = 0

    df.loc[
        df["probability_up"] >= threshold,
        "position",
    ] = 1

    df.loc[
        df["probability_up"] <= (1 - threshold),
        "position",
    ] = -1

    # ---------------------------------------------------------
    # Price return from current close to next hour close
    # ---------------------------------------------------------

    df["gross_return"] = (
        df["future_1h_return"]
        * df["position"]
    )

    # ---------------------------------------------------------
    # Trading cost
    #
    # Charge costs whenever position is non-zero.
    # This is intentionally conservative for this first test.
    # ---------------------------------------------------------

    df["cost"] = np.where(
        df["position"] != 0,
        FEE_RATE + SLIPPAGE_RATE,
        0.0,
    )

    df["net_return"] = (
        df["gross_return"] - df["cost"]
    )

    # ---------------------------------------------------------
    # Equity
    # ---------------------------------------------------------

    df["equity"] = (
        INITIAL_CAPITAL
        * (1 + df["net_return"]).cumprod()
    )

    # ---------------------------------------------------------
    # Buy and hold
    # ---------------------------------------------------------

    df["buy_hold_equity"] = (
        INITIAL_CAPITAL
        * (1 + df["future_1h_return"]).cumprod()
    )

    return df


def calculate_metrics(
    df: pd.DataFrame,
) -> dict:

    trades = df[
        df["position"] != 0
    ]

    if len(trades) == 0:

        return {
            "trades": 0,
            "win_rate": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": np.nan,
            "avg_trade_return": np.nan,
        }

    winning_trades = (
        trades["net_return"] > 0
    ).sum()

    win_rate = (
        winning_trades
        / len(trades)
    )

    total_return = (
        df["equity"].iloc[-1]
        / INITIAL_CAPITAL
        - 1
    )

    running_max = (
        df["equity"].cummax()
    )

    drawdown = (
        df["equity"] / running_max - 1
    )

    max_drawdown = drawdown.min()

    profits = trades.loc[
        trades["net_return"] > 0,
        "net_return",
    ].sum()

    losses = trades.loc[
        trades["net_return"] < 0,
        "net_return",
    ].sum()

    if losses != 0:
        profit_factor = profits / abs(losses)
    else:
        profit_factor = np.inf

    avg_trade_return = (
        trades["net_return"].mean()
    )

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
    }


def main() -> None:

    print("=" * 70)
    print("XGBOOST V1 BACKTEST")
    print("=" * 70)

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    print(
        f"\nPrediction rows: {len(df):,}"
    )

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    results = []

    for threshold in thresholds:

        strategy = run_strategy(
            df,
            threshold,
        )

        metrics = calculate_metrics(
            strategy
        )

        results.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # Convert ratios to percentages for display.
    results_df["win_rate"] *= 100
    results_df["total_return"] *= 100
    results_df["max_drawdown"] *= 100
    results_df["avg_trade_return"] *= 100

    print("\n" + "=" * 70)
    print("STRATEGY RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "trades": "{:.0f}".format,
                "win_rate": "{:.2f}%".format,
                "total_return": "{:.2f}%".format,
                "max_drawdown": "{:.2f}%".format,
                "profit_factor": (
                    lambda x:
                    f"{x:.3f}"
                    if np.isfinite(x)
                    else "inf"
                ),
                "avg_trade_return": (
                    lambda x:
                    f"{x:.4f}%"
                ),
            },
        )
    )

    # ---------------------------------------------------------
    # Buy & hold
    # ---------------------------------------------------------

    buy_hold_return = (
        (1 + df["future_1h_return"]).prod()
        - 1
    )

    print("\n" + "=" * 70)
    print("BUY & HOLD")
    print("=" * 70)

    print(
        f"Buy & Hold return: "
        f"{buy_hold_return * 100:.2f}%"
    )

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final value: "
        f"${INITIAL_CAPITAL * (1 + buy_hold_return):,.2f}"
    )

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()