from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_FILE = Path(
    "data/processed/xgboost_v1_test_predictions.csv"
)

INITIAL_CAPITAL = 10_000.0

# Assumptions for this experiment
FEE_RATE = 0.0010
SLIPPAGE_RATE = 0.0005

COST_PER_SIDE = FEE_RATE + SLIPPAGE_RATE

THRESHOLDS = [
    (0.55, 0.45),
    (0.60, 0.40),
    (0.65, 0.35),
    (0.70, 0.30),
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    return df


def generate_signal(
    probability_up: float,
    long_threshold: float,
    short_threshold: float,
) -> int:

    if probability_up >= long_threshold:
        return 1

    if probability_up <= short_threshold:
        return -1

    return 0


def backtest(
    df: pd.DataFrame,
    long_threshold: float,
    short_threshold: float,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Prediction at time T
    # --------------------------------------------------------

    result["signal"] = result[
        "probability_up"
    ].apply(
        lambda p: generate_signal(
            p,
            long_threshold,
            short_threshold,
        )
    )

    # IMPORTANT:
    # The prediction at T is applied to the movement
    # represented by future_1h_return at T.
    result["position"] = (
        result["signal"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Position changes
    # --------------------------------------------------------

    result["previous_position"] = (
        result["position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    result["position_change"] = (
        result["position"]
        - result["previous_position"]
    )

    result["trade_sides"] = (
        result["position_change"]
        .abs()
    )

    # --------------------------------------------------------
    # Strategy return
    # --------------------------------------------------------

    result["gross_return"] = (
        result["position"]
        * result["future_1h_return"]
    )

    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------

    result["trading_cost"] = (
        result["trade_sides"]
        * COST_PER_SIDE
    )

    result["net_return"] = (
        result["gross_return"]
        - result["trading_cost"]
    )

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    result["equity"] = (
        INITIAL_CAPITAL
        * (1 + result["net_return"])
        .cumprod()
    )

    return result


def metrics(
    result: pd.DataFrame,
) -> dict:

    active = result[
        result["position"] != 0
    ]

    total_return = (
        result["equity"].iloc[-1]
        / INITIAL_CAPITAL
        - 1
    )

    running_max = (
        result["equity"].cummax()
    )

    drawdown = (
        result["equity"]
        / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    if len(active) > 0:

        win_rate = (
            active["net_return"] > 0
        ).mean()

        avg_return = (
            active["net_return"].mean()
        )

    else:

        win_rate = np.nan
        avg_return = np.nan

    profits = active.loc[
        active["net_return"] > 0,
        "net_return",
    ].sum()

    losses = active.loc[
        active["net_return"] < 0,
        "net_return",
    ].sum()

    profit_factor = (
        profits / abs(losses)
        if losses != 0
        else np.inf
    )

    if result["net_return"].std() > 0:

        sharpe = (
            result["net_return"].mean()
            / result["net_return"].std()
            * np.sqrt(24 * 365)
        )

    else:
        sharpe = np.nan

    entries = (
        (
            (result["position"] != 0)
            & (result["previous_position"] == 0)
        )
    ).sum()

    exits = (
        (
            (result["position"] == 0)
            & (result["previous_position"] != 0)
        )
    ).sum()

    reversals = (
        (
            result["position"]
            * result["previous_position"]
            == -1
        )
    ).sum()

    return {
        "active_periods": len(active),
        "entries": int(entries),
        "exits": int(exits),
        "reversals": int(reversals),
        "win_rate": win_rate,
        "total_return": total_return,
        "final_equity": result["equity"].iloc[-1],
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "avg_return": avg_return,
        "sharpe": sharpe,
        "exposure": (
            len(active) / len(result)
        ),
    }


def buy_and_hold(df: pd.DataFrame) -> dict:

    equity = (
        INITIAL_CAPITAL
        * (1 + df["future_1h_return"])
        .cumprod()
    )

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1
    )

    return {
        "return": (
            equity.iloc[-1]
            / INITIAL_CAPITAL
            - 1
        ),
        "final_equity": equity.iloc[-1],
        "max_drawdown": drawdown.min(),
    }


def main() -> None:

    print("=" * 70)
    print("BTC XGBOOST V3 BACKTEST")
    print("=" * 70)

    print(
        f"\nInitial capital : "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Trading cost/side: "
        f"{COST_PER_SIDE * 100:.3f}%"
    )

    df = load_data()

    print(
        f"\nRows: {len(df):,}"
    )

    results = []

    for long_threshold, short_threshold in THRESHOLDS:

        result = backtest(
            df,
            long_threshold,
            short_threshold,
        )

        m = metrics(result)

        results.append(
            {
                "long": long_threshold,
                "short": short_threshold,
                **m,
            }
        )

    results_df = pd.DataFrame(results)

    display = results_df.copy()

    display["win_rate"] *= 100
    display["total_return"] *= 100
    display["max_drawdown"] *= 100
    display["avg_return"] *= 100
    display["exposure"] *= 100

    print("\n" + "=" * 70)
    print("STRATEGY RESULTS")
    print("=" * 70)

    print(
        display.to_string(
            index=False,
            formatters={
                "long": "{:.2f}".format,
                "short": "{:.2f}".format,
                "active_periods": "{:.0f}".format,
                "entries": "{:.0f}".format,
                "exits": "{:.0f}".format,
                "reversals": "{:.0f}".format,
                "win_rate": "{:.2f}%".format,
                "total_return": "{:.2f}%".format,
                "final_equity": "${:,.2f}".format,
                "max_drawdown": "{:.2f}%".format,
                "profit_factor": (
                    lambda x:
                    f"{x:.3f}"
                    if np.isfinite(x)
                    else "inf"
                ),
                "avg_return": "{:.4f}%".format,
                "sharpe": "{:.3f}".format,
                "exposure": "{:.2f}%".format,
            },
        )
    )

    # --------------------------------------------------------
    # Buy & Hold
    # --------------------------------------------------------

    bh = buy_and_hold(df)

    print("\n" + "=" * 70)
    print("BUY & HOLD")
    print("=" * 70)

    print(
        f"Return       : "
        f"{bh['return'] * 100:.2f}%"
    )

    print(
        f"Final equity : "
        f"${bh['final_equity']:,.2f}"
    )

    print(
        f"Max drawdown : "
        f"{bh['max_drawdown'] * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("V3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()