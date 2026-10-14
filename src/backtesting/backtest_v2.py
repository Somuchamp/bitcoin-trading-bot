from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_FILE = Path(
    "data/processed/xgboost_v1_test_predictions.csv"
)

INITIAL_CAPITAL = 10_000.0

# Trading costs per position change.
# These are assumptions for the MVP and should be replaced
# with the actual exchange/broker fee + realistic slippage.
FEE_RATE = 0.0010          # 0.10%
SLIPPAGE_RATE = 0.0005     # 0.05%

TOTAL_COST_PER_SIDE = (
    FEE_RATE + SLIPPAGE_RATE
)

# Threshold pairs:
#
# 0.55 / 0.45:
#     LONG if P(UP) >= 0.55
#     SHORT if P(UP) <= 0.45
#
# 0.60 / 0.40
# 0.65 / 0.35
# 0.70 / 0.30
#
THRESHOLDS = [
    (0.55, 0.45),
    (0.60, 0.40),
    (0.65, 0.35),
    (0.70, 0.30),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_predictions() -> pd.DataFrame:
    """Load model predictions."""

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    required_columns = [
        "open_time",
        "close",
        "future_1h_return",
        "probability_up",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(
    probability_up: float,
    long_threshold: float,
    short_threshold: float,
) -> int:
    """
    Return desired position:

        +1 = LONG
         0 = FLAT
        -1 = SHORT
    """

    if probability_up >= long_threshold:
        return 1

    if probability_up <= short_threshold:
        return -1

    return 0


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    long_threshold: float,
    short_threshold: float,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Desired signal
    # --------------------------------------------------------

    result["desired_position"] = result[
        "probability_up"
    ].apply(
        lambda probability: generate_signal(
            probability,
            long_threshold,
            short_threshold,
        )
    )

    # --------------------------------------------------------
    # Position
    #
    # We use the desired signal for the NEXT hourly period.
    #
    # This avoids pretending that we can trade before the
    # prediction has been generated.
    # --------------------------------------------------------

    result["position"] = (
        result["desired_position"]
        .shift(1)
        .fillna(0)
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

    # --------------------------------------------------------
    # Number of sides traded
    #
    # Example:
    #
    #  0 → +1 = 1 side
    # +1 → 0 = 1 side
    # +1 → -1 = 2 sides
    #
    # This is important because reversing position requires
    # closing the old position and opening the new one.
    # --------------------------------------------------------

    result["trade_sides"] = (
        result["position_change"].abs()
    )

    # --------------------------------------------------------
    # Gross strategy return
    #
    # position:
    #   +1 → receive future BTC return
    #    0 → zero return
    #   -1 → inverse of BTC return
    # --------------------------------------------------------

    result["gross_return"] = (
        result["position"]
        * result["future_1h_return"]
    )

    # --------------------------------------------------------
    # Trading costs
    # --------------------------------------------------------

    result["trading_cost"] = (
        result["trade_sides"]
        * TOTAL_COST_PER_SIDE
    )

    # --------------------------------------------------------
    # Net return
    # --------------------------------------------------------

    result["net_return"] = (
        result["gross_return"]
        - result["trading_cost"]
    )

    # --------------------------------------------------------
    # Equity curve
    # --------------------------------------------------------

    result["equity"] = (
        INITIAL_CAPITAL
        * (1 + result["net_return"])
        .cumprod()
    )

    return result


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    result: pd.DataFrame,
) -> dict:

    # --------------------------------------------------------
    # Trade rows
    # --------------------------------------------------------

    trade_rows = result[
        result["position"] != 0
    ]

    number_of_position_periods = len(
        trade_rows
    )

    number_of_entries = (
        (
            (result["position"] != 0)
            & (
                result["previous_position"]
                == 0
            )
        )
    ).sum()

    number_of_exits = (
        (
            (result["position"] == 0)
            & (
                result["previous_position"]
                != 0
            )
        )
    ).sum()

    number_of_reversals = (
        (
            (
                result["position"]
                * result["previous_position"]
            )
            == -1
        )
    ).sum()

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    if number_of_position_periods > 0:

        winning_periods = (
            trade_rows["net_return"] > 0
        ).sum()

        win_rate = (
            winning_periods
            / number_of_position_periods
        )

    else:

        win_rate = np.nan

    # --------------------------------------------------------
    # Total return
    # --------------------------------------------------------

    final_equity = result[
        "equity"
    ].iloc[-1]

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    running_max = result[
        "equity"
    ].cummax()

    drawdown = (
        result["equity"]
        / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    positive_returns = trade_rows.loc[
        trade_rows["net_return"] > 0,
        "net_return",
    ].sum()

    negative_returns = trade_rows.loc[
        trade_rows["net_return"] < 0,
        "net_return",
    ].sum()

    if negative_returns != 0:

        profit_factor = (
            positive_returns
            / abs(negative_returns)
        )

    else:

        profit_factor = np.inf

    # --------------------------------------------------------
    # Average return per position period
    # --------------------------------------------------------

    if number_of_position_periods > 0:

        average_position_return = (
            trade_rows["net_return"].mean()
        )

    else:

        average_position_return = np.nan

    # --------------------------------------------------------
    # Sharpe ratio
    #
    # This is a simple hourly Sharpe estimate.
    # It is NOT annualized here because the goal is first to
    # compare strategies consistently.
    # --------------------------------------------------------

    returns = result[
        "net_return"
    ]

    if returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(24 * 365)
        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    market_exposure = (
        (
            result["position"] != 0
        ).mean()
    )

    long_exposure = (
        (
            result["position"] == 1
        ).mean()
    )

    short_exposure = (
        (
            result["position"] == -1
        ).mean()
    )

    return {
        "position_periods":
            number_of_position_periods,

        "entries":
            int(number_of_entries),

        "exits":
            int(number_of_exits),

        "reversals":
            int(number_of_reversals),

        "win_rate":
            win_rate,

        "total_return":
            total_return,

        "final_equity":
            final_equity,

        "max_drawdown":
            max_drawdown,

        "profit_factor":
            profit_factor,

        "avg_position_return":
            average_position_return,

        "sharpe":
            sharpe,

        "market_exposure":
            market_exposure,

        "long_exposure":
            long_exposure,

        "short_exposure":
            short_exposure,
    }


# ============================================================
# BUY AND HOLD
# ============================================================

def calculate_buy_and_hold(
    df: pd.DataFrame,
) -> dict:

    returns = df[
        "future_1h_return"
    ]

    equity = (
        INITIAL_CAPITAL
        * (1 + returns)
        .cumprod()
    )

    final_equity = equity.iloc[-1]

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

    running_max = equity.cummax()

    drawdown = (
        equity
        / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("BTC XGBOOST V2 POSITION-BASED BACKTEST")
    print("=" * 70)

    print(
        f"\nInitial capital : "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Fee per side   : "
        f"{FEE_RATE * 100:.3f}%"
    )

    print(
        f"Slippage/side  : "
        f"{SLIPPAGE_RATE * 100:.3f}%"
    )

    print(
        f"Total cost/side: "
        f"{TOTAL_COST_PER_SIDE * 100:.3f}%"
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    df = load_predictions()

    print(
        f"\nPrediction rows: {len(df):,}"
    )

    print(
        f"Start: {df['open_time'].iloc[0]}"
    )

    print(
        f"End  : {df['open_time'].iloc[-1]}"
    )

    # --------------------------------------------------------
    # Run threshold tests
    # --------------------------------------------------------

    results = []

    for long_threshold, short_threshold in THRESHOLDS:

        backtest = run_backtest(
            df,
            long_threshold,
            short_threshold,
        )

        metrics = calculate_metrics(
            backtest
        )

        results.append(
            {
                "long_threshold":
                    long_threshold,

                "short_threshold":
                    short_threshold,

                **metrics,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRATEGY RESULTS")
    print("=" * 70)

    display_df = results_df.copy()

    display_df["win_rate"] *= 100
    display_df["total_return"] *= 100
    display_df["max_drawdown"] *= 100
    display_df["avg_position_return"] *= 100
    display_df["market_exposure"] *= 100
    display_df["long_exposure"] *= 100
    display_df["short_exposure"] *= 100

    print(
        display_df.to_string(
            index=False,
            formatters={
                "long_threshold":
                    "{:.2f}".format,

                "short_threshold":
                    "{:.2f}".format,

                "position_periods":
                    "{:.0f}".format,

                "entries":
                    "{:.0f}".format,

                "exits":
                    "{:.0f}".format,

                "reversals":
                    "{:.0f}".format,

                "win_rate":
                    "{:.2f}%".format,

                "total_return":
                    "{:.2f}%".format,

                "final_equity":
                    "${:,.2f}".format,

                "max_drawdown":
                    "{:.2f}%".format,

                "profit_factor":
                    lambda x: (
                        f"{x:.3f}"
                        if np.isfinite(x)
                        else "inf"
                    ),

                "avg_position_return":
                    "{:.4f}%".format,

                "sharpe":
                    "{:.3f}".format,

                "market_exposure":
                    "{:.2f}%".format,

                "long_exposure":
                    "{:.2f}%".format,

                "short_exposure":
                    "{:.2f}%".format,
            },
        )
    )

    # --------------------------------------------------------
    # Buy and hold
    # --------------------------------------------------------

    buy_hold = calculate_buy_and_hold(
        df
    )

    print("\n" + "=" * 70)
    print("BUY & HOLD")
    print("=" * 70)

    print(
        f"Return       : "
        f"{buy_hold['total_return'] * 100:.2f}%"
    )

    print(
        f"Final equity : "
        f"${buy_hold['final_equity']:,.2f}"
    )

    print(
        f"Max drawdown : "
        f"{buy_hold['max_drawdown'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Best threshold
    # --------------------------------------------------------

    best_index = results_df[
        "total_return"
    ].idxmax()

    best = results_df.loc[
        best_index
    ]

    print("\n" + "=" * 70)
    print("BEST V2 THRESHOLD BY TOTAL RETURN")
    print("=" * 70)

    print(
        f"LONG threshold : "
        f"{best['long_threshold']:.2f}"
    )

    print(
        f"SHORT threshold: "
        f"{best['short_threshold']:.2f}"
    )

    print(
        f"Return         : "
        f"{best['total_return'] * 100:.2f}%"
    )

    print(
        f"Max drawdown   : "
        f"{best['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Profit factor  : "
        f"{best['profit_factor']:.3f}"
    )

    print(
        f"Sharpe         : "
        f"{best['sharpe']:.3f}"
    )

    print("\n" + "=" * 70)
    print("BACKTEST V2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()