# backtests/simple_backtest.py

from pathlib import Path

from engine.data_loader import load_csv, load_es_multi_tf
from engine.strategies import get_strategy_class


ROOT = Path(__file__).resolve().parent.parent


def run_backtest(
    strategy_name: str = "simple_trend",
    filename: str = "es_5m_clean.csv",
):
    print("\n" + "=" * 60)
    print(f"Running backtest for strategy: {strategy_name}")
    print(f"Using data file: {filename}")

    # 1) Load dataset
    if strategy_name == "smc_po3":
        # Multi-timeframe ES (5m + 1h + 2h)
        df = load_es_multi_tf(
            f_5m="es_5m_clean.csv",
            f_1h="es_1h_clean.csv",
            f_2h="es_2h_clean.csv",
        )
    else:
        # Simple single-TF 5m data
        df = load_csv(filename)

    df = df.sort_values("time").reset_index(drop=True)

    # 2) Initialize strategy
    StrategyClass = get_strategy_class(strategy_name)
    strategy = StrategyClass()

    cash = 0.0
    equity_curve = []
    trades = []
    position = 0          # +1 long, -1 short, 0 flat
    entry_price = None

    # 3) Walk forward through each bar
    for _, row in df.iterrows():
        price = float(row["close"])
        decision = strategy.on_bar(row)

        if decision == "FLAT":
            if position != 0 and entry_price is not None:
                pnl = (price - entry_price) * position
                cash += pnl
                trades.append(
                    {"exit_time": row["time"], "exit_price": price, "pnl": pnl}
                )
                position = 0
                entry_price = None

        elif decision == "LONG":
            if position == -1 and entry_price is not None:
                pnl = (price - entry_price) * position
                cash += pnl
                trades.append(
                    {"exit_time": row["time"], "exit_price": price, "pnl": pnl}
                )
            position = 1
            entry_price = price

        elif decision == "SHORT":
            if position == 1 and entry_price is not None:
                pnl = (price - entry_price) * position
                cash += pnl
                trades.append(
                    {"exit_time": row["time"], "exit_price": price, "pnl": pnl}
                )
            position = -1
            entry_price = price

        equity_curve.append(cash)

    # 4) Close any remaining position
    if position != 0 and entry_price is not None:
        price = df.iloc[-1]["close"]
        pnl = (price - entry_price) * position
        cash += pnl
        trades.append(
            {"exit_time": df.iloc[-1]["time"], "exit_price": price, "pnl": pnl}
        )
        equity_curve[-1] = cash

    # 5) Metrics
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    if equity_curve:
        import pandas as pd
        equity_series = pd.Series(equity_curve)
        max_dd = (equity_series.cummax() - equity_series).max()
    else:
        max_dd = 0.0

    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"PnL: {cash:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "pnl": cash,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }


if __name__ == "__main__":
    for strat in ["simple_trend", "mean_reversion", "smc_po3"]:
        run_backtest(strategy_name=strat)
