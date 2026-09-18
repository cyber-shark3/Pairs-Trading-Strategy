import numpy as np
import pandas as pd


def run_backtest(data, signals, asset1, asset2, hedge_ratios):
    prev_signals = signals.shift(1).fillna(0)
    prev_hedge_ratios = hedge_ratios.shift(1).ffill().fillna(0)

    asset1_ret = data[asset1].pct_change().fillna(0)
    asset2_ret = data[asset2].pct_change().fillna(0)

    strategy_ret = prev_signals * (
        asset1_ret - prev_hedge_ratios * asset2_ret
    )

    cum_ret = (1 + strategy_ret).cumprod()
    total_ret = cum_ret.iloc[-1] - 1

    std_ret = strategy_ret.std()
    sharpe = (
        np.sqrt(252) * strategy_ret.mean() / std_ret
        if std_ret > 0
        else 0
    )

    max_dd = (cum_ret / cum_ret.cummax() - 1).min()

    metrics = {
        "Total Return": total_ret,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }

    return strategy_ret, cum_ret, metrics
