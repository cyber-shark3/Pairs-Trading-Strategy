import numpy as np
import pandas as pd


def run_backtest(data, signals, asset1, asset2, hedge_ratios):
    signals = signals.reindex(data.index).fillna(0).astype(float)
    hedge_ratios = hedge_ratios.reindex(data.index).ffill()

    asset1_ret = data[asset1].pct_change().fillna(0)
    asset2_ret = data[asset2].pct_change().fillna(0)

    beta = hedge_ratios.shift(1)
    prev_signals = signals.shift(1).fillna(0)

    # Gross-dollar-neutral sizing.
    denominator = 1.0 + beta.abs()
    w1 = 1.0 / denominator
    w2 = -beta / denominator

    strategy_ret = prev_signals * (w1 * asset1_ret + w2 * asset2_ret)
    strategy_ret = strategy_ret.replace([np.inf, -np.inf], np.nan).fillna(0)

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
        "Total Return": float(total_ret),
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown": float(max_dd),
    }

    return strategy_ret, cum_ret, metrics
