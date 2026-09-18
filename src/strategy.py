import numpy as np
import pandas as pd


def calculate_zscore(data, asset1, asset2, hedge_ratios, window=30):
    spread = data[asset1] - hedge_ratios * data[asset2]
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)
    return spread, zscore


def generate_signals(
    data,
    asset1,
    asset2,
    hedge_ratios,
    entry_z=2.0,
    exit_z=0.5,
    window=30,
):
    _, zscore = calculate_zscore(
        data, asset1, asset2, hedge_ratios, window
    )

    signals = pd.Series(0, index=data.index, dtype=int)

    long_entry = zscore < -entry_z
    short_entry = zscore > entry_z
    long_exit = zscore > -exit_z
    short_exit = zscore < exit_z

    pos = 0
    for i in range(len(zscore)):
        if pd.isna(zscore.iloc[i]):
            signals.iloc[i] = pos
            continue

        if pos == 0:
            if long_entry.iloc[i]:
                pos = 1
            elif short_entry.iloc[i]:
                pos = -1
        elif pos == 1 and long_exit.iloc[i]:
            pos = 0
        elif pos == -1 and short_exit.iloc[i]:
            pos = 0

        signals.iloc[i] = pos

    return signals
