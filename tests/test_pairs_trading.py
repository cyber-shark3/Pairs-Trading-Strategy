import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.strategy import generate_signals


def test_backtest_is_gross_dollar_neutral():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame(
        {"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 110.0]},
        index=dates,
    )
    hedge = pd.Series(2.0, index=dates)
    signals = pd.Series([0, 1, 1], index=dates)

    strategy_ret, _, _ = run_backtest(data, signals, "A", "B", hedge)

    assert np.isclose(strategy_ret.iloc[1], 0.10 / 3.0)
    assert np.isclose(strategy_ret.iloc[2], -(2.0 / 3.0) * 0.10)
    assert np.isfinite(strategy_ret).all()


def test_signal_generator_returns_valid_series():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    data = pd.DataFrame(
        {"A": np.linspace(100, 110, 40), "B": np.linspace(50, 55, 40)},
        index=dates,
    )
    hedge = pd.Series(2.0, index=dates)

    signals = generate_signals(
        data, "A", "B", hedge, entry_z=2.0, exit_z=0.5, window=5
    )

    assert signals.index.equals(data.index)
    assert set(signals.unique()).issubset({-1, 0, 1})
