import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.cointegration import test_cointegration
from src.hedge_ratio import calculate_dynamic_hedge_ratio
from src.strategy import generate_signals
from src.backtest import run_backtest
import config


def main():
    data = load_data(
        config.ASSET_1,
        config.ASSET_2,
        config.START_DATE,
        config.END_DATE,
    )

    is_coint = test_cointegration(
        data, config.ASSET_1, config.ASSET_2
    )

    if not is_coint:
        print(
            "WARNING: Assets are not strongly cointegrated. "
            "Mean reversion might fail."
        )

    hedge_ratios = calculate_dynamic_hedge_ratio(
        data[config.ASSET_1],
        data[config.ASSET_2],
    )

    signals = generate_signals(
        data,
        config.ASSET_1,
        config.ASSET_2,
        hedge_ratios,
        config.ZSCORE_ENTRY_THRESHOLD,
        config.ZSCORE_EXIT_THRESHOLD,
        config.ROLLING_WINDOW,
    )

    strategy_rets, cum_ret, metrics = run_backtest(
        data,
        signals,
        config.ASSET_1,
        config.ASSET_2,
        hedge_ratios,
    )

    print("\n--- Backtest Results ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 14), sharex=True
    )

    axes[0].plot(
        data.index,
        data[config.ASSET_1],
        label=config.ASSET_1,
    )
    axes[0].plot(
        data.index,
        data[config.ASSET_2],
        label=config.ASSET_2,
    )
    axes[0].set_title("Asset Prices")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        data.index,
        hedge_ratios,
        label="Dynamic Hedge Ratio",
    )
    axes[1].set_title("Kalman Filter Dynamic Hedge Ratio")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        cum_ret.index,
        cum_ret,
        label="Strategy Cumulative Returns",
    )
    axes[2].set_title("Cumulative Returns")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
