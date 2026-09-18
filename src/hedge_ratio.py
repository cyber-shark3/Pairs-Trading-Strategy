from pykalman import KalmanFilter
import numpy as np
import pandas as pd


def calculate_dynamic_hedge_ratio(asset1_prices, asset2_prices):
    obs_mat = np.vstack(
        [asset2_prices.values, np.ones(len(asset2_prices))]
    ).T
    obs_mat = np.expand_dims(obs_mat, axis=1)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.eye(2) * 100,
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=np.eye(1) * 1.0,
        transition_covariance=np.eye(2) * 0.01,
    )

    pred_states, _ = kf.filter(asset1_prices.values)

    return pd.Series(
        pred_states[:, 0],
        index=asset1_prices.index,
        name="hedge_ratio",
    )
