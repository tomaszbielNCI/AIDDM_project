"""The reward function, defined once.

Every policy, the environment and the evaluation harness import from here. A
comparison in which the policies optimise different objectives measures the
objectives rather than the policies.
"""

import numpy as np
import pandas as pd

from src.config import STOCK_MULTIPLIER, UNIT_COST_RATIO, WASTE_RATIO

STOCK_PROXY_COLUMN: str = "observed_demand_roll7_mean"


def stock_of(states: pd.DataFrame, multiplier: float = STOCK_MULTIPLIER) -> np.ndarray:
    """Stock available for the day.

    Uses the level reconstructed from stockout annotations where available:
    on a day that sells out, cumulative sales up to the first stockout hour
    are the stock that was there, which makes 38.8% of days an observation
    rather than an estimate. The remaining days are filled from the pair's
    own stocking behaviour.

    Falls back to the trailing-mean proxy only when the reconstruction is
    absent, so a frame built before the stock estimation step still works.
    """
    if "estimated_stock" in states.columns and states["estimated_stock"].notna().all():
        return states["estimated_stock"].to_numpy()

    column = states[STOCK_PROXY_COLUMN]
    return multiplier * column.fillna(column.mean()).to_numpy()


def reward_curve(
    demand: np.ndarray,
    stock: np.ndarray,
    action_grid: np.ndarray,
    unit_cost_ratio: float = UNIT_COST_RATIO,
    waste_ratio: float = WASTE_RATIO,
) -> np.ndarray:
    """Reward at every action level.

        sold   = min(demand, stock)
        reward = a * sold - cost * sold - waste_ratio * (stock - sold)

    The unknown per-SKU base price scales revenue, cost and waste alike, so it
    cancels from the argmax and the reward is a normalised index.

    Args:
        demand: Quantity, shape (n,) or (n, n_actions). A one-dimensional
            input is broadcast across actions.
        stock: Stock per state, shape (n,).
        action_grid: Price multipliers, shape (n_actions,).

    Returns:
        Reward per state and action, shape (n, n_actions).
    """
    grid = np.asarray(action_grid, dtype=float)
    quantity = np.asarray(demand, dtype=float)

    if quantity.ndim == 1:
        quantity = np.repeat(quantity[:, None], len(grid), axis=1)

    stock_column = np.asarray(stock, dtype=float)[:, None]
    sold = np.minimum(quantity, stock_column)
    waste = np.maximum(stock_column - sold, 0.0)

    return grid[None, :] * sold - unit_cost_ratio * sold - waste_ratio * waste
