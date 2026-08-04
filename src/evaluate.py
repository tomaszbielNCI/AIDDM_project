"""Scoring and comparison of pricing policies.

Each policy chooses; the demand model then answers the counterfactual for
that choice. Every policy is scored against the same model and the same
reward, so differences reflect the decisions.

Uncertainty is bootstrapped over store-product pairs, not rows: consecutive
days within a pair share lag features and local demand level, and resampling
rows would treat correlated observations as independent.
"""

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import ACTION_GRID, RANDOM_SEED
from src.data import ACTION_FEATURE, PAIR_KEYS
from src.demand_model import LGBMDemandModel
from src.reward import reward_curve, stock_of

logger = logging.getLogger(__name__)


def score_policy(
    policy,
    states: pd.DataFrame,
    demand_model: LGBMDemandModel,
    feature_names: Sequence[str],
    action_grid: Sequence[float] = tuple(ACTION_GRID),
) -> pd.DataFrame:
    """Realised reward per state under a policy's chosen actions."""
    grid = np.asarray(action_grid, dtype=float)
    actions = np.asarray(policy.select_action(states)).astype(int)
    discounts = grid[actions]

    X = states[list(feature_names)].copy()
    X[ACTION_FEATURE] = discounts
    quantity = demand_model.predict(X)

    stock = stock_of(states)
    # One column per row: each state is evaluated at its own chosen action.
    sold = np.minimum(quantity, stock)

    reward = reward_curve(
        quantity[:, None], stock, grid
    )[np.arange(len(states)), actions]

    return pd.DataFrame({
        "store_id": states[PAIR_KEYS[0]].to_numpy(),
        "product_id": states[PAIR_KEYS[1]].to_numpy(),
        "action": actions,
        "discount": discounts,
        "quantity": quantity,
        "sold": sold,
        "waste": np.maximum(stock - sold, 0.0),
        "reward": reward,
    })


def bootstrap_ci(per_state: pd.DataFrame, n_boot: int = 1000,
                 seed: int = RANDOM_SEED) -> Tuple[float, float, float]:
    """Mean reward with a 95% interval, resampling whole pairs."""
    frame = per_state.reset_index(drop=True)
    groups: List[np.ndarray] = [
        np.asarray(idx) for idx in frame.groupby(list(PAIR_KEYS)).indices.values()
    ]
    rewards = frame["reward"].to_numpy()
    rng = np.random.default_rng(seed)

    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        draws[b] = rewards[np.concatenate([groups[k] for k in pick])].mean()

    return (float(rewards.mean()),
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def action_entropy(actions: pd.Series) -> float:
    """Entropy of the action distribution, in nats.

    Zero means the policy collapsed to one price. With demand this inelastic
    a policy can score respectably while never adapting to the state, and
    mean reward alone would not reveal it.
    """
    p = actions.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p)).sum())


def compare_policies(
    policies: Dict[str, object],
    states: pd.DataFrame,
    demand_model: LGBMDemandModel,
    feature_names: Sequence[str],
    reference: str = "historical",
) -> pd.DataFrame:
    """Score every policy; report lift over the operator's own pricing."""
    rows = []
    for name, policy in policies.items():
        detail = score_policy(policy, states, demand_model, feature_names)
        mean, low, high = bootstrap_ci(detail)
        rows.append({
            "policy": name, "mean_reward": mean, "ci_low": low, "ci_high": high,
            "mean_discount": detail["discount"].mean(),
            "mean_quantity": detail["quantity"].mean(),
            "mean_waste": detail["waste"].mean(),
            "action_entropy": action_entropy(detail["action"]),
        })

    table = pd.DataFrame(rows).set_index("policy")
    if reference in table.index:
        table["lift_vs_reference"] = table["mean_reward"] / table.loc[reference, "mean_reward"] - 1

    logger.info("Compared %d policies on %d states", len(table), len(states))
    return table.sort_values("mean_reward", ascending=False)

def paired_bootstrap(
    per_state_a: pd.DataFrame,
    per_state_b: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Difference in mean reward between two policies, with a 95% interval.

    Paired on the state: both policies are scored on the same days and the
    same pairs, so resampling the difference removes the variance they share
    and leaves only the variance of the contrast. Comparing two independent
    intervals instead would overstate the uncertainty of the comparison.

    Returns:
        Point estimate of (a - b), lower bound, upper bound.
    """
    difference = per_state_a["reward"].to_numpy() - per_state_b["reward"].to_numpy()
    frame = per_state_a.reset_index(drop=True)
    groups = [np.asarray(idx) for idx in frame.groupby(list(PAIR_KEYS)).indices.values()]

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        draws[b] = difference[np.concatenate([groups[k] for k in pick])].mean()

    return (float(difference.mean()),
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))