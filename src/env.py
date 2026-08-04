"""Gymnasium environment for the pricing MDP.

An episode follows one store-product pair through EPISODE_LENGTH consecutive
days. The agent picks a multiplier, the demand model returns the quantity
that multiplier would have sold, and the reward comes from the shared
perishable-goods formulation.

Limitation that bounds every result downstream: the autoregressive state
features are replayed from the historical trajectory rather than regenerated
from the agent's own actions. A price cut that raised demand would, in
reality, raise tomorrow's lags too. Capturing that feedback needs a
generative transition model rather than a one-step counterfactual predictor —
which is what a neural causal model would supply. The environment is
therefore a one-step interventional simulator embedded in an exogenous state
trajectory, and the discount factor applies to a reward stream whose state
evolution the agent cannot steer.
"""

import logging
from typing import List, Optional, Protocol, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config import ACTION_GRID, EPISODE_LENGTH, RANDOM_SEED
from src.data import ACTION_FEATURE, PAIR_KEYS
from src.reward import reward_curve, stock_of

logger = logging.getLogger(__name__)


class DemandModel(Protocol):
    """Anything that answers the environment's interventional query."""

    def predict_under_action(self, X: pd.DataFrame, discount: float) -> np.ndarray: ...


class PricingEnv(gym.Env):
    """Discrete-action pricing environment over a learned demand simulator.

    Action: an index into the multiplier grid.
    Observation: the feature vector minus the action — the agent chooses it,
    so feeding the historical value would leak the behaviour policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_model: DemandModel,
        state_data: pd.DataFrame,
        feature_names: Sequence[str],
        action_grid: Sequence[float] = tuple(ACTION_GRID),
        episode_length: int = EPISODE_LENGTH,
        seed: Optional[int] = RANDOM_SEED,
    ) -> None:
        super().__init__()

        self.demand_model = demand_model
        self.feature_names = list(feature_names)
        self.action_grid = np.asarray(action_grid, dtype=float)
        self.episode_length = episode_length

        self.data = state_data.sort_values([*PAIR_KEYS, "dt"]).reset_index(drop=True)
        self.stock = stock_of(self.data)
        self._episodes = self._build_episodes()

        self.obs_features = [f for f in self.feature_names if f != ACTION_FEATURE]
        self.action_space = spaces.Discrete(len(self.action_grid))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.obs_features),), dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._block: np.ndarray = self._episodes[0]
        self._t = 0

        logger.info(
            "Environment: %d episodes, %d actions, %d observations",
            len(self._episodes), len(self.action_grid), len(self.obs_features),
        )

    def _build_episodes(self) -> List[np.ndarray]:
        """Blocks of consecutive days within one pair."""
        blocks = []
        for _, group in self.data.groupby(list(PAIR_KEYS), sort=False):
            index = group.index.to_numpy()
            for start in range(0, len(index) - self.episode_length + 1, self.episode_length):
                blocks.append(index[start:start + self.episode_length])
        return blocks

    def _observation(self) -> np.ndarray:
        row = self.data.loc[self._block[self._t], self.obs_features]
        return np.nan_to_num(row.to_numpy(dtype=np.float32), nan=0.0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._block = self._episodes[self._rng.integers(len(self._episodes))]
        self._t = 0
        return self._observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        position = self._block[self._t]
        discount = float(self.action_grid[int(action)])

        row = self.data.loc[[position], self.feature_names]
        quantity = float(self.demand_model.predict_under_action(row, discount)[0])

        reward = float(reward_curve(
            np.array([quantity]),
            self.stock[[position]],
            np.array([discount]),
        )[0, 0])

        self._t += 1
        terminated = self._t >= len(self._block)
        observation = (
            self._observation() if not terminated
            else np.zeros(len(self.obs_features), dtype=np.float32)
        )
        return observation, reward, terminated, False, {
            "quantity": quantity, "discount": discount,
        }