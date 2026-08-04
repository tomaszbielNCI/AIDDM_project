"""Pricing policies: baselines, value iteration, PPO and decision-focused.

Every policy exposes select_action(states) -> action indices, so the
evaluation harness treats them identically and differences in measured reward
are attributable to the policy rather than the harness.
"""

import logging
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import ACTION_GRID, DISCOUNT_FACTOR, RANDOM_SEED, UNIT_COST_RATIO, WASTE_RATIO
from src.data import ACTION_FEATURE
from src.demand_model import LGBMDemandModel
from src.env import PricingEnv
from src.reward import reward_curve, stock_of

logger = logging.getLogger(__name__)


class PricingPolicy(Protocol):
    """Common interface: an action index per state."""

    name: str

    def select_action(self, states: pd.DataFrame) -> np.ndarray: ...


# =========================================================================
# Baselines
# =========================================================================

class FixedPrice:
    """Charge one multiplier everywhere. The do-nothing benchmark."""

    def __init__(self, discount: float = 1.0,
                 action_grid: Sequence[float] = tuple(ACTION_GRID)) -> None:
        self.action_grid = np.asarray(action_grid)
        self.action = int(np.abs(self.action_grid - discount).argmin())
        self.name = f"fixed_{self.action_grid[self.action]:.2f}"

    def select_action(self, states: pd.DataFrame) -> np.ndarray:
        return np.full(len(states), self.action, dtype=int)


class HistoricalPolicy:
    """Replay the operator's own pricing, snapped to the grid.

    The reference baseline in revenue management: beating what the business
    actually did is the only comparison with commercial meaning.
    """

    def __init__(self, action_grid: Sequence[float] = tuple(ACTION_GRID)) -> None:
        self.action_grid = np.asarray(action_grid)
        self.name = "historical"

    def select_action(self, states: pd.DataFrame) -> np.ndarray:
        observed = states[ACTION_FEATURE].to_numpy()[:, None]
        return np.abs(observed - self.action_grid[None, :]).argmin(axis=1)


class RuleBased:
    """Markdown rule of the kind a category manager would apply.

    Discounts on weekends and holidays. Uses no demand model and no learned
    parameters, which is what makes it a baseline rather than a competitor.
    The campaign flag is deliberately not used as a trigger: it is a mediator
    of the price effect and is excluded from the demand model for that reason.
    """

    def __init__(self, promo_discount: float = 0.85,
                 action_grid: Sequence[float] = tuple(ACTION_GRID)) -> None:
        grid = np.asarray(action_grid)
        self.promo = int(np.abs(grid - promo_discount).argmin())
        self.full = int(np.abs(grid - 1.0).argmin())
        self.name = "rule_based"

    def select_action(self, states: pd.DataFrame) -> np.ndarray:
        trigger = (
            (states["is_weekend"] == 1) | (states["holiday_flag"] == 1)
        ).to_numpy()
        return np.where(trigger, self.promo, self.full)


class PredictThenOptimise:
    """Fit demand on squared error, then price by maximising expected reward.

    The standard industry pipeline and the control arm for the
    decision-focused comparison: the same decision layer sits on both, so any
    difference isolates the training loss rather than the decision rule.
    """

    def __init__(self, demand_model: LGBMDemandModel, feature_names: Sequence[str],
                 action_grid: Sequence[float] = tuple(ACTION_GRID),
                 name: str = "predict_then_optimise") -> None:
        self.model = demand_model
        self.feature_names = list(feature_names)
        self.action_grid = np.asarray(action_grid)
        self.name = name

    def expected_reward(self, states: pd.DataFrame) -> np.ndarray:
        demand = self.model.demand_curve(states[self.feature_names], self.action_grid)
        return reward_curve(demand, stock_of(states), self.action_grid)

    def select_action(self, states: pd.DataFrame) -> np.ndarray:
        return self.expected_reward(states).argmax(axis=1)


class Clairvoyant(PredictThenOptimise):
    """Upper bound: prices against the model that also generates the reward.

    Standard in revenue management as the ceiling against which feasible
    policies are measured. It is unattainable — it conditions on the outcome
    it is evaluated against — but the gap to it separates value lost to an
    imperfect demand model from value lost to a poor decision rule.

    The distinction only bites when the competing policies use a DIFFERENT
    demand model. Here the policies are fitted on observed sales while the
    evaluation simulator is fitted on demand recovered from censoring, so the
    gap measures exactly what censoring costs a pricing system.
    """

    def __init__(self, demand_model: LGBMDemandModel, feature_names: Sequence[str],
                 action_grid: Sequence[float] = tuple(ACTION_GRID)) -> None:
        super().__init__(demand_model, feature_names, action_grid, name="clairvoyant")


# =========================================================================
# Value iteration
# =========================================================================

VI_STATE_FEATURES: List[str] = ["weekday", "holiday_flag"]
VI_DEMAND_BINS: int = 4


class TabularValueIteration:
    """Value iteration on a coarsely discretised MDP.

    Included as the algorithm carrying a convergence guarantee: the Bellman
    operator is a gamma-contraction on bounded value functions under the
    supremum norm, so Banach's fixed point theorem gives a unique optimal
    value function and geometric convergence,

        ||V_n - V*||_inf <= gamma^n ||V_0 - V*||_inf.

    The residual sequence recorded here is the empirical counterpart of that
    bound and is plotted against gamma^n in the report.

    The agent's price does not steer the exogenous state, so the transition
    kernel is action-independent and only the reward depends on the action —
    a deliberate simplification of the general MDP.
    """

    def __init__(self, gamma: float = DISCOUNT_FACTOR) -> None:
        self.gamma = gamma
        self.states: List[str] = []
        self.policy = np.zeros(0, dtype=int)
        self.residuals: List[float] = []
        self.name = "value_iteration"

    def _encode(self, states: pd.DataFrame) -> pd.Series:
        demand_bin = pd.qcut(
            states["observed_demand_roll7_mean"].fillna(0),
            VI_DEMAND_BINS, labels=False, duplicates="drop",
        )
        parts = [states[f].astype(int).astype(str) for f in VI_STATE_FEATURES]
        parts.append(pd.Series(demand_bin, index=states.index).fillna(0).astype(int).astype(str))
        return parts[0].str.cat(parts[1:], sep="|")

    def fit(self, states: pd.DataFrame, rewards: np.ndarray,
            max_sweeps: int = 1000, tolerance: float = 1e-8) -> "TabularValueIteration":
        """Iterate the Bellman operator to its fixed point."""
        codes = self._encode(states)
        self.states = sorted(codes.unique())
        index = {s: i for i, s in enumerate(self.states)}
        n_states = len(self.states)
        code_idx = codes.map(index).to_numpy()

        expected_reward = np.zeros((n_states, rewards.shape[1]))
        counts = np.zeros(n_states)
        np.add.at(expected_reward, code_idx, rewards)
        np.add.at(counts, code_idx, 1)
        expected_reward /= np.maximum(counts, 1)[:, None]

        transitions = np.zeros((n_states, n_states))
        np.add.at(transitions, (code_idx[:-1], code_idx[1:]), 1)
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.divide(
            transitions, row_sums,
            out=np.full_like(transitions, 1 / n_states), where=row_sums > 0,
        )

        values = np.zeros(n_states)
        self.residuals = []
        for _ in range(max_sweeps):
            q_values = expected_reward + self.gamma * (transitions @ values)[:, None]
            updated = q_values.max(axis=1)
            residual = float(np.abs(updated - values).max())
            self.residuals.append(residual)
            values = updated
            if residual < tolerance:
                break

        self.policy = q_values.argmax(axis=1)
        logger.info("Value iteration: %d states, %d sweeps, residual %.2e",
                    n_states, len(self.residuals), self.residuals[-1])
        return self

    def select_action(self, states: pd.DataFrame) -> np.ndarray:
        index = {s: i for i, s in enumerate(self.states)}
        codes = self._encode(states).map(index)
        return np.where(codes.isna(), 0, self.policy[codes.fillna(0).astype(int)])


# =========================================================================
# PPO
# =========================================================================

class PPOAgent:
    """PPO over the learned environment, wrapped in the common interface.

    On-policy PPO against a simulator built from historical data is offline
    reinforcement learning wearing an online interface: the agent may propose
    multipliers the behaviour policy rarely took, and there the demand model
    extrapolates. The action grid is confined to the observed support for
    that reason.
    """

    def __init__(self, env: PricingEnv, learning_rate: float = 3e-4,
                 gamma: float = DISCOUNT_FACTOR, seed: int = RANDOM_SEED) -> None:
        self.obs_features = env.obs_features
        self.model = PPO(
            "MlpPolicy", DummyVecEnv([lambda: Monitor(env)]),
            learning_rate=learning_rate, gamma=gamma, seed=seed, verbose=0,
        )
        self.name = "ppo"

    def learn(self, timesteps: int = 100_000) -> "PPOAgent":
        self.model.learn(total_timesteps=timesteps, progress_bar=False)
        logger.info("PPO trained for %d timesteps", timesteps)
        return self

    def select_action(self, states: pd.DataFrame) -> np.ndarray:
        observations = np.nan_to_num(
            states[self.obs_features].to_numpy(dtype=np.float32), nan=0.0
        )
        actions, _ = self.model.predict(observations, deterministic=True)
        return np.asarray(actions).ravel()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)


# =========================================================================
# Decision-focused learning
# =========================================================================

class DecisionLoss(nn.Module):
    """Decision-aware surrogate losses over a discrete pricing decision.

    Decision regret is piecewise constant in the model parameters — the
    argmax jumps rather than slides — so its gradient vanishes almost
    everywhere. Two surrogates address this, and both fail here for
    different reasons.

    SPO+ (Elmachtoub & Grigas, Management Science 2022) is a convex upper
    bound on regret with an explicit subgradient, evaluated at the perturbed
    prediction 2*q_hat - q. It was derived for a reward linear in the
    predicted quantity. The reward used here is piecewise linear, because
    sales are capped by stock, and the surrogate loses its non-negativity:
    the training loss falls from -0.25 to -0.66, which no bound on a
    non-negative regret can do.

    The perturbed optimiser (Berthet et al., NeurIPS 2020) smooths the argmax
    by averaging over Gaussian perturbations, so the gradient survives where
    the hard argmax is locally constant. It trains, but a purely
    decision-aware loss constrains only which action wins, never the level of
    the prediction: mean predicted demand drifts to 4.85 against a true level
    near 1.0, stock saturates at every action, the waste term vanishes and
    the policy collapses to full price.

    The anchor term fixes the second failure and not the first. At weight 0.1
    the perturbed policy already differentiates (entropy 1.32), at 0.5 it
    reaches 1.79 and 1.0 adds nothing; SPO+ is unaffected at any weight.
    """

    def __init__(self, mode: str = "spo",
                 action_grid: Sequence[float] = tuple(ACTION_GRID),
                 unit_cost_ratio: float = UNIT_COST_RATIO,
                 waste_ratio: float = WASTE_RATIO,
                 temperature: float = 0.1,
                 n_perturbations: int = 10,
                 anchor_weight: float = 0.5,
                 seed: int = RANDOM_SEED) -> None:
        super().__init__()
        if mode not in {"spo", "perturbed"}:
            raise ValueError(f"unknown decision loss: {mode}")

        self.register_buffer("actions", torch.tensor(list(action_grid), dtype=torch.float32))
        self.mode = mode
        self.unit_cost_ratio = unit_cost_ratio
        self.waste_ratio = waste_ratio
        self.temperature = temperature
        self.n_perturbations = n_perturbations
        self.anchor_weight = anchor_weight
        self.generator = torch.Generator().manual_seed(seed)

    def _reward(self, quantity: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        """Reward per sample and action, shape (n, n_actions)."""
        sold = torch.minimum(quantity[:, None], stock[:, None])
        waste = torch.clamp(stock[:, None] - sold, min=0.0)
        return (self.actions[None, :] * sold
                - self.unit_cost_ratio * sold
                - self.waste_ratio * waste)

    def _spo_plus(self, predicted, actual, stock):
        best_true = self._reward(actual, stock).argmax(dim=1)
        return (
            self._reward(2 * predicted - actual, stock).max(dim=1).values
            - 2 * self._reward(predicted, stock).gather(1, best_true[:, None]).squeeze(1)
            + self._reward(actual, stock).max(dim=1).values
        ).mean()

    def _perturbed(self, predicted, actual, stock):
        """Expected regret under Gaussian perturbation of the prediction.

        Averaging over perturbations makes the induced action distribution
        smooth in the prediction, so the gradient survives even where the
        hard argmax is locally constant.
        """
        scale = self.temperature * (actual.std() + 1e-6)
        best_true = self._reward(actual, stock).argmax(dim=1)
        optimal_value = self._reward(actual, stock).gather(1, best_true[:, None]).squeeze(1)

        regret = torch.zeros_like(predicted)
        for _ in range(self.n_perturbations):
            noise = torch.randn(
                predicted.shape, generator=self.generator, device=predicted.device
            )
            weights = torch.softmax(
                self._reward(predicted + scale * noise, stock) / (scale + 1e-6), dim=1
            )
            achieved = (weights * self._reward(actual, stock)).sum(dim=1)
            regret = regret + (optimal_value - achieved)

        return (regret / self.n_perturbations).mean()

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor,
                stock: torch.Tensor) -> torch.Tensor:
        """Decision loss, optionally anchored to the level of demand.

        A purely decision-aware loss constrains only which action wins, not
        the magnitude of the prediction, so the network is free to drift in
        scale. The drift is fatal here because the reward saturates: once
        predicted demand exceeds the stock at every action, min(demand,
        stock) is constant, the waste term vanishes and the argmax collapses
        to full price regardless of what the network learnt about the price
        response. The anchor restores a weak level constraint without
        reverting to prediction-first training.
        """
        decision = (
            self._spo_plus(predicted, actual, stock) if self.mode == "spo"
            else self._perturbed(predicted, actual, stock)
        )
        if self.anchor_weight > 0:
            decision = decision + self.anchor_weight * ((predicted - actual) ** 2).mean()
        return decision


class DemandNet(nn.Module):
    """Feed-forward demand predictor.

    Capacity is held identical across loss variants: the comparison is
    between training objectives, so no variant should be able to win by
    having more parameters.
    """

    def __init__(self, n_features: int, hidden: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = n_features
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU(), nn.Dropout(0.1)]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DecisionFocused:
    """Demand network with an argmax pricing layer, trained three ways.

    loss='mse'       -- the control arm: prediction error, decision-agnostic
    loss='spo'       -- SPO+ decision regret surrogate, hard argmax
    loss='perturbed' -- the same regret, smoothed by Gaussian perturbation

    All three share architecture, features, reward and decision layer, so any
    difference in realised reward isolates the training objective.

    The price multiplier is an input feature. Without it the predicted
    quantity would be invariant to the action and the argmax would return the
    same price in every state whatever the network had learnt.
    """

    def __init__(self, feature_names: Sequence[str], loss: str = "spo",
                 action_grid: Sequence[float] = tuple(ACTION_GRID),
                 hidden: Sequence[int] = (64, 32), learning_rate: float = 1e-3,
                 seed: int = RANDOM_SEED) -> None:
        torch.manual_seed(seed)
        self.feature_names = list(feature_names)
        self.action_grid = np.asarray(action_grid)
        self.loss_name = loss
        self.learning_rate = learning_rate
        self.name = f"dfl_{loss}"

        self.net = DemandNet(len(self.feature_names), hidden)
        self.criterion = (
            nn.MSELoss() if loss == "mse" else DecisionLoss(mode=loss, action_grid=action_grid)
        )
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.history: List[float] = []

    def _prepare(self, states: pd.DataFrame, fit: bool = False) -> torch.Tensor:
        X = np.nan_to_num(
            states[self.feature_names].to_numpy(dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-8
        return torch.tensor((X - self.mean_) / self.std_, dtype=torch.float32)

    def fit(self, states: pd.DataFrame, quantity: np.ndarray,
            epochs: int = 200, batch_size: int = 512) -> "DecisionFocused":
        X = self._prepare(states, fit=True)
        y = torch.tensor(np.asarray(quantity, dtype=np.float32))
        stock = torch.tensor(stock_of(states).astype(np.float32))

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.history = []

        for _ in range(epochs):
            permutation = torch.randperm(len(X))
            total = 0.0
            for start in range(0, len(X), batch_size):
                batch = permutation[start:start + batch_size]
                optimiser.zero_grad()
                prediction = torch.relu(self.net(X[batch]))
                loss = (
                    self.criterion(prediction, y[batch])
                    if self.loss_name == "mse"
                    else self.criterion(prediction, y[batch], stock[batch])
                )
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)
            self.history.append(total / len(X))

        logger.info("%s: %d epochs, loss %.4f -> %.4f",
                    self.name, epochs, self.history[0], self.history[-1])
        return self

    def predict_demand(self, states: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            prediction = torch.relu(self.net(self._prepare(states))).numpy()
        self.net.train()
        return prediction

    def predict_under_action(self, states: pd.DataFrame, discount: float) -> np.ndarray:
        counterfactual = states.copy()
        counterfactual[ACTION_FEATURE] = discount
        return self.predict_demand(counterfactual)

    def expected_reward(self, states: pd.DataFrame) -> np.ndarray:
        demand = np.column_stack(
            [self.predict_under_action(states, a) for a in self.action_grid]
        )
        return reward_curve(demand, stock_of(states), self.action_grid)

    def select_action(self, states: pd.DataFrame) -> np.ndarray:
        return self.expected_reward(states).argmax(axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "mean": self.mean_,
                    "std": self.std_, "features": self.feature_names}, path)