"""Structural demand model: price acts through an explicit channel.

The tree ensemble is free to fit any function of the features, which is how
the promotion indicators came to absorb the price signal in the first place.
This model constrains the functional form instead:

    q(s, a) = f(s) * g(a, s)

with the normalisation g(1.0, s) = 1, so f is the demand at full price and g
is the multiplicative response to a markdown. Price cannot be absorbed by the
autoregressive features, because it enters a different term.

This is a structural equation in the sense of a Structural Causal Model
(SCM): the causal channel is imposed rather than discovered. A Neural Causal
Model (NCM) carries one such equation per endogenous variable, tied together
by an acyclic graph, and answers interventional queries about any of them.
Here there is one intervention of interest — the price — so the model reduces
to a single equation, which is what the data supports.

Interface matches LGBMDemandModel, so the environment, the policies and the
evaluation harness take it unchanged.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import ACTION_GRID, RANDOM_SEED
from src.data import ACTION_FEATURE

logger = logging.getLogger(__name__)

FULL_PRICE: float = 1.0


class _Block(nn.Module):
    """Small feed-forward stack."""

    def __init__(self, n_in: int, hidden: Sequence[int], n_out: int = 1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = n_in
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU()]
            previous = size
        layers.append(nn.Linear(previous, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class StructuralDemandModel:
    """Multiplicative demand model with the price in its own channel.

    The baseline block sees the state without the price. The response block
    sees the state and the price, and is evaluated twice per forward pass —
    at the actual price and at full price — so that the ratio enforces
    g(1.0, s) = 1 exactly rather than approximately.

    Trained on log1p demand, like the tree model, so the two are comparable.
    """

    def __init__(self, feature_names: Sequence[str],
                 baseline_hidden: Sequence[int] = (64, 32),
                 response_hidden: Sequence[int] = (32, 16),
                 learning_rate: float = 1e-3,
                 seed: int = RANDOM_SEED) -> None:
        torch.manual_seed(seed)
        self.feature_names = list(feature_names)
        self.state_features = [f for f in self.feature_names if f != ACTION_FEATURE]
        self.learning_rate = learning_rate

        self.baseline = _Block(len(self.state_features), baseline_hidden)
        self.response = _Block(len(self.state_features) + 1, response_hidden)

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.history: List[float] = []

    def _state(self, states: pd.DataFrame, fit: bool = False) -> torch.Tensor:
        X = np.nan_to_num(
            states[self.state_features].to_numpy(dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-8
        return torch.tensor((X - self.mean_) / self.std_, dtype=torch.float32)

    def _forward(self, state: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Predicted demand on the log1p scale.

        The response is the difference between the log-response at the actual
        price and at full price, which makes the multiplier exactly one when
        no markdown is applied.
        """
        full = torch.full_like(price, FULL_PRICE)
        at_price = self.response(torch.cat([state, price[:, None]], dim=1))
        at_full = self.response(torch.cat([state, full[:, None]], dim=1))
        return self.baseline(state) + (at_price - at_full)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[np.ndarray] = None,
            epochs: int = 150, batch_size: int = 512) -> "StructuralDemandModel":
        state = self._state(X_train, fit=True)
        price = torch.tensor(X_train[ACTION_FEATURE].to_numpy(dtype=np.float32))
        target = torch.tensor(np.log1p(np.asarray(y_train, dtype=np.float32)))

        parameters = list(self.baseline.parameters()) + list(self.response.parameters())
        optimiser = torch.optim.Adam(parameters, lr=self.learning_rate)
        criterion = nn.MSELoss()
        self.history = []

        for _ in range(epochs):
            permutation = torch.randperm(len(state))
            total = 0.0
            for start in range(0, len(state), batch_size):
                batch = permutation[start:start + batch_size]
                optimiser.zero_grad()
                loss = criterion(self._forward(state[batch], price[batch]), target[batch])
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)
            self.history.append(total / len(state))

        logger.info("Structural model: %d epochs, loss %.4f -> %.4f",
                    epochs, self.history[0], self.history[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.baseline.eval(); self.response.eval()
        with torch.no_grad():
            state = self._state(X)
            price = torch.tensor(X[ACTION_FEATURE].to_numpy(dtype=np.float32))
            prediction = np.expm1(self._forward(state, price).numpy())
        self.baseline.train(); self.response.train()
        return np.clip(prediction, 0.0, None)

    def predict_under_action(self, X: pd.DataFrame, discount: float) -> np.ndarray:
        counterfactual = X.copy()
        counterfactual[ACTION_FEATURE] = discount
        return self.predict(counterfactual)

    def demand_curve(self, X: pd.DataFrame,
                     action_grid: Sequence[float] = tuple(ACTION_GRID)) -> np.ndarray:
        return np.column_stack([self.predict_under_action(X, a) for a in action_grid])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"baseline": self.baseline.state_dict(),
                    "response": self.response.state_dict(),
                    "mean": self.mean_, "std": self.std_,
                    "features": self.feature_names}, path)