"""The learned demand simulator.

One model, three roles: transition dynamics for the pricing MDP, the
predictor inside the predict-then-optimise baseline, and the MSE-trained
control arm against which the decision-focused network is compared.

Fitted twice from this one code path — on observed sales and on demand
recovered from stockout censoring. The gap between the policies they induce
is the central experiment.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ACTION_GRID, RANDOM_SEED
from src.data import ACTION_FEATURE, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


class LGBMDemandModel:
    """Gradient-boosted demand model with the price multiplier as a feature.

    The multiplier has to be an input rather than a post-hoc adjustment:
    without it the predicted quantity is invariant to the action, and any
    policy maximising a reward increasing in price collapses to full price.

    The target is log1p-transformed. Daily demand is right-skewed (median
    0.7, maximum 45 on this subset), so squared error on the raw scale would
    let a handful of extreme days dominate the fit.
    """

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 40,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.params = dict(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[np.ndarray] = None,
        patience: int = 50,
    ) -> "LGBMDemandModel":
        """Fit on log1p demand, with early stopping when validation is given."""
        if ACTION_FEATURE not in X_train.columns:
            raise ValueError(f"{ACTION_FEATURE} missing from the feature matrix")

        self.feature_names = list(X_train.columns)
        categoricals = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

        eval_set, callbacks = None, []
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, np.log1p(y_valid))]
            callbacks = [lgb.early_stopping(patience, verbose=False),
                         lgb.log_evaluation(0)]

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_train, np.log1p(y_train),
            eval_set=eval_set, categorical_feature=categoricals, callbacks=callbacks,
        )

        logger.info(
            "Fitted on %d rows, %d features, best_iteration=%s",
            len(X_train), len(self.feature_names),
            getattr(self.model, "best_iteration_", None),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Demand for the multipliers present in X, back on the original scale."""
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return np.clip(np.expm1(self.model.predict(X[self.feature_names])), 0.0, None)

    def predict_under_action(self, X: pd.DataFrame, discount: float) -> np.ndarray:
        """Demand under a counterfactual multiplier.

        The interventional query the environment issues at every step. Its
        credibility rests on overlap: where the behaviour policy rarely took
        an action, the model extrapolates and an offline agent will find it.
        """
        counterfactual = X.copy()
        counterfactual[ACTION_FEATURE] = discount
        return self.predict(counterfactual)

    def demand_curve(
        self, X: pd.DataFrame, action_grid: Sequence[float] = tuple(ACTION_GRID)
    ) -> np.ndarray:
        """Demand at every action level, shape (len(X), n_actions)."""
        return np.column_stack([self.predict_under_action(X, a) for a in action_grid])

    def feature_importance(self) -> pd.Series:
        """Gain-based importance, descending."""
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return pd.Series(
            self.model.booster_.feature_importance(importance_type="gain"),
            index=self.feature_names,
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"model": self.model, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: Path) -> "LGBMDemandModel":
        blob = pd.read_pickle(path)
        instance = cls()
        instance.model, instance.feature_names = blob["model"], blob["features"]
        return instance