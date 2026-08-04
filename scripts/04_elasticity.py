"""Price response of the demand simulator after restoring deep discounts.

The earlier support cut at 0.5 removed the region where mean sales reach
2.97 against 1.02 elsewhere. This re-measures the implied elasticity with
those observations back in the training data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from archive import ACTION_GRID, DATA_DIR, MODELS_DIR, ensure_dirs
from archive import LGBMDemandModel
from archive import build_feature_matrix, get_feature_names, temporal_split

ensure_dirs()
grid = np.asarray(ACTION_GRID)

matrix = build_feature_matrix(pd.read_parquet(DATA_DIR / "subset_recovered.parquet"))
features = get_feature_names(matrix)
train, valid, evaluation = temporal_split(matrix)


def arc_elasticity(curve: np.ndarray, i: int, j: int) -> float:
    """Arc elasticity between two grid points, averaged over states."""
    q_j, q_i = curve[:, j].mean(), curve[:, i].mean()
    p_j, p_i = grid[j], grid[i]
    return ((q_j - q_i) / (q_j + q_i)) / ((p_j - p_i) / (p_j + p_i))


for target in ("observed_demand", "latent_demand"):
    model = LGBMDemandModel().fit(
        train[features], train[target].to_numpy(),
        valid[features], valid[target].to_numpy(),
    )
    model.save(MODELS_DIR / f"demand_{target}.pkl")

    curve = model.predict_demand_curve(evaluation[features])
    revenue = curve * grid[None, :]

    print(f"\n=== {target} ===")
    for k, action in enumerate(ACTION_GRID):
        print(f"  {action:.2f}  demand={curve[:, k].mean():.4f}  "
              f"revenue={revenue[:, k].mean():.4f}")

    last = len(ACTION_GRID) - 1
    print(f"  elasticity 1.00->0.90: {arc_elasticity(curve, 0, 2):+.3f}")
    print(f"  elasticity 1.00->0.60: {arc_elasticity(curve, 0, last - 1):+.3f}")
    print(f"  elasticity 1.00->0.45: {arc_elasticity(curve, 0, last):+.3f}")
    print(f"  best action on average: {ACTION_GRID[revenue.mean(0).argmax()]:.2f}")

    chosen = pd.Series([ACTION_GRID[i] for i in revenue.argmax(1)])
    print("  argmax distribution:")
    print(chosen.value_counts().sort_index().to_string())