"""Does a waste-aware reward move the optimal price away from full price?

Pure revenue leaves the optimum trivial: measured elasticity stays above -1
across the whole action grid, so any markdown reduces revenue. For perishable
goods the relevant alternative to a discounted sale is not a full-price sale
but a write-off, which is why the literature on perishable pricing uses a
multi-component reward rather than revenue alone.

Reward per state and action:
    sold   = min(demand(a), stock)
    reward = a * sold - cost_ratio * sold - waste_ratio * (stock - sold)

Stock is unobserved and approximated as a multiple of the pair's recent mean
demand, the only supply signal the data supports. Both revenue and cost scale
with the unknown per-SKU base price, so it cancels from the argmax.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from archive import ACTION_GRID, DATA_DIR, MODELS_DIR
from archive import LGBMDemandModel
from archive import build_feature_matrix, get_feature_names, temporal_split

grid = np.asarray(ACTION_GRID)

matrix = build_feature_matrix(pd.read_parquet(DATA_DIR / "subset_recovered.parquet"))
features = get_feature_names(matrix)
train, valid, evaluation = temporal_split(matrix)

model = LGBMDemandModel.load(MODELS_DIR / "demand_latent_demand.pkl")
demand = model.predict_demand_curve(evaluation[features])

baseline = evaluation["observed_demand_roll7_mean"].fillna(
    evaluation["observed_demand_roll7_mean"].mean()
).to_numpy()


def reward_matrix(cost_ratio: float, waste_ratio: float,
                  stock_multiplier: float) -> np.ndarray:
    """Reward per state and action under the waste-aware formulation."""
    stock = (stock_multiplier * baseline)[:, None]
    sold = np.minimum(demand, stock)
    return grid[None, :] * sold - cost_ratio * sold - waste_ratio * (stock - sold)


print("Sensitivity of the optimal action to the reward parameters\n")
print(f"{'cost':>6} {'waste':>6} {'stock':>6}  {'best':>6}  {'entropy':>8}  distribution")

for cost_ratio in (0.0, 0.3, 0.5):
    for waste_ratio in (0.0, 0.3, 0.5, 0.8):
        for stock_multiplier in (1.2,):
            R = reward_matrix(cost_ratio, waste_ratio, stock_multiplier)
            best = grid[R.mean(0).argmax()]
            chosen = pd.Series(grid[R.argmax(1)])
            p = chosen.value_counts(normalize=True)
            entropy = float(-(p * np.log(p)).sum())
            top = ", ".join(f"{a:.2f}:{n}" for a, n in
                            chosen.value_counts().sort_index().items())
            print(f"{cost_ratio:6.1f} {waste_ratio:6.1f} {stock_multiplier:6.1f}  "
                  f"{best:6.2f}  {entropy:8.3f}  {top}")

print("\nStock multiplier sweep at cost=0.5, waste=0.5:")
for stock_multiplier in (0.8, 1.0, 1.2, 1.5, 2.0):
    R = reward_matrix(0.5, 0.5, stock_multiplier)
    chosen = pd.Series(grid[R.argmax(1)])
    p = chosen.value_counts(normalize=True)
    print(f"  stock={stock_multiplier:.1f}  best={grid[R.mean(0).argmax()]:.2f}  "
          f"entropy={-(p * np.log(p)).sum():.3f}  "
          f"mean_price={chosen.mean():.3f}")