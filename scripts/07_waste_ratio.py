"""Recover the waste penalty from the operator's own pricing.

The penalty is currently a design parameter chosen because it differentiates
the policy. It can instead be estimated: assuming the operator prices roughly
rationally, the value of waste_ratio that best reproduces the observed
discounts is a revealed-preference estimate of what a write-off costs them.

This is inverse reinforcement learning reduced to a single parameter, which
is all the data supports.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import ACTION_GRID, MODELS_DIR, RESULTS_DIR, STOCK_MULTIPLIER
from src.data import build_feature_matrix, build_recovered_dataset, get_feature_names, temporal_split
from src.demand_model import LGBMDemandModel
from src.reward import stock_of

grid = np.asarray(ACTION_GRID)

matrix = build_feature_matrix(build_recovered_dataset())
features = get_feature_names(matrix)
train, valid, evaluation = temporal_split(matrix)

model = LGBMDemandModel.load(MODELS_DIR / "simulator_latent_demand.pkl")
demand = model.demand_curve(train[features])
stock = stock_of(train)[:, None]

# What the operator actually did, snapped to the grid.
observed = np.abs(train["discount"].to_numpy()[:, None] - grid[None, :]).argmin(axis=1)

sold = np.minimum(demand, stock)
waste = np.maximum(stock - sold, 0.0)
revenue = grid[None, :] * sold

rows = []
for waste_ratio in np.arange(0.0, 2.01, 0.05):
    implied = (revenue - waste_ratio * waste).argmax(axis=1)
    rows.append({
        "waste_ratio": waste_ratio,
        "action_match": float((implied == observed).mean()),
        "mean_abs_gap": float(np.abs(grid[implied] - grid[observed]).mean()),
        "mean_implied_price": float(grid[implied].mean()),
    })

table = pd.DataFrame(rows)
best = table.loc[table["mean_abs_gap"].idxmin()]

print(table.round(4).to_string(index=False))
print(f"\nobserved mean price: {grid[observed].mean():.4f}")
print(f"best fit at waste_ratio = {best.waste_ratio:.2f} "
      f"(match {best.action_match:.3f}, gap {best.mean_abs_gap:.4f})")
table.to_csv(RESULTS_DIR / "waste_ratio_irl.csv", index=False)