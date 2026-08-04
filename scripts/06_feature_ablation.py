"""SHAP-driven feature selection for the demand simulator.

Gain counts splits and favours high-cardinality continuous features, which
inflates rolling means against a discount snapped to eight levels. SHAP
measures contribution to individual predictions, so it ranks by influence on
the output rather than by how often a tree happened to split on the column.

Selection is by SHAP; the ablation then tests whether a smaller model
predicts as well and — the quantity that actually matters for a simulator —
whether it responds to price more sharply.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score

from src.demand_model import LGBMDemandModel
from src.config import ACTION_GRID, RESULTS_DIR, ensure_dirs
from src.data import (
    ACTION_FEATURE,
    build_feature_matrix,
    build_recovered_dataset,
    get_feature_names,
    temporal_split,
)
ensure_dirs()
grid = np.asarray(ACTION_GRID)
TARGET = "latent_demand"

matrix = build_feature_matrix(build_recovered_dataset())
all_features = get_feature_names(matrix)
train, valid, evaluation = temporal_split(matrix)


def fit(features: list[str]) -> LGBMDemandModel:
    return LGBMDemandModel().fit(
        train[features], train[TARGET].to_numpy(),
        valid[features], valid[TARGET].to_numpy(),
    )


def elasticity(curve: np.ndarray) -> float:
    """Arc elasticity between full price and the deepest grid point."""
    q_1, q_0 = curve[:, -1].mean(), curve[:, 0].mean()
    return ((q_1 - q_0) / (q_1 + q_0)) / ((grid[-1] - grid[0]) / (grid[-1] + grid[0]))


def evaluate(features: list[str], label: str) -> dict:
    model = fit(features)
    curve = model.demand_curve(evaluation[features])
    chosen = pd.Series(grid[(curve * grid).argmax(1)])
    p = chosen.value_counts(normalize=True)
    return {
        "variant": label,
        "n_features": len(features),
        "r2": r2_score(evaluation[TARGET], model.predict(evaluation[features])),
        "elasticity": elasticity(curve),
        "entropy": float(-(p * np.log(p)).sum()),
    }


# --- 1. rank by SHAP -----------------------------------------------------

print("Ranking features by mean |SHAP| ...")
full = fit(all_features)
sample = valid[all_features].reset_index(drop=True)
shap_values = shap.TreeExplainer(full.model).shap_values(sample)

shap_share = pd.Series(np.abs(shap_values).mean(0), index=all_features)
shap_share /= shap_share.sum()

gain = full.feature_importance()
gain /= gain.sum()

ranking = pd.DataFrame({"shap": shap_share, "gain": gain})
ranking["rank_shap"] = ranking["shap"].rank(ascending=False).astype(int)
ranking["rank_gain"] = ranking["gain"].rank(ascending=False).astype(int)
ranking["rank_shift"] = ranking["rank_gain"] - ranking["rank_shap"]
ranking = ranking.sort_values("shap", ascending=False)

print(ranking.round(4).to_string())
ranking.to_csv(RESULTS_DIR / "feature_ranking.csv")
print(f"\ndiscount: SHAP rank {ranking.loc[ACTION_FEATURE, 'rank_shap']}, "
      f"gain rank {ranking.loc[ACTION_FEATURE, 'rank_gain']}")

# --- 2. ablate by SHAP rank ---------------------------------------------

print("\nRefitting on the top-k features by SHAP:")
results = [evaluate(all_features, "all")]

for k in (5, 8, 12, 16, 24):
    top = ranking.head(k).index.tolist()
    if ACTION_FEATURE not in top:          # a simulator without price is useless
        top = [ACTION_FEATURE] + top[:-1]
    results.append(evaluate(top, f"top{k}_shap"))

table = pd.DataFrame(results).set_index("variant")
print(table.round(4).to_string())
table.to_csv(RESULTS_DIR / "feature_ablation.csv")

print("\nFeatures in the best-elasticity variant:")
best = table["elasticity"].idxmin()          # most negative
if best != "all":
    k = int(best.replace("top", "").replace("_shap", ""))
    print(ranking.head(k).index.tolist())