"""Does the model treat price as a global additive effect, or does it vary?

If SHAP for the discount feature is wide and differs systematically across
products, the model has learnt product-specific price response and there is
something for a policy to learn. If it is narrow and flat, the optimal action
is the same everywhere and the problem needs a capacity or margin term to
become non-trivial.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import shap

from archive import MODELS_DIR
from archive import LGBMDemandModel
from archive import build_feature_matrix, get_feature_names, temporal_split

fm = build_feature_matrix(pd.read_parquet("data/subset_recovered.parquet"))
feats = get_feature_names(fm)
train, valid, evaluation = temporal_split(fm)

model = LGBMDemandModel.load(MODELS_DIR / "demand_latent_demand.pkl")

X = evaluation[feats].reset_index(drop=True)
shap_values = shap.TreeExplainer(model.model).shap_values(X)
discount_shap = pd.Series(shap_values[:, feats.index("discount")])

print("SHAP(discount) across all evaluation rows:")
print(discount_shap.describe().round(4).to_string())

frame = evaluation.reset_index(drop=True).assign(shap=discount_shap.values)

print("\nWithin-pair variation of SHAP(discount):")
print(frame.groupby(["store_id", "product_id"])["shap"].std().describe().round(4).to_string())

print("\nMean SHAP(discount) by price level:")
bucket = pd.cut(frame["discount"], [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01])
print(frame.groupby(bucket, observed=True)["shap"].agg(["size", "mean", "std"]).round(4).to_string())

print("\nMean SHAP(discount) by category:")
print(frame.groupby("first_category_id")["shap"].agg(["size", "mean"]).round(4).head(12).to_string())