"""Diagnostics for the stock reconstruction.

The reconstruction itself lives in src.data.estimate_stock and runs as part
of the pipeline; this notebook only inspects what it produced.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import STOCK_MULTIPLIER
from src.data import PAIR_KEYS, build_recovered_dataset

daily = build_recovered_dataset()

proxy = STOCK_MULTIPLIER * daily.groupby(list(PAIR_KEYS))["observed_demand"].transform(
    lambda s: s.shift(1).rolling(7, min_periods=1).mean()
)

print(daily[["estimated_stock", "observed_demand", "latent_demand"]].describe().round(3).to_string())
print(f"\nobserved on {daily['stock_is_observed'].mean():.1%} of days")
print(f"correlation with the trailing-mean proxy: {daily['estimated_stock'].corr(proxy):.3f}")
print(f"days where stock falls below realised sales: "
      f"{(daily['estimated_stock'] < daily['observed_demand'] - 1e-9).mean():.4%}")

known = daily[daily["stock_is_observed"] & (daily["latent_demand"] > 0)]
ratio = known["estimated_stock"] / known["latent_demand"]
print(f"\nstock-to-demand ratio on stockout days: median {ratio.median():.3f}, "
      f"IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f}")

print("\n" + daily.groupby("stock_is_observed")[["estimated_stock", "observed_demand"]]
      .mean().round(3).to_string())