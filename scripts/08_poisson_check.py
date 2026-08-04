# notebooks/08_poisson_check.py
"""Does the elasticity survive a different model family?

The simulator is a gradient-boosted ensemble with no assumed functional form.
A Poisson GLM assumes one: log E[q] is linear in the covariates, so with the
multiplier entered in logs its coefficient IS the elasticity, with a standard
error attached. If two families this far apart agree, the estimate does not
depend on the choice of learner.

Demand here is normalised and continuous rather than a count, so this is a
quasi-likelihood fit; the point estimate is consistent under the mean
specification alone, which is what is being tested.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data import build_feature_matrix, build_recovered_dataset, get_feature_names, temporal_split

matrix = build_feature_matrix(build_recovered_dataset())
features = get_feature_names(matrix)
train, _, _ = temporal_split(matrix)

covariates = [f for f in features if f != "discount"]
design = pd.concat(
    [
        np.log(train["discount"]).rename("log_price").reset_index(drop=True),
        train[covariates].reset_index(drop=True),
    ],
    axis=1,
)
design = sm.add_constant(design.fillna(design.median()))


for target in ("observed_demand", "latent_demand"):
    y = train[target].to_numpy()
    fit = sm.GLM(y, design, family=sm.families.Poisson()).fit()
    low, high = fit.conf_int().loc["log_price"]
    print(f"{target:16s} elasticity = {fit.params['log_price']:+.4f} "
          f"[{low:+.4f}, {high:+.4f}]  n={len(y)}")