# notebooks/02_subset_validity.py
"""Is the censoring bias a property of the data or of our pair selection?

The working subset keeps the 400 pairs with the most promotional days.
High-promotion items tend to be fast movers, which stock out more often, so
the measured bias may be inflated by selection rather than by the data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from archive import RANDOM_SEED, TRAIN_PARQUET
from archive import flag_unrecoverable_days, recover_latent_demand, censoring_bias
from archive import clean_discount, explode_hourly_data

full = clean_discount(pd.read_parquet(TRAIN_PARQUET, filters=[("city_id", "==", 0)]))
full["dt"] = pd.to_datetime(full["dt"])
full["weekday"] = full["dt"].dt.weekday

rng = np.random.default_rng(RANDOM_SEED)
pairs = full[["store_id", "product_id"]].drop_duplicates()
sample = pairs.sample(400, random_state=RANDOM_SEED)
random_subset = full.merge(sample, on=["store_id", "product_id"])

daily = flag_unrecoverable_days(recover_latent_demand(explode_hourly_data(random_subset)))
print(f"random 400 pairs  bias={censoring_bias(daily):.4f}  "
      f"stockout days={(daily.censored_hours > 0).mean():.3f}")