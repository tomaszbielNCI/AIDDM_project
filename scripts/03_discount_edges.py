# notebooks/03_discount_edges.py
"""What are the discount values outside [0.5, 1.0]?

Two competing readings: recording artefacts, or genuine pricing behaviour
that the action space should represent. The difference matters because a
markdown-to-zero write-off and a missing value look identical in the column
but imply opposite treatment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from archive import TRAIN_PARQUET

df = pd.read_parquet(TRAIN_PARQUET, filters=[("city_id", "==", 0)])
df["dt"] = pd.to_datetime(df["dt"])

zero = df[df.discount == 0]
high = df[df.discount > 1.0]
normal = df[df.discount.between(0.5, 1.0)]

for name, part in (("discount == 0", zero), ("discount > 1.0", high),
                   ("in support", normal)):
    print(f"\n{name}: n={len(part)}")
    if not len(part):
        continue
    print(f"  sale_amount: mean={part.sale_amount.mean():.3f} "
          f"median={part.sale_amount.median():.3f} "
          f"zero_share={(part.sale_amount == 0).mean():.3f}")
    print(f"  stockout hours: mean={part.stock_hour6_22_cnt.mean():.2f} "
          f"any={(part.stock_hour6_22_cnt > 0).mean():.3f}")
    print(f"  activity_flag={part.activity_flag.mean():.3f} "
          f"holiday={part.holiday_flag.mean():.3f}")

print("\ndiscount > 1.0 — distribution of values:")
print(high.discount.round(3).value_counts().head(10).to_string())

print("\nposition within each product's own series:")
first_day = df.groupby("product_id")["dt"].transform("min")
df["day_index"] = (df["dt"] - first_day).dt.days
for name, mask in (("zero", df.discount == 0), ("high", df.discount > 1.0),
                   ("support", df.discount.between(0.5, 1.0))):
    print(f"  {name}: median day_index = {df.loc[mask, 'day_index'].median():.0f}")

print("\nday-before and day-after discount around zero-discount days:")
df = df.sort_values(["store_id", "product_id", "dt"])
g = df.groupby(["store_id", "product_id"])["discount"]
df["prev"], df["next"] = g.shift(1), g.shift(-1)
print(df.loc[df.discount == 0, ["prev", "next"]].describe().round(3).to_string())

# dopisz do 03_discount_edges.py
deep = df[(df.discount > 0) & (df.discount < 0.5)]
print(f"\ndeep discounts (0, 0.5): n={len(deep)}")
print(deep.discount.describe().round(3).to_string())
print(f"  sale_amount: mean={deep.sale_amount.mean():.3f} "
      f"vs support {normal.sale_amount.mean():.3f}")
print(f"  stockout any={(deep.stock_hour6_22_cnt > 0).mean():.3f}")
print(f"  activity_flag={deep.activity_flag.mean():.3f}")
print("\n  distribution:")
print(pd.cut(deep.discount, [0, .1, .2, .3, .4, .5]).value_counts().sort_index().to_string())
print("\n  pairs affected:", deep.groupby(['store_id','product_id']).ngroups)