"""One-off reconnaissance of the raw FreshRetailNet-50K files.

Answers the questions that must be settled before choosing a working
subset: what the date coverage is, how the eval split was constructed,
which cities and categories are large enough, and what discount levels
actually occur (which fixes the action grid).
"""

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
TRAIN = "data/raw/data/train.parquet"
EVAL = "data/raw/data/eval.parquet"

# --- 1. date coverage -------------------------------------------------
for name, path in (("train", TRAIN), ("eval", EVAL)):
    dt = pq.read_table(path, columns=["dt"]).column("dt")
    print(f"{name}: {pc.min(dt).as_py()} .. {pc.max(dt).as_py()}  "
          f"({pc.count_distinct(dt).as_py()} distinct dates)")

# --- 2. is eval a temporal split or a series split? -------------------
keys = ["store_id", "product_id"]
tr = pq.read_table(TRAIN, columns=keys).to_pandas().drop_duplicates()
ev = pq.read_table(EVAL, columns=keys).to_pandas().drop_duplicates()
tr_set = set(map(tuple, tr.values))
ev_set = set(map(tuple, ev.values))
print(f"\ntrain pairs: {len(tr_set)}  eval pairs: {len(ev_set)}  "
      f"overlap: {len(tr_set & ev_set)}")

# --- 3. where the data is dense ---------------------------------------
cols = ["city_id", "first_category_id", "store_id", "product_id"]
df = pq.read_table(TRAIN, columns=cols).to_pandas()
print("\nrows per city:")
print(df["city_id"].value_counts().head(10))
print("\ntop city x category blocks (by store-product pairs):")
print(df.groupby(["city_id", "first_category_id"])
        .apply(lambda g: g[["store_id", "product_id"]].drop_duplicates().shape[0])
        .sort_values(ascending=False)
        .head(15))
del df

# --- 4. the action grid, empirically ----------------------------------
d = pq.read_table(TRAIN, columns=["discount"]).column("discount").to_pandas()
print(f"\ndiscount: {len(d)} rows, {d.nunique()} distinct values")
print("\nmost frequent levels:")
print(d.round(3).value_counts().head(20))
print("\nquantiles:")
print(d.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1]))
print(f"\nshare at full price (1.0): {(d == 1.0).mean():.1%}")

# --- 5. which block has usable price variation? ------------------------
cols = ["city_id", "first_category_id", "store_id", "product_id",
        "discount", "sale_amount", "stock_hour6_22_cnt"]
df = pq.read_table(TRAIN, columns=cols, filters=[("city_id", "==", 0)]).to_pandas()

# drop physically impossible records before judging anything
clean = df[(df["discount"] > 0.4) & (df["discount"] <= 1.0)]
print(f"dropped {len(df) - len(clean)} rows ({1 - len(clean)/len(df):.2%}) "
      f"outside [0.4, 1.0]")

block = (clean.groupby("first_category_id")
              .agg(pairs=("product_id", lambda s: s.nunique()),
                   rows=("discount", "size"),
                   promo_share=("discount", lambda s: (s < 0.99).mean()),
                   discount_std=("discount", "std"),
                   mean_sales=("sale_amount", "mean"),
                   stockout_share=("stock_hour6_22_cnt", lambda s: (s > 0).mean()))
              .sort_values("rows", ascending=False))
print(block.head(20))