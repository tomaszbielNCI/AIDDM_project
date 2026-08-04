"""Data pipeline: download, subset, censoring recovery, features, split.

Source: HuggingFace repo "Dingdong-Inc/FreshRetailNet-50K", read as parquet
via huggingface_hub. The `datasets` library is avoided deliberately — it
requires pyarrow>=15 and would break the pyarrow 14 / pandas 2.1 pairing in
this environment.

Schema: identifiers (city, store, four category levels, product), dt,
sale_amount (normalised daily volume), hours_sale and hours_stock_status
(24-element sequences), stock_hour6_22_cnt, discount (PRICE MULTIPLIER: 1.0
is full price), holiday_flag, activity_flag, and four weather columns. There
is no price and no revenue column.

Censoring: observed sales are censored from above whenever a product is out
of stock, so a zero in a flagged hour records unavailability rather than
absence of demand. The dataset annotates stock status per hour, so the
mechanism is known rather than assumed.
Mediators: the price multiplier acts on demand partly through the promotion
regime it defines. Any feature that is a deterministic function of today's
multiplier — or that flags the campaign the multiplier belongs to — sits on
that path, and including it in the demand model suppresses the estimated
price response. Such features are excluded; see NON_FEATURE_COLUMNS.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from src.config import (
    DISCOUNT_MAX,
    DISCOUNT_MIN,
    EVAL_PARQUET,
    FILTER_CITY_ID,
    FILTER_FIRST_CATEGORY_ID,
    HF_REPO_ID,
    HF_REPO_TYPE,
    HOURS_PER_DAY,
    N_STORE_PRODUCT_PAIRS,
    RAW_DATA_DIR,
    STOCK_PATH,
    SUBSET_PATH,
    TRAIN_PARQUET,
    VALIDATION_DAYS,
)

logger = logging.getLogger(__name__)

PAIR_KEYS: Tuple[str, str] = ("store_id", "product_id")
DAY_KEYS: Tuple[str, str, str] = ("store_id", "product_id", "dt")
PROFILE_KEYS: Tuple[str, str, str] = ("store_id", "product_id", "weekday")

TRADING_START, TRADING_END = 6, 22  # end exclusive
N_TRADING_HOURS = TRADING_END - TRADING_START

ACTION_FEATURE: str = "discount"

CATEGORICAL_FEATURES: List[str] = [
    "store_id", "product_id", "first_category_id", "second_category_id", "weekday",
]
# Recorded during the day, therefore unknown when the price is chosen. The
# censoring intermediates matter as much as the targets: each is computed from
# the day's realised hourly sales, so together they reconstruct the outcome.
LEAKING_COLUMNS: List[str] = [
    "stock_hour6_22_cnt", "hours_stock_status", "hours_sale", "sale_amount",
    "observed_demand", "latent_demand", "imputed_demand", "censored_hours",
    "recoverable", "observed_uncensored", "profile_uncensored", "profile_censored",
    "estimated_stock", "observed_stock", "stock_is_observed", "pair_ratio",
]
# Promotion indicators are excluded from the demand model. activity_flag
# correlates with the price multiplier at -0.785 and marks the days on which
# the operator runs a campaign, so it lies on the causal path from price to
# demand rather than confounding it. Conditioning on a mediator blocks the
# very effect the simulator has to represent: with the promotion indicators
# in the feature set the implied elasticity is -0.40, without them -1.10, at
# no cost in predictive accuracy (R2 0.662 against 0.670).
NON_FEATURE_COLUMNS = {
    "dt", "split", "city_id", "management_group_id", "third_category_id",
    "activity_flag",
}

# =========================================================================
# Loading
# =========================================================================

def download_raw_data() -> Path:
    """Fetch the parquet files. Idempotent: hashes are verified and skipped."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        allow_patterns=["*.parquet"], local_dir=RAW_DATA_DIR,
    )
    logger.info("Raw data at %s", path)
    return Path(path)


def _read_filtered(
    path: Path,
    city_id: int = FILTER_CITY_ID,
    category_id: Optional[int] = FILTER_FIRST_CATEGORY_ID,
) -> pd.DataFrame:
    """Read with predicate pushdown so the 4.5M-row split never loads."""
    filters = [("city_id", "==", city_id)]
    if category_id is not None:
        filters.append(("first_category_id", "==", category_id))

    df = pd.read_parquet(path, filters=filters)
    logger.info("Read %d rows from %s", len(df), path.name)
    return df


def _select_pairs(df: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    """Keep the pairs with the most promotional days.

    Every pair spans the same 97 days, so row count carries no information.
    Price response is identified by variation in the action, and a pair held
    at full price throughout contributes none. This selects on the treatment
    rather than the outcome, but it does bias the subset toward actively
    promoted products — noted as a limitation.
    """
    score = (
        df.assign(promo=df[ACTION_FEATURE] < 0.99)
        .groupby(list(PAIR_KEYS))
        .agg(promo_days=("promo", "sum"), price_std=(ACTION_FEATURE, "std"))
        .sort_values(["promo_days", "price_std"], ascending=False)
    )
    pairs = score.head(n_pairs).index
    subset = df.set_index(list(PAIR_KEYS)).loc[pairs].reset_index()
    logger.info("Selected %d pairs, %d rows", len(pairs), len(subset))
    return subset


def load_or_create_subset(n_pairs: int = N_STORE_PRODUCT_PAIRS) -> pd.DataFrame:
    """Build the working subset from the raw files, or load the cache.

    Both raw files are combined into one frame carrying a `split` column:
    intraday profiles and lag features need an unbroken series per pair, so
    the train/eval boundary travels as a column rather than as two files.
    """
    if SUBSET_PATH.exists():
        df = pd.read_parquet(SUBSET_PATH)
        logger.info("Loaded cached subset: %d rows", len(df))
        return df

    frames = []
    for split, path in (("train", TRAIN_PARQUET), ("eval", EVAL_PARQUET)):
        part = _read_filtered(path)
        part["split"] = split
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)

    in_support = df[ACTION_FEATURE].between(DISCOUNT_MIN, DISCOUNT_MAX)
    logger.info("Dropped %d rows outside the multiplier support", (~in_support).sum())
    df = df[in_support]

    df = _select_pairs(df, n_pairs)
    df["dt"] = pd.to_datetime(df["dt"])
    df["weekday"] = df["dt"].dt.weekday
    df = df.sort_values([*PAIR_KEYS, "dt"]).reset_index(drop=True)

    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SUBSET_PATH, index=False)
    logger.info("Wrote %s: %d rows", SUBSET_PATH.name, len(df))
    return df


# =========================================================================
# Censoring recovery
# =========================================================================

def explode_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the 24-element sequences to one row per pair, date and hour."""
    keep = [c for c in (*PAIR_KEYS, "dt", "weekday", ACTION_FEATURE, "split")
            if c in df.columns]

    hourly = pd.DataFrame({
        col: np.repeat(df[col].to_numpy(), HOURS_PER_DAY) for col in keep
    })
    hourly["hour"] = np.tile(np.arange(HOURS_PER_DAY), len(df))
    hourly["hourly_sale"] = np.concatenate(df["hours_sale"].to_numpy())
    hourly["is_stockout"] = np.concatenate(
        df["hours_stock_status"].to_numpy()
    ).astype(np.int8)

    logger.info("Exploded %d days into %d hours", len(df), len(hourly))
    return hourly


def _trading(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Hours 6-21. Zeros outside this window are closing time, not demand."""
    return df_hourly[df_hourly["hour"].between(TRADING_START, TRADING_END - 1)]


def _intraday_profile(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean hourly sales over uncensored hours, per pair and weekday.

    Built from uncensored hours only, so the profile is free of censoring by
    construction — though not of selection: an hour stays uncensored partly
    because demand was low enough not to exhaust stock, which leaves the
    profile mildly conservative.
    """
    uncensored = _trading(df_hourly).query("is_stockout == 0")
    return (
        uncensored.groupby([*PROFILE_KEYS, "hour"])["hourly_sale"]
        .mean()
        .unstack("hour")
        .reindex(columns=range(TRADING_START, TRADING_END))
        .fillna(0.0)
    )


def recover_latent_demand(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct daily demand net of stockout censoring.

    Observed sales are kept for every hour, censored or not: a stockout flag
    suppresses demand but does not forbid a sale after restocking, and about
    3% of flagged hours carry sales. Each censored trading hour then receives
    the group profile for that hour, rescaled so the profile mass over the
    day's uncensored hours matches what was actually sold in them. With y the
    observed sales, P the profile, U the uncensored hours and C the censored:

        s      = sum_U y / sum_U P
        latent = sum y + s * sum_C P

    The rescaling is what separates this from plain profile substitution: it
    carries day-specific shocks into the imputed hours instead of flattening
    every day to the group average. It is also why the correction cannot
    steepen the demand curve — being proportional to observed sales, it moves
    the level rather than the slope.

    Returns:
        One row per pair-day with observed_demand, imputed_demand,
        latent_demand, censored_hours and recoverable.
    """
    profile = _intraday_profile(df_hourly)
    long_profile = profile.stack().rename("profile").reset_index()

    trading = _trading(df_hourly).merge(
        long_profile, on=[*PROFILE_KEYS, "hour"], how="left"
    )
    trading["profile"] = trading["profile"].fillna(0.0)

    keys = list(DAY_KEYS)
    uncensored = trading["is_stockout"] == 0

    daily = trading.groupby(keys).agg(
        observed_demand=("hourly_sale", "sum"),
        censored_hours=("is_stockout", "sum"),
    )
    daily["observed_uncensored"] = trading[uncensored].groupby(keys)["hourly_sale"].sum()
    daily["profile_uncensored"] = trading[uncensored].groupby(keys)["profile"].sum()
    daily["profile_censored"] = trading[~uncensored].groupby(keys)["profile"].sum()
    daily = daily.fillna(0.0)

    has_scale = daily["profile_uncensored"] > 0
    scale = np.where(
        has_scale,
        daily["observed_uncensored"] / daily["profile_uncensored"].where(has_scale, 1.0),
        0.0,
    )
    daily["recoverable"] = has_scale
    daily["imputed_demand"] = scale * daily["profile_censored"]
    daily["latent_demand"] = daily["observed_demand"] + daily["imputed_demand"]

    logger.info(
        "Recovered %d days, %d unrecoverable, mean uplift %.4f",
        len(daily), (~daily["recoverable"]).sum(), daily["imputed_demand"].mean(),
    )
    return daily.reset_index()


def flag_unrecoverable(df_daily: pd.DataFrame, min_hours: int = 4) -> pd.DataFrame:
    """Zero the imputation on days with too few uncensored hours.

    The scale factor is a ratio over the uncensored hours; with one or two of
    them it is noise. Flagged days fall back to observed demand rather than
    being dropped — removing high-stockout days would delete exactly the
    observations this project is about and would select on the outcome.
    """
    enough = (N_TRADING_HOURS - df_daily["censored_hours"]) >= min_hours

    out = df_daily.copy()
    out["recoverable"] = out["recoverable"] & enough
    out.loc[~out["recoverable"], "imputed_demand"] = 0.0
    out["latent_demand"] = out["observed_demand"] + out["imputed_demand"]

    logger.info("Flagged %d of %d days unrecoverable", (~out["recoverable"]).sum(), len(out))
    return out

def estimate_stock(df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the stock available each day from the stockout annotations.

    On a day whose first stockout hour is h, everything that was going to sell
    up to h did sell and nothing after: the stock available that day equals
    the sales accumulated up to h. That is an observation rather than an
    estimate, and it covers the days that sell out — 38.8% of the city 0
    subset.

    On the remaining days stock is censored from below: it exceeded that day's
    sales by an unknown margin. It is filled with the larger of the observed
    sales and the pair's typical stocking level, so the estimate is never
    below what demonstrably sold.

    This is the mirror image of demand recovery: there the stockout truncates
    demand, here it reveals supply.

    Returns:
        The daily frame with estimated_stock, observed_stock and
        stock_is_observed added.
    """
    trading = _trading(df_hourly).sort_values([*DAY_KEYS, "hour"]).copy()
    trading["cumulative_sale"] = trading.groupby(list(DAY_KEYS))["hourly_sale"].cumsum()

    first_stockout = (
        trading[trading["is_stockout"] == 1]
        .groupby(list(DAY_KEYS))
        .agg(stockout_hour=("hour", "min"))
    )
    observed = (
        trading.merge(first_stockout, on=list(DAY_KEYS), how="inner")
        .query("hour < stockout_hour")
        .groupby(list(DAY_KEYS))
        .agg(observed_stock=("cumulative_sale", "max"))
        .reset_index()
    )

    out = df_daily.merge(observed, on=list(DAY_KEYS), how="left")

    typical = out.groupby(list(PAIR_KEYS))["observed_stock"].transform("median")
    typical = typical.fillna(out["observed_demand"].median())

    # Sales after a mid-day restock mean the pre-stockout total can understate
    # the stock that passed through the shelf, so the observed branch is
    # floored at the day's realised sales.
    out["stock_is_observed"] = out["observed_stock"].notna()
    out["estimated_stock"] = np.where(
        out["stock_is_observed"],
        np.maximum(out["observed_stock"], out["observed_demand"]),
        np.maximum(out["observed_demand"], typical),
    )

    logger.info(
        "Stock observed on %d of %d days (%.1f%%), mean level %.3f",
        int(out["stock_is_observed"].sum()), len(out),
        100 * out["stock_is_observed"].mean(), out["estimated_stock"].mean(),
    )
    return out

def censoring_bias(df_daily: pd.DataFrame) -> float:
    """Share of true demand that censored sales fail to record.

    Computed over ALL days, including unrecoverable ones. Restricting it to
    the recoverable subset would select on the quantity being measured: a
    stricter threshold removes the most censored days and shrinks the
    estimate from 17.5% to 8.9% without any change in the underlying bias.
    """
    return float(1 - df_daily["observed_demand"].mean() / df_daily["latent_demand"].mean())


def hourly_censoring_rate(df_hourly: pd.DataFrame) -> float:
    """Share of tradeable hours censored — comparable with the authors' ~20%."""
    return float(_trading(df_hourly)["is_stockout"].mean())


def daily_censoring_rate(df_hourly: pd.DataFrame) -> float:
    """Share of pair-days with at least one censored hour. Much higher."""
    return float(_trading(df_hourly).groupby(list(DAY_KEYS))["is_stockout"].max().mean())


def build_recovered_dataset() -> pd.DataFrame:
    """Subset joined with recovered demand and reconstructed stock, cached.

    Both reconstructions read the same hourly explosion, so they are computed
    together and cached as one file. The reward silently falls back to a
    trailing-mean proxy when the stock column is absent, which is why this
    step is part of the pipeline rather than a separate script.
    """
    if STOCK_PATH.exists():
        return pd.read_parquet(STOCK_PATH)

    subset = load_or_create_subset()
    hourly = explode_hourly(subset)

    daily = flag_unrecoverable(recover_latent_demand(hourly))
    out = subset.merge(daily, on=list(DAY_KEYS), how="left")
    out = estimate_stock(out, hourly)

    out.to_parquet(STOCK_PATH, index=False)
    logger.info("Wrote %s: %d rows", STOCK_PATH.name, len(out))
    return out


# =========================================================================
# Features
# =========================================================================

def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Weekday twice: as a category for tree splits, as sin/cos for the net."""
    out = df.copy()
    dates = pd.to_datetime(out["dt"])

    out["weekday"] = dates.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(np.int8)
    out["day_of_month"] = dates.dt.day
    out["days_elapsed"] = (dates - dates.min()).dt.days
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    return out


def _add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Rain enters twice: whether it rains at all, and how much (log-scaled)."""
    out = df.copy()
    out["is_rainy"] = (out["precpt"] > 0.5).astype(np.int8)
    out["log_precpt"] = np.log1p(out["precpt"])
    return out


def _add_price_context(df: pd.DataFrame) -> pd.DataFrame:
    """Depth relative to the pair's own past pricing.

    The absolute multiplier does not say whether a price is unusual for this
    product. Both the reference mean and the lagged price use past days only.
    """
    out = df.sort_values([*PAIR_KEYS, "dt"]).copy()
    lagged = out.groupby(list(PAIR_KEYS))[ACTION_FEATURE].shift(1)

    out["discount_lag1"] = lagged
    out["discount_pair_mean"] = (
        lagged.groupby([out[PAIR_KEYS[0]], out[PAIR_KEYS[1]]])
        .expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    # discount_depth and is_promo were dropped here: both are functions of
    # today's multiplier, so a counterfactual query that changes the price
    # while holding them fixed describes a state that cannot occur. Their
    # SHAP contribution was high (discount_depth ranked 4th, above the raw
    # multiplier at 6th) precisely because they carried the price signal the
    # simulator then failed to attribute to the price itself.
    # discount_lag1 and discount_pair_mean are kept: both are built from past
    # days only and are unaffected by the action under evaluation.
    return out


def _add_history(
    df: pd.DataFrame,
    value_col: str = "observed_demand",
    lags: Sequence[int] = (1, 2, 7),
    windows: Sequence[int] = (7, 14),
) -> pd.DataFrame:
    """Lags and rolling statistics, shifted so the current day never enters.

    History comes from OBSERVED demand, not the recovered series: at decision
    time the operator sees what was sold, not what would have sold had stock
    held. Grouping by pair stops one series bleeding into the next.
    """
    out = df.sort_values([*PAIR_KEYS, "dt"]).copy()
    grouped = out.groupby(list(PAIR_KEYS))[value_col]

    for lag in lags:
        out[f"{value_col}_lag{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in windows:
        rolled = (
            shifted.groupby([out[PAIR_KEYS[0]], out[PAIR_KEYS[1]]])
            .rolling(window, min_periods=1)
        )
        out[f"{value_col}_roll{window}_mean"] = (
            rolled.mean().reset_index(level=[0, 1], drop=True)
        )
        out[f"{value_col}_roll{window}_std"] = (
            rolled.std().reset_index(level=[0, 1], drop=True)
        )
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Run the feature steps in order. Targets stay in the frame."""
    out = _add_history(_add_price_context(_add_weather(_add_calendar(df))))
    logger.info("Feature matrix: %d rows, %d columns", len(out), out.shape[1])
    return out


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """Model inputs, action first.

    Assembled by exclusion, so a feature added upstream reaches the model
    without a second edit — while anything in LEAKING_COLUMNS stays out by
    construction.
    """
    excluded = set(LEAKING_COLUMNS) | NON_FEATURE_COLUMNS
    features = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if ACTION_FEATURE not in features:
        raise ValueError(f"{ACTION_FEATURE} missing: the model would be blind to price")
    return [ACTION_FEATURE] + [c for c in features if c != ACTION_FEATURE]


def temporal_split(
    df: pd.DataFrame, validation_days: int = VALIDATION_DAYS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validation and held-out evaluation frames.

    The evaluation horizon is the authors' own seven-day file and travels in
    the `split` column. Validation is the tail of the training window, so
    hyperparameters are never chosen against the test horizon. Random K-fold
    would place a store's future beside its past and inflate every metric.
    """
    dates = pd.to_datetime(df["dt"])
    is_train = df["split"] == "train"
    cutoff = dates[is_train].max() - pd.Timedelta(days=validation_days)

    train = df[is_train & (dates <= cutoff)]
    valid = df[is_train & (dates > cutoff)]
    evaluation = df[df["split"] == "eval"]

    logger.info(
        "Split: train %d (to %s) | valid %d | eval %d",
        len(train), cutoff.date(), len(valid), len(evaluation),
    )
    return train, valid, evaluation