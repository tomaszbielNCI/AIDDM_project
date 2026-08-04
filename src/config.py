"""Project constants: paths, data selection, action space, reward parameters.

Every magic number in the project lives here, with the measurement that
justifies it. Values without a stated reason are defaults, not findings.
"""


import os
from pathlib import Path
from typing import List, Optional

# --- run identity ---------------------------------------------------------

# Every artefact path carries the run tag, so runs over different cities or
# simulator families accumulate instead of overwriting one another. Set from
# the environment by run_all.py; "default" when run by hand.
RUN_TAG: str = os.environ.get("AIDDM_RUN", "default")

# Which demand model family the run uses: "lgbm" or "structural".
SIMULATOR_FAMILY: str = os.environ.get("AIDDM_SIMULATOR", "lgbm")


# --- paths ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models" / RUN_TAG
RESULTS_DIR = ARTIFACTS_DIR / "results" / RUN_TAG
FIGURES_DIR = ARTIFACTS_DIR / "figures" / RUN_TAG

TRAIN_PARQUET = RAW_DATA_DIR / "data" / "train.parquet"
EVAL_PARQUET = RAW_DATA_DIR / "data" / "eval.parquet"

HF_REPO_ID = "Dingdong-Inc/FreshRetailNet-50K"
HF_REPO_TYPE = "dataset"

RANDOM_SEED: int = 42

# --- subset selection -----------------------------------------------------

# City 0 holds 2.32M of the 4.5M training rows. Category is a model feature
# rather than a filter: restricting to one first_category_id leaves 34
# store-product pairs, too few to identify price response across eight levels.
FILTER_CITY_ID: int = int(os.environ.get("AIDDM_CITY", 0))
FILTER_FIRST_CATEGORY_ID: Optional[int] = None
N_STORE_PRODUCT_PAIRS: int = 400

# Cached data carries the city, not the run tag: two runs over the same city
# with different simulator families share the same subset.
SUBSET_PATH = DATA_DIR / f"subset_city{FILTER_CITY_ID}.parquet"
STOCK_PATH = DATA_DIR / f"subset_with_stock_city{FILTER_CITY_ID}.parquet"



# --- action space ---------------------------------------------------------

# Support of the observed multiplier. The lower bound is deliberately not a
# distributional percentile: mean sales below 0.5 reach 2.97 against 1.02
# elsewhere, so a percentile cut would remove the most price-responsive part
# of the demand curve. Above 1.0 there are 34 records in the entire dataset
# and none in city 0, so price increases stay outside the action space.
DISCOUNT_MIN: float = 0.08
DISCOUNT_MAX: float = 1.0

ACTION_GRID: List[float] = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.45]

# --- temporal split -------------------------------------------------------

# Validation is carved from the tail of the training window; the evaluation
# horizon arrives as a separate file and is never used for model selection.
VALIDATION_DAYS: int = 14

# --- MDP ------------------------------------------------------------------

EPISODE_LENGTH: int = 30  # days; the decision step is one day
HOURS_PER_DAY: int = 24  # hourly data serves censoring recovery only
DISCOUNT_FACTOR: float = 0.95  # gamma

# --- reward ---------------------------------------------------------------

# Measured elasticity stays above -1 across the whole grid, so revenue alone
# makes full price optimal in every state and the pricing problem collapses.
# For perishable goods the alternative to a discounted sale is a write-off,
# which is why the perishable-pricing literature uses a multi-component
# reward. Only the difference (WASTE_RATIO - UNIT_COST_RATIO) affects the
# argmax; the sensitivity sweep in notebooks/05 sets both.
UNIT_COST_RATIO: float = 0.0
WASTE_RATIO: float = 0.8
STOCK_MULTIPLIER: float = 1.2  # stock as a multiple of recent mean demand

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def ensure_dirs() -> None:
    """Create output directories. Called from entry points, not at import."""
    for directory in (RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)