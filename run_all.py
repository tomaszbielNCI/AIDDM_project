"""Sequential runs across cities and simulator families.

Each configuration writes to its own artefact directory, so results
accumulate rather than overwrite. Stock is re-estimated per city before the
experiment, because the reconstruction depends on that city's stockout
pattern and the reward silently falls back to the trailing-mean proxy if the
file is missing.

Intended to run unattended.
"""

import os
import subprocess
import sys
from datetime import datetime

# (city_id, simulator family). The first entry reproduces the verified run
# under the tagged layout, so the sweep starts from a known point.
CONFIGURATIONS = [
    (0, "lgbm"),
    (12, "lgbm"),
    (16, "lgbm"),
    (3, "lgbm"),
    (0, "structural"),
    (12, "structural"),
]

for city, family in CONFIGURATIONS:
    tag = f"city{city}_{family}"
    started = datetime.now()
    print(f"\n=== {tag} — started {started:%H:%M:%S} ===", flush=True)

    environment = dict(
        os.environ,
        AIDDM_RUN=tag,
        AIDDM_CITY=str(city),
        AIDDM_SIMULATOR=family,
    )

    result = subprocess.run(
        [sys.executable, "run_experiment.py"],
        env=environment, check=False,
    )

    elapsed = (datetime.now() - started).total_seconds() / 60
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"=== {tag} — {status} after {elapsed:.1f} min ===", flush=True)
