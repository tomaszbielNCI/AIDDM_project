# AIDDM - AI-based Dynamic Demand Management

This project implements reinforcement learning and decision-focused learning methods for dynamic pricing in retail. The system uses the FreshRetailNet-50K dataset to train agents that optimize pricing decisions through learned demand response patterns.

## Dataset Attribution

This project uses the FreshRetailNet-50K dataset from Dingdong-Inc, licensed under CC BY 4.0. The dataset is described in the paper available at arXiv:2505.16319.

## Installation

Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

Run experiments for a single city and simulator family:

```bash
python run_experiment.py
```

Run experiments across multiple cities and simulator families:

```bash
python run_all.py
```

Environment variables control the run configuration:
- `AIDDM_RUN`: Run tag for artifact directory naming (default: "default")
- `AIDDM_CITY`: City ID to filter data (default: "0")
- `AIDDM_SIMULATOR`: Simulator family - "lgbm" or "structural" (default: "lgbm")

## Project Structure

```
C:\python\AIDDM_project\
├── src\
│   ├── config.py           # paths, constants, action grid, run configuration
│   ├── data.py             # data pipeline: download, subset, censoring recovery, stock estimation
│   ├── demand_model.py     # LGBM demand model fitting and prediction
│   ├── env.py              # Gymnasium environment for RL agents
│   ├── evaluate.py         # policy scoring, bootstrap CIs, comparison tables
│   ├── policies.py         # all pricing policies (baselines, VI, PPO, decision-focused)
│   ├── reward.py           # reward function with waste penalty
│   └── structural.py       # structural demand model alternative
├── scripts\                # analysis and validation scripts
│   ├── 00_recon.py         # censoring recovery validation
│   ├── 01_shap_check.py    # SHAP feature importance
│   ├── 02_subset_validity.py
│   ├── 03_discount_edges.py
│   ├── 04_elasticity.py    # price elasticity estimation
│   ├── 05_reward_design.py
│   ├── 06_feature_ablation.py
│   ├── 07_stock_estimation.py
│   ├── 07_waste_ratio.py
│   └── 08_poisson_check.py
├── notebooks\
│   ├── 01_eda.ipynb        # exploratory data analysis
│   └── 02_results.ipynb    # results visualization and reporting
├── run_experiment.py      # main experiment script
├── run_all.py              # batch run across cities and simulators
├── data\                   # gitignored: raw parquet files
├── artifacts\              # gitignored: models, results, figures per run
└── tests\
    └── test_env.py         # Gymnasium environment contract tests
```

## Algorithms Implemented

### Baselines
- **FixedPrice**: Constant price multiplier
- **HistoricalPolicy**: Replay operator's historical pricing
- **RuleBased**: Markdown on weekends and holidays
- **PredictThenOptimise**: LGBM demand prediction + reward maximization
- **Clairvoyant**: Upper bound using the evaluation simulator

### Reinforcement Learning
- **TabularValueIteration**: Value iteration on discretized MDP state space
- **PPOAgent**: Proximal Policy Optimization over learned environment

### Decision-Focused Learning
- **DecisionFocused (MSE)**: Prediction-first control arm
- **DecisionFocused (SPO+)**: SPO+ decision regret surrogate
- **DecisionFocused (Perturbed)**: Perturbed optimizer with Gaussian smoothing

## Key Features

- **Censoring Recovery**: Reconstructs latent demand from stockout annotations using intraday profiles
- **Stock Estimation**: Reconstructs daily stock levels from stockout patterns
- **Multiple Simulator Families**: LGBM and structural demand models
- **Run Tagging**: Artifacts organized by run tag for multi-city experiments
- **Bootstrap Uncertainty**: Paired bootstrap over store-product pairs for policy comparison
- **Action Entropy**: Measures policy diversity to detect collapse to single action
