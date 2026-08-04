# AIDDM - AI-based Dynamic Demand Management

This project implements reinforcement learning and decision-focused learning methods for dynamic pricing in retail. The system uses the FreshRetailNet-50K dataset to train agents that optimize pricing decisions through learned demand response patterns.

## Dataset Attribution

This project uses the FreshRetailNet-50K dataset from Dingdong-Inc, licensed under CC BY 4.0. The dataset is described in the paper available at arXiv:2505.16319.

## Environment

This project runs in the Anaconda base environment with the following pre-installed dependencies:
- torch 2.11.0
- torchvision 0.26.0
- numpy 1.26.4
- pandas 2.1.4
- pyarrow 14.0.2
- scikit-learn 1.4.2
- lightgbm 4.6.0
- optuna 4.7.0
- shap 0.44.1
- matplotlib 3.7.5
- seaborn 0.13.2

The only additional packages required are:
- gymnasium
- stable-baselines3
- sb3-contrib

## Project Structure

```
C:\python\AIDDM_project\
├── src\
│   ├── config.py        # paths, constants, RANDOM_SEED, action grid
│   ├── data.py          # HF parquet download, subset filtering, local cache
│   ├── censoring.py     # latent demand recovery from hourly stockout flags
│   ├── features.py      # feature engineering, time-based train/test split
│   ├── env.py           # Gymnasium environment wrapping the demand simulator
│   ├── agents\
│   │   ├── value_iteration.py   # tabular VI on a discretised MDP
│   │   ├── ppo_agent.py         # PPO training and rollout
│   │   └── dfl.py               # decision-focused learning, SPO+ loss
│   ├── baselines.py     # fixed price, rule-based heuristic, LGBM predict-then-optimise
│   └── evaluate.py      # metrics, bootstrap CIs, comparison tables, plots
├── notebooks\
│   ├── 01_eda.ipynb
│   └── 02_results.ipynb
├── data\                # gitignored
├── artifacts\           # gitignored: models, result tables, figures
└── tests\
    └── test_env.py      # sanity checks on the Gymnasium env contract
```
