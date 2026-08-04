"""Full pipeline plus every artefact the report needs.

Run from the project root. Configuration comes from the environment, so a
sweep over cities and simulator families needs no edits here:

    AIDDM_RUN        artefact directory name, e.g. "city0_lgbm"
    AIDDM_CITY       city_id to filter on
    AIDDM_SIMULATOR  "lgbm" or "structural"

Writes to artifacts/results/<AIDDM_RUN>/:

  dataset_summary.csv        subset size, censoring rates, censoring bias
  simulator_diagnostics.csv  R2, WMAPE and implied elasticity, both targets
  mediator_ablation.csv      elasticity with and without promotion indicators
  feature_ranking.csv        gain against mean absolute SHAP (tree runs only)
  demand_curve.csv           mean demand and reward at every action level
  vi_residuals.csv           Bellman residual per sweep, against gamma^n
  anchor_sweep.csv           decision-loss variants across anchor weights
  dfl_<loss>_history.csv     training loss per epoch, three variants
  policy_comparison.csv      mean reward, bootstrap interval, entropy, lift
  paired_differences.csv     paired bootstrap of each policy against the operator
  action_distribution.csv    chosen multiplier frequencies per policy
  reward_sensitivity.csv     optimal action under a sweep of the waste penalty
  per_state_<policy>.csv     per-state detail for the strongest policies
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import (
    ACTION_GRID, FILTER_CITY_ID, LOG_FORMAT, MODELS_DIR, RESULTS_DIR, RUN_TAG,
    SIMULATOR_FAMILY, UNIT_COST_RATIO, ensure_dirs,
)
from src.data import (
    build_feature_matrix, build_recovered_dataset, censoring_bias,
    daily_censoring_rate, explode_hourly, get_feature_names,
    hourly_censoring_rate, load_or_create_subset, temporal_split,
)
from src.demand_model import LGBMDemandModel
from src.env import PricingEnv
from src.evaluate import (
    action_entropy, compare_policies, paired_bootstrap, score_policy,
)
from src.policies import (
    Clairvoyant, DecisionFocused, DecisionLoss, FixedPrice, HistoricalPolicy,
    PPOAgent, PredictThenOptimise, RuleBased, TabularValueIteration,
)
from src.reward import reward_curve, stock_of

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("run")

# The evaluation simulator is fitted on recovered demand, the policies on
# observed sales. The gap between them is what censoring costs a pricing
# system, and it is what makes the clairvoyant bound meaningful.
EVAL_TARGET = "latent_demand"
POLICY_TARGET = "observed_demand"

PPO_TIMESTEPS = 100_000
DFL_EPOCHS = 200
GRID = np.asarray(ACTION_GRID)


# =========================================================================
# Helpers
# =========================================================================

def fit_simulator(train, valid, features, target):
    """Fit the demand model of the family selected for this run.

    The structural variant constrains price to a separate multiplicative
    channel, q = f(state) * g(price, state); the tree ensemble leaves the
    functional form free. Comparing the two is the check that the estimated
    elasticity is a property of the data rather than of the learner.
    """
    if SIMULATOR_FAMILY == "structural":
        from src.structural import StructuralDemandModel
        return StructuralDemandModel(features).fit(
            train[features], train[target].to_numpy(),
            valid[features], valid[target].to_numpy(),
        )

    return LGBMDemandModel().fit(
        train[features], train[target].to_numpy(),
        valid[features], valid[target].to_numpy(),
    )


def arc_elasticity(curve: np.ndarray, i: int = 0, j: int = -1) -> float:
    """Arc elasticity between two points of the action grid."""
    q_j, q_i = curve[:, j].mean(), curve[:, i].mean()
    p_j, p_i = GRID[j], GRID[i]
    return float(((q_j - q_i) / (q_j + q_i)) / ((p_j - p_i) / (p_j + p_i)))


def describe_dataset() -> None:
    """Subset size and the censoring figures quoted in the report."""
    subset = load_or_create_subset()
    hourly = explode_hourly(subset)
    daily = build_recovered_dataset()

    summary = pd.Series({
        "city_id": FILTER_CITY_ID,
        "rows": len(subset),
        "pairs": subset.groupby(["store_id", "product_id"]).ngroups,
        "stores": subset["store_id"].nunique(),
        "products": subset["product_id"].nunique(),
        "days": subset["dt"].nunique(),
        "promo_share": float((subset["discount"] < 0.99).mean()),
        "campaign_price_correlation": float(
            subset["activity_flag"].corr(subset["discount"])
        ),
        "hourly_censoring_rate": hourly_censoring_rate(hourly),
        "daily_censoring_rate": daily_censoring_rate(hourly),
        "censoring_bias": censoring_bias(daily),
        "zero_sale_share": float((subset["sale_amount"] == 0).mean()),
        "stock_observed_share": float(daily["stock_is_observed"].mean())
        if "stock_is_observed" in daily.columns else np.nan,
    })
    summary.to_csv(RESULTS_DIR / "dataset_summary.csv", header=False)
    logger.info("Dataset: %s", summary.round(4).to_dict())


def diagnose_simulator(model, evaluation, features, target) -> dict:
    """Predictive accuracy and implied price response on the held-out horizon."""
    predicted = model.predict(evaluation[features])
    actual = evaluation[target].to_numpy()
    curve = model.demand_curve(evaluation[features])

    return {
        "family": SIMULATOR_FAMILY,
        "target": target,
        "r2": r2_score(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        # WMAPE rather than MAPE: demand is zero on part of the days, which
        # leaves the per-observation percentage error undefined. It is also
        # the metric the dataset authors report, so the two are comparable.
        "wmape": mean_absolute_error(actual, predicted) * len(actual) / actual.sum(),
        "elasticity_full_range": arc_elasticity(curve),
        "elasticity_local": arc_elasticity(curve, 0, 2),
    }


def mediator_ablation(train, valid, evaluation, features) -> None:
    """What conditioning on the promotion indicators costs the elasticity.

    The indicators are excluded from the model by construction, so this
    reconstructs them and refits. Removing any single one changes little,
    because the three are collinear and each carries the same signal; the
    effect appears only when the set goes together. That is the practical
    warning: excluding the obvious campaign flag is not enough, every
    deterministic function of today's price has to go with it.
    """
    rows = []
    for label, add_mediators in (("without_mediators", False), ("with_mediators", True)):
        frames = {"train": train.copy(), "valid": valid.copy(), "eval": evaluation.copy()}
        feature_set = list(features)

        if add_mediators:
            for frame in frames.values():
                past_mean = frame.groupby(["store_id", "product_id"])["discount"].transform(
                    lambda s: s.shift(1).expanding().mean()
                )
                frame["discount_depth"] = frame["discount"] - past_mean
                frame["is_promo"] = (frame["discount"] < 0.99).astype(int)
            feature_set += ["activity_flag", "discount_depth", "is_promo"]

        model = fit_simulator(frames["train"], frames["valid"], feature_set, EVAL_TARGET)
        curve = model.demand_curve(frames["eval"][feature_set])
        rows.append({
            "variant": label,
            "n_features": len(feature_set),
            "r2": r2_score(
                frames["eval"][EVAL_TARGET], model.predict(frames["eval"][feature_set])
            ),
            "elasticity": arc_elasticity(curve),
        })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "mediator_ablation.csv", index=False)
    logger.info("Mediator ablation: %s", rows)


def rank_features(model, valid, features) -> None:
    """Gain against mean absolute SHAP.

    Gain counts splits and favours high-cardinality continuous columns; SHAP
    measures contribution to individual predictions. Computed on validation,
    so feature attribution never sees the evaluation horizon. Tree models
    only — the structural network has no split-based importance.
    """
    import shap

    gain = model.feature_importance()
    gain = (gain / gain.sum()).rename("gain")

    sample = valid[features].reset_index(drop=True)
    shap_values = np.abs(shap.TreeExplainer(model.model).shap_values(sample)).mean(0)
    shap_share = pd.Series(shap_values, index=features)
    shap_share = (shap_share / shap_share.sum()).rename("shap")

    ranking = pd.concat([shap_share, gain], axis=1)
    ranking["rank_shap"] = ranking["shap"].rank(ascending=False).astype(int)
    ranking["rank_gain"] = ranking["gain"].rank(ascending=False).astype(int)
    ranking["rank_shift"] = ranking["rank_gain"] - ranking["rank_shap"]
    ranking.sort_values("shap", ascending=False).to_csv(RESULTS_DIR / "feature_ranking.csv")


def record_demand_curve(model, evaluation, features) -> None:
    """Mean demand, revenue and reward at each action level."""
    demand = model.demand_curve(evaluation[features])
    rewards = reward_curve(demand, stock_of(evaluation), GRID)

    pd.DataFrame({
        "discount": GRID,
        "mean_demand": demand.mean(axis=0),
        "mean_revenue": (demand * GRID[None, :]).mean(axis=0),
        "mean_reward": rewards.mean(axis=0),
    }).to_csv(RESULTS_DIR / "demand_curve.csv", index=False)


def reward_sensitivity(model, evaluation, features) -> None:
    """How the optimal action responds to the waste penalty.

    Only the difference between the waste penalty and the unit cost affects
    the argmax, because the cost of goods is sunk once the stock is bought.
    The sweep documents how much of the resulting policy is driven by that
    single design parameter.
    """
    demand = model.demand_curve(evaluation[features])
    stock = stock_of(evaluation)

    rows = []
    for waste in np.arange(0.0, 1.61, 0.2):
        rewards = reward_curve(demand, stock, GRID, UNIT_COST_RATIO, waste)
        chosen = pd.Series(GRID[rewards.argmax(axis=1)])
        rows.append({
            "waste_ratio": waste,
            "best_action_on_average": GRID[rewards.mean(axis=0).argmax()],
            "mean_chosen_price": chosen.mean(),
            "entropy": action_entropy(chosen),
        })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "reward_sensitivity.csv", index=False)


def anchor_sweep(train, evaluation, features) -> None:
    """How the anchor weight rescues the decision-focused variants.

    Both decision losses are trained at several anchor weights and scored by
    the level of predicted demand and the entropy of the induced policy. The
    contrast separates two distinct failure modes: scale drift, which the
    anchor fixes, and the loss of non-negativity in the SPO+ surrogate under
    a piecewise-linear reward, which it does not.
    """
    rows = []
    for mode in ("spo", "perturbed"):
        for weight in (0.0, 0.1, 0.5, 1.0):
            model = DecisionFocused(features, loss=mode)
            model.criterion = DecisionLoss(mode=mode, anchor_weight=weight)
            model.fit(train, train[POLICY_TARGET].to_numpy(), epochs=60)

            chosen = pd.Series(GRID[model.select_action(evaluation)])
            rows.append({
                "loss": mode,
                "anchor_weight": weight,
                "mean_predicted_demand": float(model.predict_demand(evaluation).mean()),
                "mean_price": float(chosen.mean()),
                "entropy": action_entropy(chosen),
                "final_loss": model.history[-1],
            })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "anchor_sweep.csv", index=False)
    logger.info("Anchor sweep written")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    ensure_dirs()
    logger.info("Run '%s': city %d, simulator family '%s'",
                RUN_TAG, FILTER_CITY_ID, SIMULATOR_FAMILY)

    describe_dataset()

    matrix = build_feature_matrix(build_recovered_dataset())
    features = get_feature_names(matrix)
    train, valid, evaluation = temporal_split(matrix)

    logger.info("Fitting the evaluation simulator on %s", EVAL_TARGET)
    simulator = fit_simulator(train, valid, features, EVAL_TARGET)
    simulator.save(MODELS_DIR / f"simulator_{EVAL_TARGET}.pkl")

    logger.info("Fitting the policy demand model on %s", POLICY_TARGET)
    policy_model = fit_simulator(train, valid, features, POLICY_TARGET)
    policy_model.save(MODELS_DIR / f"demand_{POLICY_TARGET}.pkl")

    diagnostics = pd.DataFrame([
        diagnose_simulator(simulator, evaluation, features, EVAL_TARGET),
        diagnose_simulator(policy_model, evaluation, features, POLICY_TARGET),
    ])
    diagnostics.to_csv(RESULTS_DIR / "simulator_diagnostics.csv", index=False)
    logger.info("Simulator diagnostics:\n%s", diagnostics.round(4).to_string(index=False))

    mediator_ablation(train, valid, evaluation, features)
    record_demand_curve(simulator, evaluation, features)
    reward_sensitivity(simulator, evaluation, features)

    if SIMULATOR_FAMILY == "lgbm":
        rank_features(simulator, valid, features)

    policies = {
        "clairvoyant": Clairvoyant(simulator, features),
        "predict_then_optimise": PredictThenOptimise(policy_model, features),
        "fixed_full_price": FixedPrice(1.0),
        "fixed_discount": FixedPrice(0.85),
        "historical": HistoricalPolicy(),
        "rule_based": RuleBased(),
    }

    logger.info("Value iteration")
    value_iteration = TabularValueIteration().fit(
        train, policies["predict_then_optimise"].expected_reward(train)
    )
    policies["value_iteration"] = value_iteration
    sweeps = np.arange(1, len(value_iteration.residuals) + 1)
    pd.DataFrame({
        "sweep": sweeps,
        "residual": value_iteration.residuals,
        "gamma_power": value_iteration.gamma ** sweeps,
    }).to_csv(RESULTS_DIR / "vi_residuals.csv", index=False)

    logger.info("PPO for %d timesteps", PPO_TIMESTEPS)
    agent = PPOAgent(PricingEnv(policy_model, train, features)).learn(PPO_TIMESTEPS)
    agent.save(MODELS_DIR / "ppo_agent")
    policies["ppo"] = agent

    anchor_sweep(train, evaluation, features)

    for loss in ("mse", "spo", "perturbed"):
        logger.info("Decision-focused learning: %s", loss)
        model = DecisionFocused(features, loss=loss).fit(
            train, train[POLICY_TARGET].to_numpy(), epochs=DFL_EPOCHS,
        )
        model.save(MODELS_DIR / f"dfl_{loss}.pt")
        policies[f"dfl_{loss}"] = model
        pd.DataFrame({
            "epoch": np.arange(1, len(model.history) + 1), "loss": model.history,
        }).to_csv(RESULTS_DIR / f"dfl_{loss}_history.csv", index=False)

    logger.info("Evaluating on the held-out horizon")
    table = compare_policies(policies, evaluation, simulator, features)
    table.to_csv(RESULTS_DIR / "policy_comparison.csv")

    detail = {
        name: score_policy(policy, evaluation, simulator, features)
        for name, policy in policies.items()
    }

    reference = detail["historical"]
    differences = []
    for name, scored in detail.items():
        point, low, high = paired_bootstrap(scored, reference)
        differences.append({
            "policy": name, "difference": point, "ci_low": low, "ci_high": high,
            "significant": (low > 0) or (high < 0),
        })
    difference_table = pd.DataFrame(differences).set_index("policy")
    difference_table.to_csv(RESULTS_DIR / "paired_differences.csv")

    distribution = pd.DataFrame({
        name: pd.Series(GRID[scored["action"]]).value_counts().sort_index()
        for name, scored in detail.items()
    }).fillna(0).astype(int)
    distribution.to_csv(RESULTS_DIR / "action_distribution.csv")

    for name in ("clairvoyant", "predict_then_optimise", "ppo", "historical"):
        detail[name].to_csv(RESULTS_DIR / f"per_state_{name}.csv", index=False)

    print("\n" + table.round(4).to_string())
    print("\nPaired differences against the operator's own pricing:")
    print(difference_table.round(4).to_string())
    print(f"\nAll artefacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()