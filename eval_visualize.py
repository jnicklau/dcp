import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import merge_datasets, get_time_features
from config import *
from train_simple import make_gaussian_net, calculate_metrics, denormalize
from stoch_optimizer import StochasticOptimizer
from plots import plot_predictions, plot_optimization


# ── Data loading ──────────────────────────────────────────────────────────────

def load_week_data(week_num):
    """Load and slice one calendar week from the merged dataset, with lag features."""
    merged = merge_datasets(FUNDIUM_DATA_PATH, PRICE_DATA_PATH)

    # Pre-compute lag features on the full series before slicing.
    # Long lags (e.g. 672 steps = 1 week) need the complete history to be valid.
    for lag in ABWAERME_LAG_STEPS:
        merged[f"abwaerme_lag_{lag}"] = merged["ofen_abwaerme_nestle_5893_mw"].shift(lag).ffill().bfill()
    for lag in DLA_LAG_STEPS:
        merged[f"dla_lag_{lag}"] = merged["dla_stromverbrauch_kwh"].shift(lag).ffill().bfill()
    for lag in PRICE_LAG_STEPS:
        merged[f"price_lag_{lag}"] = merged["price_eur_mwh"].shift(lag).ffill().bfill()

    train_size = int(len(merged) * TRAIN_SPLIT)

    start = pd.Timestamp("2023-01-01")
    week_start = start + pd.Timedelta(days=(week_num - 1) * 7)
    week_end   = week_start + pd.Timedelta(days=6, hours=23, minutes=45)
    mask = (merged["new_time"] >= week_start) & (merged["new_time"] <= week_end)
    week_data = merged[mask].reset_index(drop=True)

    is_test = week_data.index[0] >= train_size
    print(f"  Week {week_num}: {week_start.date()} to {week_end.date()}")
    print(f"  Samples: {len(week_data)} ({'test' if is_test else 'train'} period)")
    return week_data


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
    """Instantiate BNN architectures and load saved weights + normalisation stats."""
    dla_input_dim      = 6 + len(DLA_LAG_STEPS)
    price_input_dim    = 8 + len(PRICE_LAG_STEPS)
    abwaerme_input_dim = 6 + len(ABWAERME_LAG_STEPS)

    dla_bnn      = make_gaussian_net(dla_input_dim,      HIDDEN_DIMS, init_noise=0.5)
    price_bnn    = make_gaussian_net(price_input_dim,    HIDDEN_DIMS, init_noise=0.1)
    abwaerme_bnn = make_gaussian_net(abwaerme_input_dim, HIDDEN_DIMS, init_noise=0.3)

    dla_bnn.load_state_dict(torch.load(f"{MODELS_DIR}/dla_bnn.pt",      weights_only=True))
    price_bnn.load_state_dict(torch.load(f"{MODELS_DIR}/price_bnn.pt",  weights_only=True))
    abwaerme_bnn.load_state_dict(torch.load(f"{MODELS_DIR}/abwaerme_bnn.pt", weights_only=True))
    dla_bnn.eval(); price_bnn.eval(); abwaerme_bnn.eval()
    print("  BNN models loaded.")

    # Normalisation stats saved during training
    dla_norm      = np.load(f"{NORMS_DIR}/dla_norm.npz")
    price_norm    = np.load(f"{NORMS_DIR}/price_norm.npz")
    abwaerme_norm = np.load(f"{NORMS_DIR}/abwaerme_norm.npz")

    norms = {
        "dla_mean":          float(dla_norm["mean"]),
        "dla_std":           float(dla_norm["std"]),
        "price_mean":        float(price_norm["mean"]),
        "price_std":         float(price_norm["std"]),
        "market_feats_mean": price_norm["market_feats_mean"],
        "market_feats_std":  price_norm["market_feats_std"],
        "abwaerme_mean":     float(abwaerme_norm["mean"]),
        "abwaerme_std":      float(abwaerme_norm["std"]),
    }
    return dla_bnn, price_bnn, abwaerme_bnn, norms


# ── Prediction ────────────────────────────────────────────────────────────────

def make_predictions(week_data, dla_bnn, price_bnn, abwaerme_bnn, norms):
    """
    Build feature tensors for each signal and run MC forward passes.

    Returns per-signal dicts with keys: mu, sigma (aleatoric), mu_std (epistemic),
    total_std, and actual (ground-truth values for the week).
    """
    dla_mean  = norms["dla_mean"];   dla_std   = norms["dla_std"]
    price_mean = norms["price_mean"]; price_std = norms["price_std"]
    abwaerme_mean = norms["abwaerme_mean"]; abwaerme_std = norms["abwaerme_std"]

    time_feats = get_time_features(week_data["new_time"]).values

    # Market features: renewable generation + grid load, normalised with training stats
    mf_raw = np.column_stack([week_data["renewable_gen_mw"].values, week_data["load_mw"].values])
    market_feats_norm = (mf_raw - norms["market_feats_mean"]) / (norms["market_feats_std"] + 1e-8)

    # ── DLA ──────────────────────────────────────────────────────────────────
    dla_lag_cols = [f"dla_lag_{lag}" for lag in DLA_LAG_STEPS]
    dla_lags_norm = (week_data[dla_lag_cols].values - dla_mean) / (dla_std + 1e-8)
    batch_dla = torch.tensor(np.hstack([time_feats, dla_lags_norm]), dtype=torch.float32)

    dla_mu_norm, dla_sigma_norm, dla_mu_std_norm, _ = dla_bnn.predict(batch_dla, n_samples=NUM_MC_SAMPLES)
    dla_mu    = denormalize(dla_mu_norm, dla_mean, dla_std)
    dla_sigma = dla_sigma_norm * dla_std
    dla_mu_std = dla_mu_std_norm * dla_std

    # ── Price ─────────────────────────────────────────────────────────────────
    price_lag_cols = [f"price_lag_{lag}" for lag in PRICE_LAG_STEPS]
    price_lags_norm = (week_data[price_lag_cols].values - price_mean) / (price_std + 1e-8)
    batch_price = torch.tensor(np.hstack([time_feats, market_feats_norm, price_lags_norm]), dtype=torch.float32)

    price_mu_norm, price_sigma_norm, price_mu_std_norm, _ = price_bnn.predict(
        batch_price, n_samples=NUM_MC_SAMPLES
    )
    price_mu     = denormalize(price_mu_norm,     price_mean, price_std)
    price_sigma  = price_sigma_norm  * price_std
    price_mu_std = price_mu_std_norm * price_std

    # ── Abwärme ───────────────────────────────────────────────────────────────
    abwaerme_lag_cols = [f"abwaerme_lag_{lag}" for lag in ABWAERME_LAG_STEPS]
    abwaerme_lags_norm = (week_data[abwaerme_lag_cols].values - abwaerme_mean) / (abwaerme_std + 1e-8)
    batch_abwaerme = torch.tensor(np.hstack([time_feats, abwaerme_lags_norm]), dtype=torch.float32)

    abwaerme_mu_norm, abwaerme_sigma_norm, abwaerme_mu_std_norm, _ = abwaerme_bnn.predict(
        batch_abwaerme, n_samples=NUM_MC_SAMPLES
    )
    abwaerme_mu    = denormalize(abwaerme_mu_norm, abwaerme_mean, abwaerme_std)
    abwaerme_sigma = abwaerme_sigma_norm * abwaerme_std
    abwaerme_mu_std = abwaerme_mu_std_norm * abwaerme_std

    # Combined predictive std:  sqrt( epistemic^2 + aleatoric^2 )
    dla_total_std      = np.sqrt(dla_mu_std**2      + dla_sigma**2)
    price_total_std    = np.sqrt(price_mu_std**2    + price_sigma**2)
    abwaerme_total_std = np.sqrt(abwaerme_mu_std**2 + abwaerme_sigma**2)

    preds = {
        "dla":      {"mu": dla_mu,      "sigma": dla_sigma,      "mu_std": dla_mu_std,      "total_std": dla_total_std,      "actual": week_data["dla_stromverbrauch_kwh"].values},
        "price":    {"mu": price_mu,    "sigma": price_sigma,    "mu_std": price_mu_std,    "total_std": price_total_std,    "actual": week_data["price_eur_mwh"].values},
        "abwaerme": {"mu": abwaerme_mu, "sigma": abwaerme_sigma, "mu_std": abwaerme_mu_std, "total_std": abwaerme_total_std, "actual": week_data["ofen_abwaerme_nestle_5893_mw"].values},
    }
    return preds


# ── Optimization ──────────────────────────────────────────────────────────────

def run_optimization(preds):
    """
    MPC receding-horizon stochastic battery optimization.

    At each re-optimization step we solve N_SCENARIOS LP instances over the
    HORIZON_HOURS look-ahead, then commit only the first OPT_FREQUENCY steps
    (3 hours) before re-solving with updated predictions.
    """
    # Load Cholesky factors for temporally-correlated scenario sampling
    dla_chol_path   = f"{NORMS_DIR}/dla_chol.npz"
    price_chol_path = f"{NORMS_DIR}/price_chol.npz"
    dla_chol   = np.load(dla_chol_path)["L"]   if os.path.exists(dla_chol_path)   else None
    price_chol = np.load(price_chol_path)["L"] if os.path.exists(price_chol_path) else None
    if dla_chol is not None:
        print("  Cholesky correlated sampling enabled for DLA and price scenarios.")
    else:
        print("  Independent Gaussian sampling (run train_simple.py to enable Cholesky).")

    optimizer = StochasticOptimizer(
        battery_capacity=BATTERY_CAPACITY_KWH,
        max_power=BATTERY_MAX_POWER_KW,
        efficiency=BATTERY_ROUND_TRIP_EFFICIENCY,
        horizon_hours=HORIZON_HOURS,
        n_scenarios=N_SCENARIOS,
        dla_chol=dla_chol,
        price_chol=price_chol,
    )

    price_mu    = preds["price"]["mu"]
    price_std   = preds["price"]["total_std"]
    dla_mu      = preds["dla"]["mu"]
    dla_std     = preds["dla"]["total_std"]
    price_actual = preds["price"]["actual"]
    dla_actual   = preds["dla"]["actual"]

    current_soc = BATTERY_CAPACITY_KWH * 0.5
    results, result_times = [], []
    n_steps = len(price_mu) - optimizer.horizon_steps

    for i in tqdm(range(0, n_steps, OPT_FREQUENCY), desc="Optimizing"):
        opt_result = optimizer.stochastic_optimize(
            price_mu[i : i + optimizer.horizon_steps],
            price_std[i : i + optimizer.horizon_steps],
            dla_mu[i : i + optimizer.horizon_steps],
            dla_std[i : i + optimizer.horizon_steps],
            current_soc,
            discharge_allowed=True,
            mode=SO_MODE,  # "extensive" or "scenario_avg" for better uncertainty handling (but slower)
        )
        if opt_result:
            results.append(opt_result)
            result_times.append(i)
            # Commit the first OPT_FREQUENCY steps; carry the resulting SOC forward
            current_soc = np.clip(opt_result["avg_soc"][OPT_FREQUENCY], 0, BATTERY_CAPACITY_KWH)

    print(f"  Optimized {len(results)} MPC steps")

    # ── Cost comparison: schedule from predictions applied to actual data ─────
    baseline_cost      = np.sum(dla_actual / 1000 * price_actual)
    actual_opt_cost    = 0.0
    inv_sqrt_eff       = 1.0 / np.sqrt(BATTERY_ROUND_TRIP_EFFICIENCY)
    start_idx = 0
    for r in results:
        charge_sched    = r["avg_charge"][:OPT_FREQUENCY]
        discharge_sched = r["avg_discharge"][:OPT_FREQUENCY]
        max_t = min(len(charge_sched), len(dla_actual) - start_idx)
        for t in range(max_t):
            net = (dla_actual[start_idx + t] / 1000
                   - (charge_sched[t] - discharge_sched[t]) / 1000 * inv_sqrt_eff)
            actual_opt_cost += net * price_actual[start_idx + t]
        start_idx += OPT_FREQUENCY

    savings = baseline_cost - actual_opt_cost
    print(f"  Baseline cost:  {baseline_cost:.1f} EUR")
    print(f"  Optimized cost: {actual_opt_cost:.1f} EUR")
    print(f"  Savings:        {savings:.1f} EUR ({100 * savings / baseline_cost:.1f}%)")

    return results, result_times, baseline_cost, actual_opt_cost


# ── Main entry point ──────────────────────────────────────────────────────────

def run_week_visualization(week_num=48):
    print("=" * 70)
    print(f"Evaluation and Visualization — Week {week_num}")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    week_data = load_week_data(week_num)

    print("\n[2/5] Loading models and normalisation stats...")
    dla_bnn, price_bnn, abwaerme_bnn, norms = load_models()

    print("\n[3/5] Making predictions...")
    preds = make_predictions(week_data, dla_bnn, price_bnn, abwaerme_bnn, norms)

    dla_metrics      = calculate_metrics(preds["dla"]["actual"],      preds["dla"]["mu"],      preds["dla"]["mu_std"],      preds["dla"]["sigma"])
    price_metrics    = calculate_metrics(preds["price"]["actual"],    preds["price"]["mu"],    preds["price"]["mu_std"],    preds["price"]["sigma"])
    abwaerme_metrics = calculate_metrics(preds["abwaerme"]["actual"], preds["abwaerme"]["mu"], preds["abwaerme"]["mu_std"], preds["abwaerme"]["sigma"])

    print("\n  DLA Consumption:")
    print(f"    RMSE: {dla_metrics['rmse']:.2f} kWh | MAE: {dla_metrics['mae']:.2f} kWh | Coverage: {dla_metrics['coverage']:.1f}%")
    print("\n  Price:")
    print(f"    RMSE: {price_metrics['rmse']:.2f} EUR/MWh | MAE: {price_metrics['mae']:.2f} EUR/MWh | Coverage: {price_metrics['coverage']:.1f}%")
    print("\n  Abwärme Nestle:")
    print(f"    RMSE: {abwaerme_metrics['rmse']:.4f} MW | MAE: {abwaerme_metrics['mae']:.4f} MW | Coverage: {abwaerme_metrics['coverage']:.1f}%")

    print("\n[4/5] Running stochastic MPC optimization...")
    results, result_times, baseline_cost, actual_opt_cost = run_optimization(preds)

    print("\n[5/5] Creating plots...")
    t_hours = np.arange(len(preds["dla"]["actual"])) * 0.25

    plot_predictions(
        week_num, t_hours,
        preds["dla"]["actual"],      preds["dla"]["mu"],      preds["dla"]["sigma"],      preds["dla"]["total_std"],      dla_metrics,
        preds["price"]["actual"],    preds["price"]["mu"],    preds["price"]["sigma"],    preds["price"]["total_std"],    price_metrics,
        preds["abwaerme"]["actual"], preds["abwaerme"]["mu"], preds["abwaerme"]["sigma"], preds["abwaerme"]["total_std"], abwaerme_metrics,
    )
    plot_optimization(
        week_num, t_hours, preds["price"]["actual"],
        results, result_times, OPT_FREQUENCY,
        baseline_cost, actual_opt_cost,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  DLA   — RMSE: {dla_metrics['rmse']:.2f} kWh | Coverage: {dla_metrics['coverage']:.1f}%")
    print(f"  Price — RMSE: {price_metrics['rmse']:.2f} EUR/MWh | Coverage: {price_metrics['coverage']:.1f}%")
    savings = baseline_cost - actual_opt_cost
    print(f"  Costs — Baseline: {baseline_cost:.1f} EUR | Optimized: {actual_opt_cost:.1f} EUR | Savings: {savings:.1f} EUR ({100 * savings / baseline_cost:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_week_visualization(week_num=48)
