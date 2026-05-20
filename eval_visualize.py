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

def load_eval_data():
    """Load and slice the evaluation window defined by EVAL_START_DATE / EVAL_N_DAYS."""
    merged = merge_datasets(FUNDIUM_DATA_PATH, PRICE_DATA_PATH)

    # Pre-compute lag features on the full series before slicing.
    # Long lags (e.g. 672 steps = 1 week) need the complete history to be valid.
    for lag in ABWAERME_LAG_STEPS:
        merged[f"abwaerme_lag_{lag}"] = merged["ofen_abwaerme_nestle_5893_mw"].shift(lag).ffill().bfill()
    for lag in DLA_LAG_STEPS:
        merged[f"dla_lag_{lag}"] = merged["dla_stromverbrauch_kwh"].shift(lag).ffill().bfill()
    for lag in PL2_LAG_STEPS:
        merged[f"pl2_lag_{lag}"] = merged["pl2_stromverbrauch_kwh"].shift(lag).ffill().bfill()
    for lag in PRICE_LAG_STEPS:
        merged[f"price_lag_{lag}"] = merged["price_eur_mwh"].shift(lag).ffill().bfill()

    train_size  = int(len(merged) * TRAIN_SPLIT)
    eval_start  = pd.Timestamp(EVAL_START_DATE)
    # Load EVAL_N_DAYS of eval data plus HORIZON_HOURS of look-ahead tail so the
    # MPC loop can always fill the full horizon window at every step of the eval period.
    eval_end    = eval_start + pd.Timedelta(days=EVAL_N_DAYS) - pd.Timedelta(minutes=15)
    load_end    = eval_end   + pd.Timedelta(hours=HORIZON_HOURS)
    mask        = (merged["new_time"] >= eval_start) & (merged["new_time"] <= load_end)
    eval_data   = merged[mask].reset_index(drop=True)

    if eval_data.empty:
        raise ValueError(f"No data found for {eval_start.date()} – {eval_end.date()}. "
                         "Check EVAL_START_DATE and that the date is within 2023.")
    is_test = int(eval_data.index[-1]) >= train_size
    print(f"  Period: {eval_start.date()} to {eval_end.date()} ({EVAL_N_DAYS} days)")
    print(f"  Samples: {len(eval_data)} ({'test' if is_test else 'train'} period)")
    return eval_data, eval_start, eval_end


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
    """Instantiate BNN architectures and load saved weights + normalisation stats."""
    dla_input_dim      = 6 + len(DLA_LAG_STEPS)
    price_input_dim    = 8 + len(PRICE_LAG_STEPS)
    abwaerme_input_dim = 6 + len(ABWAERME_LAG_STEPS)
    pl2_input_dim      = 6 + len(PL2_LAG_STEPS)

    dla_bnn      = make_gaussian_net(dla_input_dim,      HIDDEN_DIMS, init_noise=0.5)
    price_bnn    = make_gaussian_net(price_input_dim,    HIDDEN_DIMS, init_noise=0.1)
    abwaerme_bnn = make_gaussian_net(abwaerme_input_dim, HIDDEN_DIMS, init_noise=0.3)
    pl2_bnn      = make_gaussian_net(pl2_input_dim,      HIDDEN_DIMS, init_noise=0.5)

    dla_bnn.load_state_dict(torch.load(f"{MODELS_DIR}/dla_bnn.pt",      weights_only=True))
    price_bnn.load_state_dict(torch.load(f"{MODELS_DIR}/price_bnn.pt",  weights_only=True))
    abwaerme_bnn.load_state_dict(torch.load(f"{MODELS_DIR}/abwaerme_bnn.pt", weights_only=True))
    pl2_bnn.load_state_dict(torch.load(f"{MODELS_DIR}/pl2_bnn.pt",      weights_only=True))
    dla_bnn.eval(); price_bnn.eval(); abwaerme_bnn.eval(); pl2_bnn.eval()
    print("  BNN models loaded.")

    # Normalisation stats saved during training
    dla_norm      = np.load(f"{NORMS_DIR}/dla_norm.npz")
    price_norm    = np.load(f"{NORMS_DIR}/price_norm.npz")
    abwaerme_norm = np.load(f"{NORMS_DIR}/abwaerme_norm.npz")
    pl2_norm      = np.load(f"{NORMS_DIR}/pl2_norm.npz")

    norms = {
        "dla_mean":          float(dla_norm["mean"]),
        "dla_std":           float(dla_norm["std"]),
        "price_mean":        float(price_norm["mean"]),
        "price_std":         float(price_norm["std"]),
        "market_feats_mean": price_norm["market_feats_mean"],
        "market_feats_std":  price_norm["market_feats_std"],
        "abwaerme_mean":     float(abwaerme_norm["mean"]),
        "abwaerme_std":      float(abwaerme_norm["std"]),
        "pl2_mean":          float(pl2_norm["mean"]),
        "pl2_std":           float(pl2_norm["std"]),
    }
    return dla_bnn, price_bnn, abwaerme_bnn, pl2_bnn, norms


# ── Prediction ────────────────────────────────────────────────────────────────

def make_predictions(week_data, dla_bnn, price_bnn, abwaerme_bnn, pl2_bnn, norms):
    """
    Build feature tensors for each signal and run MC forward passes.

    Returns per-signal dicts with keys: mu, sigma (aleatoric), mu_std (epistemic),
    total_std, and actual (ground-truth values for the week).
    """
    dla_mean  = norms["dla_mean"];   dla_std   = norms["dla_std"]
    price_mean = norms["price_mean"]; price_std = norms["price_std"]
    abwaerme_mean = norms["abwaerme_mean"]; abwaerme_std = norms["abwaerme_std"]
    pl2_mean  = norms["pl2_mean"];   pl2_std   = norms["pl2_std"]

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

    # ── PL2 ────────────────────────────────────────────────────────────────────────
    pl2_lag_cols = [f"pl2_lag_{lag}" for lag in PL2_LAG_STEPS]
    pl2_lags_norm = (week_data[pl2_lag_cols].values - pl2_mean) / (pl2_std + 1e-8)
    batch_pl2 = torch.tensor(np.hstack([time_feats, pl2_lags_norm]), dtype=torch.float32)

    pl2_mu_norm, pl2_sigma_norm, pl2_mu_std_norm, _ = pl2_bnn.predict(batch_pl2, n_samples=NUM_MC_SAMPLES)
    pl2_mu     = denormalize(pl2_mu_norm,     pl2_mean, pl2_std)
    pl2_sigma  = pl2_sigma_norm  * pl2_std
    pl2_mu_std = pl2_mu_std_norm * pl2_std

    # Combined predictive std:  sqrt( epistemic^2 + aleatoric^2 )
    dla_total_std      = np.sqrt(dla_mu_std**2      + dla_sigma**2)
    price_total_std    = np.sqrt(price_mu_std**2    + price_sigma**2)
    abwaerme_total_std = np.sqrt(abwaerme_mu_std**2 + abwaerme_sigma**2)
    pl2_total_std      = np.sqrt(pl2_mu_std**2      + pl2_sigma**2)

    preds = {
        "dla":      {"mu": dla_mu,      "sigma": dla_sigma,      "mu_std": dla_mu_std,      "total_std": dla_total_std,      "actual": week_data["dla_stromverbrauch_kwh"].values},
        "price":    {"mu": price_mu,    "sigma": price_sigma,    "mu_std": price_mu_std,    "total_std": price_total_std,    "actual": week_data["price_eur_mwh"].values},
        "abwaerme": {"mu": abwaerme_mu, "sigma": abwaerme_sigma, "mu_std": abwaerme_mu_std, "total_std": abwaerme_total_std, "actual": week_data["ofen_abwaerme_nestle_5893_mw"].values},
        "pl2":      {"mu": pl2_mu,      "sigma": pl2_sigma,      "mu_std": pl2_mu_std,      "total_std": pl2_total_std,      "actual": week_data["pl2_stromverbrauch_kwh"].values},
    }
    return preds


# ── Optimization ──────────────────────────────────────────────────────────────

def run_optimization(preds, initial_soc=None):
    """
    MPC receding-horizon stochastic battery optimization.

    At each re-optimization step we solve N_SCENARIOS LP instances over the
    HORIZON_HOURS look-ahead, then commit only the first OPT_FREQUENCY steps
    (3 hours) before re-solving with updated predictions.

    Parameters
    ----------
    initial_soc : float or None
        Starting SOC in kWh. Defaults to INITIAL_SOC_KWH from config.
        Pass the ending SOC of a previous run to chain periods together.
    """
    # Load Cholesky factors for temporally-correlated scenario sampling
    dla_chol_path   = f"{NORMS_DIR}/dla_chol.npz"
    price_chol_path = f"{NORMS_DIR}/price_chol.npz"
    pl2_chol_path   = f"{NORMS_DIR}/pl2_chol.npz"
    dla_chol   = np.load(dla_chol_path)["L"]   if os.path.exists(dla_chol_path)   else None
    price_chol = np.load(price_chol_path)["L"] if os.path.exists(price_chol_path) else None
    pl2_chol   = np.load(pl2_chol_path)["L"]   if os.path.exists(pl2_chol_path)   else None
    if dla_chol is not None:
        print("  Cholesky correlated sampling enabled for DLA, PL2, and price scenarios.")
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
        pl2_chol=pl2_chol,
    )

    price_mu    = preds["price"]["mu"]
    price_std   = preds["price"]["total_std"]
    dla_mu      = preds["dla"]["mu"]
    dla_std     = preds["dla"]["total_std"]
    price_actual = preds["price"]["actual"]
    dla_actual   = preds["dla"]["actual"]
    # PL2 predictions — zeros until the PL2 BNN is trained and wired into make_predictions()
    pl2_mu     = preds["pl2"]["mu"]          if "pl2" in preds else np.zeros_like(dla_mu)
    pl2_std    = preds["pl2"]["total_std"]   if "pl2" in preds else np.zeros_like(dla_std)
    pl2_actual = preds["pl2"]["actual"]      if "pl2" in preds else np.zeros_like(dla_actual)

    current_soc = INITIAL_SOC_KWH if initial_soc is None else float(initial_soc)
    results, result_times = [], []
    n_steps = len(price_mu) - optimizer.horizon_steps

    for i in tqdm(range(0, n_steps, OPT_FREQUENCY), desc="Optimizing"):
        slice_ = slice(i, i + optimizer.horizon_steps)
        if SO_MODE == "scenario_approach":
            # Auto-computes the Campi-Garatti N and draws that many scenarios internally
            opt_result = optimizer.optimize_from_stats(
                price_mu[slice_], price_std[slice_],
                dla_mu[slice_],   dla_std[slice_],
                pl2_mu[slice_],   pl2_std[slice_],
                current_soc, discharge_allowed=True, mode=SO_MODE,
            )
        else:
            scenarios = optimizer.sample_scenarios(
                price_mu[slice_], price_std[slice_],
                dla_mu[slice_],   dla_std[slice_],
                pl2_mu[slice_],   pl2_std[slice_],
            )
            opt_result = optimizer.optimize(scenarios, current_soc, discharge_allowed=True, mode=SO_MODE)
        if opt_result:
            results.append(opt_result)
            result_times.append(i)
            # Commit the first OPT_FREQUENCY steps; carry the resulting SOC forward
            current_soc = np.clip(opt_result["avg_soc"][OPT_FREQUENCY], 0, BATTERY_CAPACITY_KWH)

    print(f"  Optimized {len(results)} MPC steps")

    # ── Deterministic MPC baseline (one LP per step on forecast means) ──────
    det_soc = INITIAL_SOC_KWH if initial_soc is None else float(initial_soc)
    det_results, det_result_times = [], []

    for i in tqdm(range(0, n_steps, OPT_FREQUENCY), desc="Det. baseline"):
        slice_ = slice(i, i + optimizer.horizon_steps)
        det_result = optimizer.optimize(
            [(price_mu[slice_], dla_mu[slice_], pl2_mu[slice_])],
            det_soc, discharge_allowed=True, mode="mean_scenario",
        )
        if det_result:
            det_results.append(det_result)
            det_result_times.append(i)
            det_soc = np.clip(det_result["avg_soc"][OPT_FREQUENCY], 0, BATTERY_CAPACITY_KWH)

    print(f"  Deterministic baseline: {len(det_results)} MPC steps")

    # ── Cost replay (shared efficiency factor) ────────────────────────────────
    inv_sqrt_eff = 1.0 / np.sqrt(BATTERY_ROUND_TRIP_EFFICIENCY)

    def _replay_cost(res_list, res_times):
        """Compute actual cost of a schedule applied to realised prices/consumption."""
        total = 0.0
        mask  = np.zeros(len(dla_actual), dtype=bool)
        for r, i in zip(res_list, res_times):
            chg  = r["avg_charge"][:OPT_FREQUENCY]
            dis  = r["avg_discharge"][:OPT_FREQUENCY]
            max_t = min(OPT_FREQUENCY, len(dla_actual) - i)
            for t in range(max_t):
                consumption = dla_actual[i + t] + pl2_actual[i + t]
                net = (consumption + (chg[t] - dis[t]) * inv_sqrt_eff) / 1000
                total += net * price_actual[i + t]
                mask[i + t] = True
        return total, mask

    actual_opt_cost, covered     = _replay_cost(results,     result_times)
    det_opt_cost,    det_covered = _replay_cost(det_results, det_result_times)

    # Restrict no-battery baseline to the same timesteps as the stochastic schedule
    total_actual  = dla_actual + pl2_actual
    baseline_cost = np.sum(total_actual[covered] / 1000 * price_actual[covered])

    for label, cost in [("Baseline (no battery)", baseline_cost),
                        ("Det. MPC (1 scenario)", det_opt_cost),
                        (f"Stoch. MPC ({N_SCENARIOS} scenarios)", actual_opt_cost)]:
        savings = baseline_cost - cost
        pct = 100 * savings / baseline_cost if baseline_cost else 0
        print(f"  {label:35s}  {cost:8.1f} EUR  ({pct:+.2f}%)")

    return results, result_times, baseline_cost, det_opt_cost, actual_opt_cost, optimizer


# ── Main entry point ──────────────────────────────────────────────────────────

def run_week_visualization():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    period_label = f"{EVAL_START_DATE}  +{EVAL_N_DAYS}d"
    print("=" * 70)
    print(f"Evaluation and Visualization — {period_label}")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    week_data, eval_start, eval_end = load_eval_data()

    print("\n[2/5] Loading models and normalisation stats...")
    dla_bnn, price_bnn, abwaerme_bnn, pl2_bnn, norms = load_models()

    print("\n[3/5] Making predictions...")
    preds = make_predictions(week_data, dla_bnn, price_bnn, abwaerme_bnn, pl2_bnn, norms)

    dla_metrics      = calculate_metrics(preds["dla"]["actual"],      preds["dla"]["mu"],      preds["dla"]["mu_std"],      preds["dla"]["sigma"])
    price_metrics    = calculate_metrics(preds["price"]["actual"],    preds["price"]["mu"],    preds["price"]["mu_std"],    preds["price"]["sigma"])
    abwaerme_metrics = calculate_metrics(preds["abwaerme"]["actual"], preds["abwaerme"]["mu"], preds["abwaerme"]["mu_std"], preds["abwaerme"]["sigma"])
    pl2_metrics      = calculate_metrics(preds["pl2"]["actual"],      preds["pl2"]["mu"],      preds["pl2"]["mu_std"],      preds["pl2"]["sigma"])

    print("\n  DLA Consumption:")
    print(f"    RMSE: {dla_metrics['rmse']:.2f} kWh | MAE: {dla_metrics['mae']:.2f} kWh | Coverage: {dla_metrics['coverage']:.1f}%")
    print("\n  Price:")
    print(f"    RMSE: {price_metrics['rmse']:.2f} EUR/MWh | MAE: {price_metrics['mae']:.2f} EUR/MWh | Coverage: {price_metrics['coverage']:.1f}%")
    print("\n  Abwärme Nestle:")
    print(f"    RMSE: {abwaerme_metrics['rmse']:.4f} MW | MAE: {abwaerme_metrics['mae']:.4f} MW | Coverage: {abwaerme_metrics['coverage']:.1f}%")
    print("\n  PL2 Consumption:")
    print(f"    RMSE: {pl2_metrics['rmse']:.2f} kWh | MAE: {pl2_metrics['mae']:.2f} kWh | Coverage: {pl2_metrics['coverage']:.1f}%")

    print(f"\n[4/5] Running {N_SCENARIOS} scenarios for {HORIZON_HOURS} hours as {SO_MODE} stochastic MPC optimization ...")
    if SO_MODE == "scenario_approach":
        print(f"  (Campi-Garatti bound with ε={RISK_EPSILON} and δ={RISK_DELTA} gives different N_scenarios = (1/{RISK_EPSILON}) (log(1/δ) + 2 + 1) per step)")
    if SO_MODE == "cvar":
        print(f"  (CVaR relaxation with risk level ε={RISK_EPSILON})")
    results, result_times, baseline_cost, det_opt_cost, actual_opt_cost, optimizer = run_optimization(preds)

    print("\n[5/5] Creating plots...")
    t_hours = np.arange(len(preds["dla"]["actual"])) * 0.25

    plot_predictions(
        period_label, t_hours,
        preds["dla"]["actual"],   preds["dla"]["mu"],   preds["dla"]["sigma"],   preds["dla"]["total_std"],   dla_metrics,
        preds["price"]["actual"], preds["price"]["mu"], preds["price"]["sigma"], preds["price"]["total_std"], price_metrics,
        preds["pl2"]["actual"],   preds["pl2"]["mu"],   preds["pl2"]["sigma"],   preds["pl2"]["total_std"],   pl2_metrics,
    )
    plot_optimization(
        period_label, t_hours, preds["price"]["actual"],
        results, result_times, OPT_FREQUENCY,
        baseline_cost, det_opt_cost, actual_opt_cost,
        optimizer_scenarios=optimizer.n_scenarios,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    # print(f"  PL2   — RMSE: {pl2_metrics['rmse']:.2f} kWh | Coverage: {pl2_metrics['coverage']:.1f}%")
    # print(f"  DLA   — RMSE: {dla_metrics['rmse']:.2f} kWh | Coverage: {dla_metrics['coverage']:.1f}%")
    # print(f"  Price — RMSE: {price_metrics['rmse']:.2f} EUR/MWh | Coverage: {price_metrics['coverage']:.1f}%")
    savings = baseline_cost - actual_opt_cost
    print(f"  Costs — Baseline: {baseline_cost:.1f} EUR | Optimized: {actual_opt_cost:.1f} EUR | Savings: {savings:.1f} EUR ({100 * savings / baseline_cost:.3f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_week_visualization()
