import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

from data_loader import merge_datasets, get_time_features
from config import *
from train_simple import make_gaussian_net, make_normalized_net, calculate_metrics
from stoch_optimizer import StochasticOptimizer
from plotting import plot_predictions, plot_optimization


def run_week_visualization(week_num=48):
    print("=" * 70)
    print(f"Evaluation and Visualization - Week {week_num}")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    merged = merge_datasets(FUNDIUM_DATA_PATH, PRICE_DATA_PATH)

    # Pre-compute all lag features on the full dataset (long lags like 672-step need full series)
    for lag in ABWAERME_LAG_STEPS:
        merged[f"abwaerme_lag_{lag}"] = (
            merged["ofen_abwaerme_nestle_5893_mw"].shift(lag).ffill().bfill()
        )
    for lag in DLA_LAG_STEPS:
        merged[f"dla_lag_{lag}"] = (
            merged["dla_stromverbrauch_kwh"].shift(lag).ffill().bfill()
        )
    for lag in PRICE_LAG_STEPS:
        merged[f"price_lag_{lag}"] = (
            merged["price_eur_mwh"].shift(lag).ffill().bfill()
        )

    train_size = int(len(merged) * TRAIN_SPLIT)

    start = pd.Timestamp("2023-01-01")
    week_start = start + pd.Timedelta(days=(week_num - 1) * 7)
    week_end = week_start + pd.Timedelta(days=6, hours=23, minutes=45)

    mask = (merged["new_time"] >= week_start) & (merged["new_time"] <= week_end)
    week_data = merged[mask].reset_index(drop=True)

    is_test = week_data.index[0] >= train_size
    print(f"  Week {week_num}: {week_start.date()} to {week_end.date()}")
    print(f"  Samples: {len(week_data)} ({'test' if is_test else 'train'} period)")

    print("\n[2/7] Loading models...")
    dla_input_dim = 6 + len(DLA_LAG_STEPS)
    price_input_dim = 8 + len(PRICE_LAG_STEPS)
    abwaerme_input_dim = 6 + len(ABWAERME_LAG_STEPS)

    dla_bnn = make_gaussian_net(dla_input_dim, HIDDEN_DIMS, init_noise=0.5)
    price_bnn = make_normalized_net(price_input_dim, HIDDEN_DIMS, init_noise=0.1)
    abwaerme_bnn = make_gaussian_net(abwaerme_input_dim, HIDDEN_DIMS, init_noise=0.3)

    dla_bnn.load_state_dict(torch.load("dla_bnn.pt", weights_only=True))
    price_bnn.load_state_dict(torch.load("price_bnn.pt", weights_only=True))
    abwaerme_bnn.load_state_dict(torch.load("abwaerme_bnn.pt", weights_only=True))
    dla_bnn.eval()
    price_bnn.eval()
    abwaerme_bnn.eval()
    print("  Models loaded!")

    norm_data = np.load("dla_norm.npz")
    dla_mean = norm_data["mean"]
    dla_std = norm_data["std"]

    price_norm = np.load("price_norm.npz")
    price_min = price_norm["price_min"]
    price_max = price_norm["price_max"]

    abwaerme_norm = np.load("abwaerme_norm.npz")
    abwaerme_mean = float(abwaerme_norm["mean"])
    abwaerme_std = float(abwaerme_norm["std"])

    print("\n[3/7] Making predictions...")
    time_feats = get_time_features(week_data["new_time"]).values

    renewable = week_data["renewable_gen_mw"].values
    load = week_data["load_mw"].values
    price_cols = np.column_stack([renewable, load])
    price_cols_mean = price_cols[:1000].mean(axis=0)
    price_cols_std = price_cols[:1000].std(axis=0)
    price_cols_norm = (price_cols - price_cols_mean) / (price_cols_std + 1e-8)

    # DLA input: time features + normalized DLA lags
    dla_lag_cols = [f"dla_lag_{lag}" for lag in DLA_LAG_STEPS]
    dla_lag_feats_norm = (week_data[dla_lag_cols].values - dla_mean) / (dla_std + 1e-8)
    batch_dla = torch.tensor(np.hstack([time_feats, dla_lag_feats_norm]), dtype=torch.float32)

    # Price input: time features + market features + normalized price lags
    price_lag_cols = [f"price_lag_{lag}" for lag in PRICE_LAG_STEPS]
    price_lag_feats_norm = (week_data[price_lag_cols].values - price_min) / (price_max - price_min + 1e-8)
    price_features = np.hstack([time_feats, price_cols_norm, price_lag_feats_norm])
    batch_price = torch.tensor(price_features, dtype=torch.float32)

    dla_mu_norm, dla_sigma_norm, dla_mu_std, _ = dla_bnn.predict(
        batch_dla, n_samples=NUM_MC_SAMPLES
    )
    dla_mu = dla_mu_norm * dla_std + dla_mean
    dla_sigma = dla_sigma_norm * dla_std
    dla_mu_std = dla_mu_std * dla_std

    price_mu, price_sigma, price_mu_std, _ = price_bnn.predict_denormalized(
        batch_price, n_samples=NUM_MC_SAMPLES, price_min=price_min, price_max=price_max
    )

    # Abwaerme input: time features + normalized lag features
    abwaerme_lag_cols = [f"abwaerme_lag_{lag}" for lag in ABWAERME_LAG_STEPS]
    abwaerme_lag_feats_norm = (week_data[abwaerme_lag_cols].values - abwaerme_mean) / (abwaerme_std + 1e-8)
    batch_abwaerme = torch.tensor(
        np.hstack([time_feats, abwaerme_lag_feats_norm]), dtype=torch.float32
    )

    abwaerme_mu_norm, abwaerme_sigma_norm, abwaerme_mu_std, _ = abwaerme_bnn.predict(
        batch_abwaerme, n_samples=NUM_MC_SAMPLES
    )
    abwaerme_mu = abwaerme_mu_norm * abwaerme_std + abwaerme_mean
    abwaerme_sigma = abwaerme_sigma_norm * abwaerme_std
    abwaerme_mu_std = abwaerme_mu_std * abwaerme_std

    # Combined predictive std: sqrt(epistemic^2 + aleatoric^2)
    dla_total_std = np.sqrt(dla_mu_std**2 + dla_sigma**2)
    price_total_std = np.sqrt(price_mu_std**2 + price_sigma**2)
    abwaerme_total_std = np.sqrt(abwaerme_mu_std**2 + abwaerme_sigma**2)

    dla_actual = week_data["dla_stromverbrauch_kwh"].values
    price_actual = week_data["price_eur_mwh"].values
    abwaerme_actual = week_data["ofen_abwaerme_nestle_5893_mw"].values

    print(f"  DLA predictions: {len(dla_mu)} samples")
    print(f"  Price predictions: {len(price_mu)} samples")
    print(f"  Abwaerme predictions: {len(abwaerme_mu)} samples")

    print("\n[4/7] Calculating metrics...")

    dla_metrics = calculate_metrics(dla_actual, dla_mu, dla_mu_std, dla_sigma)
    price_metrics = calculate_metrics(price_actual, price_mu, price_mu_std, price_sigma)
    abwaerme_metrics = calculate_metrics(abwaerme_actual, abwaerme_mu, abwaerme_mu_std, abwaerme_sigma)

    print("\n  DLA Consumption Metrics:")
    print(f"    RMSE:        {dla_metrics['rmse']:.2f} kWh")
    print(f"    MAE:        {dla_metrics['mae']:.2f} kWh")
    print(f"    MAPE:       {dla_metrics['mape']:.1f}%")
    print(f"    Coverage:   {dla_metrics['coverage']:.1f}% (95% CI)")
    print(f"    Scale:      {dla_metrics['scale_factor']:.2f}x")

    print("\n  Price Metrics:")
    print(f"    RMSE:        {price_metrics['rmse']:.2f} EUR/MWh")
    print(f"    MAE:        {price_metrics['mae']:.2f} EUR/MWh")
    print(f"    Coverage:   {price_metrics['coverage']:.1f}% (95% CI)")
    print(f"    Scale:      {price_metrics['scale_factor']:.2f}x")

    print("\n  Abwaerme Nestle Metrics:")
    print(f"    RMSE:        {abwaerme_metrics['rmse']:.4f} MW")
    print(f"    MAE:        {abwaerme_metrics['mae']:.4f} MW")
    print(f"    MAPE:       {abwaerme_metrics['mape']:.1f}%")
    print(f"    Coverage:   {abwaerme_metrics['coverage']:.1f}% (95% CI)")
    print(f"    Scale:      {abwaerme_metrics['scale_factor']:.2f}x")

    print("\n[5/7] Running stoch. optimization (discharge allowed)...")

    # Load Cholesky factors for correlated scenario sampling
    dla_chol = np.load("dla_chol.npz")["L"] if os.path.exists("dla_chol.npz") else None
    price_chol = np.load("price_chol.npz")["L"] if os.path.exists("price_chol.npz") else None
    if dla_chol is not None:
        print("  Using Cholesky correlated sampling for DLA and price scenarios")
    else:
        print("  Using independent sampling (run train_simple.py to enable Cholesky)")

    optimizer = StochasticOptimizer(
        battery_capacity=BATTERY_CAPACITY_KWH,
        max_power=BATTERY_MAX_POWER_KW,
        efficiency=BATTERY_ROUND_TRIP_EFFICIENCY,
        horizon_hours=HORIZON_HOURS,
        n_scenarios=N_SCENARIOS,
        verbose=False,
        dla_chol=dla_chol,
        price_chol=price_chol,
    )

    current_soc = BATTERY_CAPACITY_KWH * 0.5
    results = []
    result_times = []

    step_interval = OPT_FREQUENCY 
    n_steps = len(week_data) - optimizer.horizon_steps
    step_range = range(0, n_steps, step_interval)

    for i in tqdm(step_range, desc="Optimizing"):
        price_mean = price_mu[i : i + optimizer.horizon_steps]
        price_std = price_total_std[i : i + optimizer.horizon_steps]
        dla_mean_batch = dla_mu[i : i + optimizer.horizon_steps]
        dla_std_batch = dla_total_std[i : i + optimizer.horizon_steps]

        opt_result = optimizer.stochastic_optimize(
            price_mean,
            price_std,
            dla_mean_batch,
            dla_std_batch,
            current_soc,
            discharge_allowed=True,
        )

        if opt_result:
            results.append(opt_result)
            result_times.append(i)
            # MPC: commit only the first step_interval steps, carry SOC from there
            current_soc = np.clip(opt_result["avg_soc"][step_interval], 0, BATTERY_CAPACITY_KWH)

    print(f"  Optimized {len(results)} time steps")

    all_charge = []
    all_discharge = []
    for r in results:
        if r:
            # Only count the committed steps (first step_interval), not the full horizon
            all_charge.extend(r["avg_charge"][:step_interval])
            all_discharge.extend(r["avg_discharge"][:step_interval])

    if all_charge:
        print(f"  Avg charge:    {np.mean(all_charge):.2f} kWh/interval")
        print(f"  Max charge:   {np.max(all_charge):.2f} kWh/interval")
        print(f"  Total charged: {np.sum(all_charge):.0f} kWh")

    print("\n[6/7] Computing fair cost comparison...")
    print("  (Schedule from predictions -> Apply to actual data)")

    baseline_cost = np.sum(dla_actual / 1000 * price_actual)
    total_energy_kwh = np.sum(dla_actual)
    avg_price = np.mean(price_actual)

    sqrt_eff = np.sqrt(BATTERY_ROUND_TRIP_EFFICIENCY)
    inv_sqrt_eff = 1.0 / sqrt_eff

    actual_optimized_cost = 0
    start_idx = 0
    for r in results:
        if r:
            # MPC: only apply the committed steps (first step_interval) to actual data
            charge_schedule = r["avg_charge"][:step_interval]
            discharge_schedule = r["avg_discharge"][:step_interval]
            max_t = min(len(charge_schedule), len(dla_actual) - start_idx)
            for t in range(max_t):
                actual_consumption = dla_actual[start_idx + t] / 1000
                actual_price = price_actual[start_idx + t]

                net_consumption = actual_consumption - (
                    (charge_schedule[t] - discharge_schedule[t]) / 1000 * inv_sqrt_eff
                )
                actual_optimized_cost += net_consumption * actual_price

            start_idx += step_interval

    print(f"  Energy consumed: {total_energy_kwh:.0f} kWh")
    print(f"  Baseline cost (no battery): {baseline_cost:.1f} EUR")
    print(f"  Optimized cost (predictions -> actual): {actual_optimized_cost:.1f} EUR")
    if actual_optimized_cost > 0:
        print(
            f"  Cost savings: {baseline_cost - actual_optimized_cost:.1f} EUR ({100 * (1 - actual_optimized_cost / baseline_cost):.1f}%)"
        )

    print("\n[7/7] Creating plots...")
    t_hours = np.arange(len(dla_actual)) * 0.25

    plot_predictions(
        week_num, t_hours,
        dla_actual, dla_mu, dla_sigma, dla_total_std, dla_metrics,
        price_actual, price_mu, price_sigma, price_total_std, price_metrics,
        abwaerme_actual, abwaerme_mu, abwaerme_sigma, abwaerme_total_std, abwaerme_metrics,
    )

    plot_optimization(
        week_num, t_hours, price_actual,
        results, result_times, step_interval,
        baseline_cost, actual_optimized_cost,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nDLA Consumption:")
    print(f"  RMSE: {dla_metrics['rmse']:.2f} kWh | MAE: {dla_metrics['mae']:.2f} kWh")
    print(
        f"  Coverage: {dla_metrics['coverage']:.1f}% | Scale: {dla_metrics['scale_factor']:.2f}x"
    )
    print(f"\nPrice:")
    print(
        f"  RMSE: {price_metrics['rmse']:.2f} EUR/MWh | MAE: {price_metrics['mae']:.2f} EUR/MWh"
    )
    print(
        f"  Coverage: {price_metrics['coverage']:.1f}% | Scale: {price_metrics['scale_factor']:.2f}x"
    )
    print(f"\nCosts:")
    print(f"  Baseline (no battery): {baseline_cost:.1f} EUR")
    print(f"  Optimized (with battery): {actual_optimized_cost:.1f} EUR")
    print(
        f"  Savings: {baseline_cost - actual_optimized_cost:.1f} EUR ({100 * (1 - actual_optimized_cost / baseline_cost):.1f}%)"
    )
    print(f"  Total energy: {total_energy_kwh:.0f} kWh")
    print("=" * 70)


if __name__ == "__main__":
    run_week_visualization(week_num=48)
