import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

from data_loader import merge_datasets, get_time_features
from config import *
from train_simple import GaussianBNN, NormalizedBNN, calculate_metrics


class StochasticOptimizer:
    def __init__(
        self,
        battery_capacity=BATTERY_CAPACITY_KWH,
        max_power=BATTERY_MAX_POWER_KW,
        efficiency=BATTERY_ROUND_TRIP_EFFICIENCY,
        horizon_hours=HORIZON_HOURS,
        steps_per_hour=4,
        n_scenarios=N_SCENARIOS,
        verbose=False,
        dla_chol=None,
        price_chol=None,
    ):
        self.capacity = battery_capacity
        self.max_power = max_power
        self.efficiency = efficiency
        self.horizon_steps = horizon_hours * steps_per_hour
        self.sqrt_eff = np.sqrt(efficiency)
        self.inv_sqrt_eff = 1.0 / self.sqrt_eff
        self.n_scenarios = n_scenarios
        self.verbose = verbose
        self.dla_chol = dla_chol    # Cholesky of residual correlation matrix
        self.price_chol = price_chol

    def optimize_single_scenario(
        self, prices, consumption, current_soc, discharge_allowed=False
    ):
        n = len(prices)
        charge = cp.Variable(n, nonneg=True)
        if discharge_allowed:
            discharge = cp.Variable(n, nonneg=True)
        soc = cp.Variable(n + 1, nonneg=True)

        # consumption is kWh per interval, convert to MWh and multiply by price
        cost = cp.sum(cp.multiply(prices, consumption / 1000))
        # battery charging cost (energy * efficiency loss)
        cost += cp.sum(cp.multiply(prices, charge / 1000)) * self.inv_sqrt_eff
        if discharge_allowed:
            # battery discharging offset (just offsets consumption, no revenue)
            cost -= cp.sum(cp.multiply(prices, discharge / 1000)) * self.inv_sqrt_eff

        constraints = [soc[0] == current_soc, soc <= self.capacity]
        for t in range(n):
            constraints.append(charge[t] <= self.max_power * 0.25)
            if discharge_allowed:
                constraints.append(discharge[t] <= self.max_power * 0.25)
                energy_delta = (
                    charge[t] * self.sqrt_eff - discharge[t] * self.inv_sqrt_eff
                )
            else:
                energy_delta = charge[t] * self.sqrt_eff
            constraints.append(soc[t + 1] == soc[t] - energy_delta)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            return None
        return {
            "cost": problem.value,
            "soc": np.array(soc.value).flatten(),
            "charge": np.array(charge.value).flatten(),
            "discharge": np.array(discharge.value).flatten()
            if discharge_allowed
            else np.zeros(n),
        }

    def stochastic_optimize(
        self,
        price_mean,
        price_std,
        dla_mean,
        dla_std,
        current_soc,
        discharge_allowed=False,
        verbose=False,
    ):
        costs, all_soc, all_charge, all_discharge = [], [], [], []
        n = len(price_mean)
        for _ in range(self.n_scenarios):
            # Sample correlated noise via Cholesky if available and dimensions match
            if self.price_chol is not None and self.price_chol.shape[0] == n:
                z = np.random.randn(n)
                price_noise = price_std * (self.price_chol @ z)
            else:
                price_noise = price_std * np.random.randn(n)

            if self.dla_chol is not None and self.dla_chol.shape[0] == n:
                z = np.random.randn(n)
                dla_noise = dla_std * (self.dla_chol @ z)
            else:
                dla_noise = dla_std * np.random.randn(n)

            prices = np.clip(price_mean + price_noise, 0.01, None)
            consumption = np.maximum(dla_mean + dla_noise, 0)
            result = self.optimize_single_scenario(
                prices, consumption, current_soc, discharge_allowed=discharge_allowed
            )
            if result:
                costs.append(result["cost"])
                all_soc.append(result["soc"])
                all_charge.append(result["charge"])
                all_discharge.append(result["discharge"])
        if not costs:
            return None
        return {
            "expected_cost": np.mean(costs),
            "cost_std": np.std(costs),
            "avg_soc": np.mean(all_soc, axis=0),
            "avg_charge": np.mean(all_charge, axis=0),
            "avg_discharge": np.mean(all_discharge, axis=0),
        }


def run_week_visualization(week_num=48):
    print("=" * 70)
    print(f"Evaluation and Visualization - Week {week_num}")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    merged = merge_datasets(FUNDIUM_DATA_PATH, PRICE_DATA_PATH)

    # Pre-compute abwaerme lag features on the full dataset so lag-672 (1wk) is valid
    for lag in ABWAERME_LAG_STEPS:
        merged[f"abwaerme_lag_{lag}"] = (
            merged["ofen_abwaerme_nestle_5893_mw"].shift(lag).ffill().bfill()
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
    dla_input_dim = 6
    price_input_dim = 8
    abwaerme_input_dim = 6 + len(ABWAERME_LAG_STEPS)

    dla_bnn = GaussianBNN(dla_input_dim, HIDDEN_DIMS)
    price_bnn = NormalizedBNN(price_input_dim, HIDDEN_DIMS)
    abwaerme_bnn = GaussianBNN(abwaerme_input_dim, HIDDEN_DIMS)

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
    abwaerme_mean = abwaerme_norm["mean"]
    abwaerme_std = abwaerme_norm["std"]

    print("\n[3/7] Making predictions...")
    time_feats = get_time_features(week_data["new_time"]).values

    renewable = week_data["renewable_gen_mw"].values
    load = week_data["load_mw"].values
    price_cols = np.column_stack([renewable, load])
    price_cols_mean = price_cols[:1000].mean(axis=0)
    price_cols_std = price_cols[:1000].std(axis=0)
    price_cols_norm = (price_cols - price_cols_mean) / (price_cols_std + 1e-8)

    batch_dla = torch.tensor(time_feats, dtype=torch.float32)
    price_features = np.hstack([time_feats, price_cols_norm])
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

    abwaerme_lag_cols = [f"abwaerme_lag_{lag}" for lag in ABWAERME_LAG_STEPS]
    abwaerme_lag_feats = week_data[abwaerme_lag_cols].values  # (N, 4)
    abwaerme_lag_feats_norm = (abwaerme_lag_feats - abwaerme_mean) / (abwaerme_std + 1e-8)
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

    # --- Figure 1: Predictions ---
    fig1, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig1.suptitle(f"Week {week_num} 2023 — Predictions", fontsize=14, fontweight="bold")

    axes[0, 0].plot(t_hours, dla_actual, "b-", label="Actual", linewidth=1.5)
    axes[0, 0].plot(t_hours, dla_mu, "r--", label="Predicted", linewidth=1.5)
    axes[0, 0].fill_between(
        t_hours,
        dla_mu - 1.96 * dla_sigma,
        dla_mu + 1.96 * dla_sigma,
        alpha=0.3,
        color="red",
        label="95% CI",
    )
    axes[0, 0].set_xlabel("Hours")
    axes[0, 0].set_ylabel("kWh")
    axes[0, 0].set_title("DLA Power Consumption")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(dla_actual, dla_mu, alpha=0.5, color="steelblue", s=20)
    max_val_dla = max(dla_actual.max(), dla_mu.max())
    axes[0, 1].plot([0, max_val_dla], [0, max_val_dla], "k--", label="Perfect")
    axes[0, 1].set_xlabel("Actual (kWh)")
    axes[0, 1].set_ylabel("Predicted (kWh)")
    axes[0, 1].set_title(f"DLA Predicted vs Actual (RMSE={dla_metrics['rmse']:.1f} kWh)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_hours, price_actual, "b-", label="Actual", linewidth=1.5)
    axes[1, 0].plot(t_hours, price_mu, "r--", label="Predicted", linewidth=1.5)
    axes[1, 0].fill_between(
        t_hours,
        price_mu - 1.96 * price_sigma,
        price_mu + 1.96 * price_sigma,
        alpha=0.3,
        color="red",
        label="95% CI",
    )
    axes[1, 0].set_xlabel("Hours")
    axes[1, 0].set_ylabel("EUR/MWh")
    axes[1, 0].set_title("Power Price (Germany EXAA)")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(price_actual, price_mu, alpha=0.5, color="darkorange", s=20)
    max_val_p = max(price_actual.max(), price_mu.max())
    min_val_p = min(price_actual.min(), price_mu.min())
    axes[1, 1].plot([min_val_p, max_val_p], [min_val_p, max_val_p], "k--", label="Perfect")
    axes[1, 1].set_xlabel("Actual (EUR/MWh)")
    axes[1, 1].set_ylabel("Predicted (EUR/MWh)")
    axes[1, 1].set_title(f"Price Predicted vs Actual (RMSE={price_metrics['rmse']:.2f} EUR/MWh)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(t_hours, abwaerme_actual, "b-", label="Actual", linewidth=1.5)
    axes[2, 0].plot(t_hours, abwaerme_mu, "r--", label="Predicted", linewidth=1.5)
    axes[2, 0].fill_between(
        t_hours,
        abwaerme_mu - 1.96 * abwaerme_sigma,
        abwaerme_mu + 1.96 * abwaerme_sigma,
        alpha=0.3,
        color="red",
        label="95% CI",
    )
    axes[2, 0].set_xlabel("Hours")
    axes[2, 0].set_ylabel("MW")
    axes[2, 0].set_title("Abwaerme Nestle (ofen_abwaerme_nestle_5893_mw)")
    axes[2, 0].legend(loc="upper right")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].scatter(abwaerme_actual, abwaerme_mu, alpha=0.5, color="purple", s=20)
    max_val_aw = max(abwaerme_actual.max(), abwaerme_mu.max())
    min_val_aw = min(abwaerme_actual.min(), abwaerme_mu.min())
    axes[2, 1].plot([min_val_aw, max_val_aw], [min_val_aw, max_val_aw], "k--", label="Perfect")
    axes[2, 1].set_xlabel("Actual (MW)")
    axes[2, 1].set_ylabel("Predicted (MW)")
    axes[2, 1].set_title(f"Abwaerme Predicted vs Actual (RMSE={abwaerme_metrics['rmse']:.4f} MW)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(f"week{week_num}_predictions.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to: week{week_num}_predictions.png")

    # --- Figure 2: Optimization results ---
    fig2, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig2.suptitle(f"Week {week_num} 2023 — Optimization Results", fontsize=14, fontweight="bold")

    soc_values = []
    soc_times = []
    for r_idx, r in enumerate(results):
        start_idx = result_times[r_idx]
        committed_soc = r["avg_soc"][:step_interval + 1]
        soc_timesteps = np.arange(len(committed_soc))
        soc_times.extend(start_idx + soc_timesteps)
        soc_values.extend(committed_soc)

    if soc_values:
        soc_times_hours = np.array(soc_times) * 0.25
        axes[0, 0].plot(soc_times_hours, soc_values, "g-", linewidth=2)
        axes[0, 0].axhline(
            y=BATTERY_CAPACITY_KWH, color="r", linestyle="--", alpha=0.5, label="Max"
        )
        axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Min")
    axes[0, 0].set_xlabel("Hours")
    axes[0, 0].set_ylabel("SOC (kWh)")
    axes[0, 0].set_title("Battery State of Charge")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, BATTERY_CAPACITY_KWH * 1.1)

    all_charge = []
    all_discharge = []
    charge_times = []
    for r_idx, r in enumerate(results):
        start_idx = result_times[r_idx]
        for t in range(step_interval):
            charge_times.append(start_idx + t)
            all_charge.append(r["avg_charge"][t])
            all_discharge.append(r["avg_discharge"][t])

    if all_charge:
        charge_times_hours = np.array(charge_times) * 0.25
        axes[0, 1].fill_between(
            charge_times_hours, 0, all_charge, color="green", alpha=0.5, label="Charge"
        )
        axes[0, 1].fill_between(
            charge_times_hours,
            0,
            [-d for d in all_discharge],
            color="red",
            alpha=0.5,
            label="Discharge",
        )
    axes[0, 1].set_xlabel("Hours")
    axes[0, 1].set_ylabel("Power (kW)")
    axes[0, 1].set_title("Battery Charge/Discharge Schedule")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_hours, price_actual, "b-", label="Price", linewidth=1.5)
    if soc_values:
        ax_soc = axes[1, 0].twinx()
        ax_soc.plot(soc_times_hours, soc_values, "g--", linewidth=1.5, alpha=0.7, label="SOC")
        ax_soc.set_ylabel("SOC (kWh)", color="green")
        ax_soc.tick_params(axis="y", labelcolor="green")
        ax_soc.set_ylim(0, BATTERY_CAPACITY_KWH * 1.1)
    axes[1, 0].set_xlabel("Hours")
    axes[1, 0].set_ylabel("EUR/MWh")
    axes[1, 0].set_title("Price vs Battery SOC")
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(
        ["Baseline\n(no battery)", "Optimized\n(with battery)"],
        [baseline_cost, actual_optimized_cost],
        color=["gray", "green"],
        alpha=0.7,
    )
    axes[1, 1].set_ylabel("Cost (EUR)")
    savings = baseline_cost - actual_optimized_cost
    axes[1, 1].set_title(
        f"Weekly Costs: {savings:.0f} EUR savings ({100 * (1 - actual_optimized_cost / baseline_cost):.1f}%)"
    )
    for i, (cost, label) in enumerate(
        [(baseline_cost, "Baseline"), (actual_optimized_cost, "Optimized")]
    ):
        axes[1, 1].text(i, cost + 100, f"{cost:.0f}", ha="center", fontsize=10)

    fig2.tight_layout()
    fig2.savefig(f"week{week_num}_optimization.png", dpi=150, bbox_inches="tight")
    print(f"  Saved to: week{week_num}_optimization.png")

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
