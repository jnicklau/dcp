import numpy as np
import matplotlib.pyplot as plt

from config import BATTERY_CAPACITY_KWH


def plot_predictions(
    week_num,
    t_hours,
    dla_actual, dla_mu, dla_sigma, dla_total_std, dla_metrics,
    price_actual, price_mu, price_sigma, price_total_std, price_metrics,
    abwaerme_actual, abwaerme_mu, abwaerme_sigma, abwaerme_total_std, abwaerme_metrics,
):
    """Figure 1: time-series predictions with split aleatoric/epistemic uncertainty bands
    and predicted-vs-actual scatter plots for all three signals."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(f"Week {week_num} 2023 — Predictions", fontsize=14, fontweight="bold")

    # ── DLA ──────────────────────────────────────────────────────────────────
    axes[0, 0].plot(t_hours, dla_actual, "b-", label="Actual", linewidth=1.5)
    axes[0, 0].plot(t_hours, dla_mu, "r--", label="Predicted", linewidth=1.5)
    axes[0, 0].fill_between(
        t_hours,
        dla_mu - 1.96 * dla_sigma,
        dla_mu + 1.96 * dla_sigma,
        alpha=0.45, color="orange", label="Aleatoric (95%)",
    )
    axes[0, 0].fill_between(
        t_hours,
        dla_mu + 1.96 * dla_sigma,
        dla_mu + 1.96 * dla_total_std,
        alpha=0.3, color="red", label="Epistemic (95%)",
    )
    axes[0, 0].fill_between(
        t_hours,
        dla_mu - 1.96 * dla_total_std,
        dla_mu - 1.96 * dla_sigma,
        alpha=0.3, color="red",
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

    # ── Price ─────────────────────────────────────────────────────────────────
    axes[1, 0].plot(t_hours, price_actual, "b-", label="Actual", linewidth=1.5)
    axes[1, 0].plot(t_hours, price_mu, "r--", label="Predicted", linewidth=1.5)
    axes[1, 0].fill_between(
        t_hours,
        price_mu - 1.96 * price_sigma,
        price_mu + 1.96 * price_sigma,
        alpha=0.45, color="orange", label="Aleatoric (95%)",
    )
    axes[1, 0].fill_between(
        t_hours,
        price_mu + 1.96 * price_sigma,
        price_mu + 1.96 * price_total_std,
        alpha=0.3, color="red", label="Epistemic (95%)",
    )
    axes[1, 0].fill_between(
        t_hours,
        price_mu - 1.96 * price_total_std,
        price_mu - 1.96 * price_sigma,
        alpha=0.3, color="red",
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

    # ── Abwärme ───────────────────────────────────────────────────────────────
    axes[2, 0].plot(t_hours, abwaerme_actual, "b-", label="Actual", linewidth=1.5)
    axes[2, 0].plot(t_hours, abwaerme_mu, "r--", label="Predicted", linewidth=1.5)
    axes[2, 0].fill_between(
        t_hours,
        abwaerme_mu - 1.96 * abwaerme_sigma,
        abwaerme_mu + 1.96 * abwaerme_sigma,
        alpha=0.45, color="orange", label="Aleatoric (95%)",
    )
    axes[2, 0].fill_between(
        t_hours,
        abwaerme_mu + 1.96 * abwaerme_sigma,
        abwaerme_mu + 1.96 * abwaerme_total_std,
        alpha=0.3, color="red", label="Epistemic (95%)",
    )
    axes[2, 0].fill_between(
        t_hours,
        abwaerme_mu - 1.96 * abwaerme_total_std,
        abwaerme_mu - 1.96 * abwaerme_sigma,
        alpha=0.3, color="red",
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

    fig.tight_layout()
    path = f"week{week_num}_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {path}")
    plt.close(fig)


def plot_optimization(
    week_num,
    t_hours,
    price_actual,
    results,
    result_times,
    step_interval,
    baseline_cost,
    actual_optimized_cost,
):
    """Figure 2: battery SOC, charge/discharge schedule, price vs SOC, and cost bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Week {week_num} 2023 — Optimization Results", fontsize=14, fontweight="bold")

    # ── SOC trajectory ────────────────────────────────────────────────────────
    soc_values, soc_times = [], []
    for r_idx, r in enumerate(results):
        start_idx = result_times[r_idx]
        committed_soc = r["avg_soc"][:step_interval + 1]
        soc_timesteps = np.arange(len(committed_soc))
        soc_times.extend(start_idx + soc_timesteps)
        soc_values.extend(committed_soc)

    if soc_values:
        soc_times_hours = np.array(soc_times) * 0.25
        axes[0, 0].plot(soc_times_hours, soc_values, "g-", linewidth=2)
        axes[0, 0].axhline(y=BATTERY_CAPACITY_KWH, color="r", linestyle="--", alpha=0.5, label="Max")
        axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Min")
    axes[0, 0].set_xlabel("Hours")
    axes[0, 0].set_ylabel("SOC (kWh)")
    axes[0, 0].set_title("Battery State of Charge")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, BATTERY_CAPACITY_KWH * 1.1)

    # ── Charge / discharge schedule ───────────────────────────────────────────
    all_charge, all_discharge, charge_times = [], [], []
    for r_idx, r in enumerate(results):
        start_idx = result_times[r_idx]
        for t in range(step_interval):
            charge_times.append(start_idx + t)
            all_charge.append(r["avg_charge"][t])
            all_discharge.append(r["avg_discharge"][t])

    if all_charge:
        charge_times_hours = np.array(charge_times) * 0.25
        axes[0, 1].fill_between(charge_times_hours, 0, all_charge, color="green", alpha=0.5, label="Charge")
        axes[0, 1].fill_between(charge_times_hours, 0, [-d for d in all_discharge], color="red", alpha=0.5, label="Discharge")
    axes[0, 1].set_xlabel("Hours")
    axes[0, 1].set_ylabel("Power (kW)")
    axes[0, 1].set_title("Battery Charge/Discharge Schedule")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ── Price vs SOC ──────────────────────────────────────────────────────────
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

    # ── Cost bar chart ────────────────────────────────────────────────────────
    savings = baseline_cost - actual_optimized_cost
    axes[1, 1].bar(
        ["Baseline\n(no battery)", "Optimized\n(with battery)"],
        [baseline_cost, actual_optimized_cost],
        color=["gray", "green"],
        alpha=0.7,
    )
    axes[1, 1].set_ylabel("Cost (EUR)")
    axes[1, 1].set_title(
        f"Weekly Costs: {savings:.0f} EUR savings ({100 * (1 - actual_optimized_cost / baseline_cost):.1f}%)"
    )
    for i, cost in enumerate([baseline_cost, actual_optimized_cost]):
        axes[1, 1].text(i, cost + 100, f"{cost:.0f}", ha="center", fontsize=10)

    fig.tight_layout()
    path = f"week{week_num}_optimization.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {path}")
    plt.close(fig)
