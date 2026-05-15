"""
plots.py — Visualisation helpers for the stochastic battery optimisation pipeline.

Two public functions:
  plot_predictions(...)  — time-series forecasts with aleatoric/epistemic bands
  plot_optimization(...) — price & SOC vs time, net power schedule, price-SOC-power scatter, cost bar
"""

import numpy as np
import matplotlib.pyplot as plt

from config import BATTERY_CAPACITY_KWH, BATTERY_MAX_POWER_KW


# ── Prediction plots ──────────────────────────────────────────────────────────

def plot_predictions(
    week_num,
    t_hours,
    dla_actual, dla_mu, dla_sigma, dla_total_std, dla_metrics,
    price_actual, price_mu, price_sigma, price_total_std, price_metrics,
    abwaerme_actual, abwaerme_mu, abwaerme_sigma, abwaerme_total_std, abwaerme_metrics,
):
    """3×2 figure: time-series forecasts + predicted-vs-actual scatter for DLA, price, Abwärme.

    Orange band = aleatoric (95%), red band extension = epistemic (95%).
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(f"Week {week_num} 2023 — Predictions", fontsize=14, fontweight="bold")

    # ── DLA ───────────────────────────────────────────────────────────────────
    _plot_series_row(
        axes[0], t_hours,
        dla_actual, dla_mu, dla_sigma, dla_total_std,
        label_y="kWh", title="DLA Power Consumption",
        scatter_color="steelblue",
        scatter_title=f"DLA Predicted vs Actual (RMSE={dla_metrics['rmse']:.1f} kWh)",
        scatter_xlabel="Actual (kWh)", scatter_ylabel="Predicted (kWh)",
    )

    # ── Price ──────────────────────────────────────────────────────────────────
    _plot_series_row(
        axes[1], t_hours,
        price_actual, price_mu, price_sigma, price_total_std,
        label_y="EUR/MWh", title="Power Price (Germany EXAA)",
        scatter_color="darkorange",
        scatter_title=f"Price Predicted vs Actual (RMSE={price_metrics['rmse']:.2f} EUR/MWh)",
        scatter_xlabel="Actual (EUR/MWh)", scatter_ylabel="Predicted (EUR/MWh)",
    )

    # ── Abwärme ────────────────────────────────────────────────────────────────
    _plot_series_row(
        axes[2], t_hours,
        abwaerme_actual, abwaerme_mu, abwaerme_sigma, abwaerme_total_std,
        label_y="MW", title="Abwärme Nestlé (ofen_abwaerme_nestle_5893_mw)",
        scatter_color="purple",
        scatter_title=f"Abwärme Predicted vs Actual (RMSE={abwaerme_metrics['rmse']:.4f} MW)",
        scatter_xlabel="Actual (MW)", scatter_ylabel="Predicted (MW)",
    )

    fig.tight_layout()
    path = f"week{week_num}_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _plot_series_row(
    axes_row, t_hours,
    actual, mu, sigma, total_std,
    label_y, title, scatter_color,
    scatter_title, scatter_xlabel, scatter_ylabel,
):
    """Helper: fill one row (time-series left, scatter right)."""
    ax_ts, ax_sc = axes_row

    # Time-series with split uncertainty bands
    ax_ts.plot(t_hours, actual, "b-", label="Actual", linewidth=1.5)
    ax_ts.plot(t_hours, mu, "r--", label="Predicted", linewidth=1.5)
    ax_ts.fill_between(t_hours, mu - 1.96 * sigma,     mu + 1.96 * sigma,
                       alpha=0.45, color="orange", label="Aleatoric (95%)")
    ax_ts.fill_between(t_hours, mu + 1.96 * sigma,     mu + 1.96 * total_std,
                       alpha=0.30, color="red",    label="Epistemic (95%)")
    ax_ts.fill_between(t_hours, mu - 1.96 * total_std, mu - 1.96 * sigma,
                       alpha=0.30, color="red")
    ax_ts.set_xlabel("Hours")
    ax_ts.set_ylabel(label_y)
    ax_ts.set_title(title)
    ax_ts.legend(loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    # Predicted vs actual scatter
    ax_sc.scatter(actual, mu, alpha=0.5, color=scatter_color, s=20)
    lo = min(actual.min(), mu.min())
    hi = max(actual.max(), mu.max())
    ax_sc.plot([lo, hi], [lo, hi], "k--", label="Perfect")
    ax_sc.set_xlabel(scatter_xlabel)
    ax_sc.set_ylabel(scatter_ylabel)
    ax_sc.set_title(scatter_title)
    ax_sc.legend()
    ax_sc.grid(True, alpha=0.3)


# ── Optimization plots ────────────────────────────────────────────────────────

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
    """2×2 figure: price & SOC vs time, net power schedule, price-SOC-power scatter, cost bar chart."""
    battery_color = "darkorange"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Week {week_num} 2023 — Optimization Results", fontsize=14, fontweight="bold")

    # ── Gather committed arrays ────────────────────────────────────────────────
    soc_values, soc_times = [], []
    all_charge, all_discharge, charge_times = [], [], []
    soc_at_step = []  # SOC at the start of each committed step (aligned with charge_times)

    for r_idx, r in enumerate(results):
        start_idx = result_times[r_idx]
        for t, soc in enumerate(r["avg_soc"][:step_interval + 1]):
            soc_times.append(start_idx + t)
            soc_values.append(soc)
        for t in range(step_interval):
            charge_times.append(start_idx + t)
            all_charge.append(r["avg_charge"][t])
            all_discharge.append(r["avg_discharge"][t])
            soc_at_step.append(r["avg_soc"][t])

    soc_times_hours    = np.array(soc_times)   * 0.25
    charge_times_hours = np.array(charge_times) * 0.25

    # ── (0,0) Price & SOC vs time (merged dual-axis) ───────────────────────────
    axes[0, 0].plot(t_hours, price_actual, "steelblue", linewidth=1.5, label="Price (EUR/MWh)")
    axes[0, 0].set_xlabel("Hours")
    axes[0, 0].set_ylabel("EUR/MWh", color="steelblue")
    axes[0, 0].tick_params(axis="y", labelcolor="steelblue")
    axes[0, 0].set_title("Price & Battery SOC")
    axes[0, 0].grid(True, alpha=0.3)
    if soc_values:
        ax_soc = axes[0, 0].twinx()
        ax_soc.plot(soc_times_hours, soc_values, color=battery_color, linewidth=2, label="SOC (kWh)")
        ax_soc.axhline(BATTERY_CAPACITY_KWH, color="r", linestyle="--", alpha=0.4)
        ax_soc.axhline(0,                    color="r", linestyle="--", alpha=0.4)
        ax_soc.set_ylabel("SOC (kWh)", color=battery_color)
        ax_soc.tick_params(axis="y", labelcolor=battery_color)
        ax_soc.set_ylim(0, BATTERY_CAPACITY_KWH * 1.1)
        lines  = axes[0, 0].get_lines() + ax_soc.get_lines()
        labels = [l.get_label() for l in lines]
        axes[0, 0].legend(lines, labels, loc="upper left", fontsize=8)

    # ── (0,1) Net battery power schedule ──────────────────────────────────────
    _first_plan = True
    for r_idx, r in enumerate(results):
        start_idx = result_times[r_idx]
        plan_len  = len(r["avg_charge"])
        plan_times_h = (np.arange(plan_len) + start_idx) * 0.25
        net_plan  = r["avg_charge"] - r["avg_discharge"]
        label_plan = "Planned (per MPC window)" if _first_plan else None
        axes[0, 1].plot(plan_times_h, net_plan, color=battery_color, alpha=0.15,
                        linewidth=1, label=label_plan)
        _first_plan = False
    if all_charge:
        net_realized = np.array(all_charge) - np.array(all_discharge)
        axes[0, 1].plot(charge_times_hours, net_realized, color=battery_color,
                        linewidth=1.8, label="Realized")
    axes[0, 1].axhline(0, color="black", linewidth=0.6, alpha=0.4)
    axes[0, 1].set_xlabel("Hours")
    axes[0, 1].set_ylabel("Net power (kW)  [+ = charge, − = discharge]")
    axes[0, 1].set_title("Battery Net Power Schedule")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # ── (1,0) Scatter: SOC vs price, colored by net power ─────────────────────
    if soc_at_step:
        soc_arr      = np.array(soc_at_step)
        net_arr      = np.array(all_charge) - np.array(all_discharge)
        price_at_step = price_actual[np.array(charge_times)]
        clim = max(abs(net_arr.max()), abs(net_arr.min()), 1.0)
        sc = axes[1, 0].scatter(
            soc_arr, price_at_step, c=net_arr,
            cmap="RdYlGn", s=18, alpha=0.75,
            vmin=-clim, vmax=clim,
        )
        fig.colorbar(sc, ax=axes[1, 0], label="Net power (kW)  [+ charge / − discharge]")
    axes[1, 0].set_xlabel("SOC (kWh)")
    axes[1, 0].set_ylabel("Price (EUR/MWh)")
    axes[1, 0].set_title("Price vs SOC (colored by net power)")
    axes[1, 0].grid(True, alpha=0.3)

    # ── (1,1) Cost comparison bar chart ───────────────────────────────────────
    savings   = baseline_cost - actual_optimized_cost
    pct_saved = 100 * savings / baseline_cost if baseline_cost else 0
    axes[1, 1].bar(
        ["Baseline\n(no battery)", "Optimized\n(with battery)"],
        [baseline_cost, actual_optimized_cost],
        color=["gray", battery_color], alpha=0.7,
    )
    axes[1, 1].set_ylabel("Cost (EUR)")
    axes[1, 1].set_title(f"Weekly Costs: {savings:.0f} EUR savings ({pct_saved:.1f}%)")
    for i, cost in enumerate([baseline_cost, actual_optimized_cost]):
        axes[1, 1].text(i, cost + max(abs(baseline_cost) * 0.01, 1), f"{cost:.0f}",
                        ha="center", fontsize=10)

    fig.tight_layout()
    path = f"week{week_num}_optimization.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)
