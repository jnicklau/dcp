"""
correlation_analysis.py — Temporal correlation analysis for Fondium plant signals.

For each signal, generates a combined figure with:
  Left column  : mean profile ± std for hour-of-day, day-of-week, month-of-year
  Right column : Partial ACF (PACF) up to 96 lags (24 h at 15-min resolution)

Signals analysed:
  1. DLA power consumption       (dla_stromverbrauch_kwh)
  2. Formanlage PL2 power        (pl2_stromverbrauch_kwh)
  3. Oven G-Koks demand          (ofen_g_koks_kg)
  4. Oven F-Koks demand          (ofen_f_koks_kg)

Output: one PNG per signal, saved to the working directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_pacf

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "fondium_15_min_data_2023.csv"
TIMESTAMP   = "new_time"
PACF_LAGS   = 96          # 24 h at 15-min resolution
PACF_METHOD = "ywm"       # Yule-Walker (more stable for large lags)
OUTPUT_DIR  = "."

SIGNALS = [
    {
        "col":   "dla_stromverbrauch_kwh",
        "label": "DLA Power Consumption (kWh / 15 min)",
        "fname": "corr_dla_power.png",
        "color": "steelblue",
    },
    {
        "col":   "pl2_stromverbrauch_kwh",
        "label": "Formanlage PL2 Power (kWh / 15 min)",
        "fname": "corr_pl2_power.png",
        "color": "darkorange",
    },
    {
        "col":   "ofen_g_koks_kg",
        "label": "Oven G-Koks Demand (kg / 15 min)",
        "fname": "corr_ofen_g_koks.png",
        "color": "firebrick",
    },
    {
        "col":   "ofen_f_koks_kg",
        "label": "Oven F-Koks Demand (kg / 15 min)",
        "fname": "corr_ofen_f_koks.png",
        "color": "purple",
    },
]

DAY_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=[TIMESTAMP])
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)
    df["hour"]       = df[TIMESTAMP].dt.hour + df[TIMESTAMP].dt.minute / 60
    df["hour_int"]   = df[TIMESTAMP].dt.hour * 4 + df[TIMESTAMP].dt.minute // 15
    df["dayofweek"]  = df[TIMESTAMP].dt.dayofweek   # 0=Mon … 6=Sun
    df["month"]      = df[TIMESTAMP].dt.month        # 1–12
    return df


# ── Profile helper ─────────────────────────────────────────────────────────────
def mean_profile(df, group_col, value_col):
    grp   = df.groupby(group_col)[value_col]
    mu    = grp.mean()
    sigma = grp.std()
    return mu, sigma


# ── Plot one signal ────────────────────────────────────────────────────────────
def plot_signal(df, sig):
    col   = sig["col"]
    label = sig["label"]
    color = sig["color"]
    fname = sig["fname"]

    series = df[col].dropna()

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Temporal Correlation Analysis — {label}", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax_tod  = fig.add_subplot(gs[0, 0])   # hour of day
    ax_dow  = fig.add_subplot(gs[1, 0])   # day of week
    ax_moy  = fig.add_subplot(gs[2, 0])   # month of year
    ax_pacf = fig.add_subplot(gs[:, 1])   # PACF spans all three rows

    # ── Hour of day ───────────────────────────────────────────────────────────
    mu_h, std_h = mean_profile(df.dropna(subset=[col]), "hour_int", col)
    x_h = mu_h.index / 4          # convert 15-min slot index to hours
    ax_tod.plot(x_h, mu_h.values, color=color, linewidth=1.8, label="Mean")
    ax_tod.fill_between(x_h,
                        mu_h.values - std_h.values,
                        mu_h.values + std_h.values,
                        alpha=0.25, color=color, label="±1 std")
    ax_tod.set_xlabel("Hour of day")
    ax_tod.set_ylabel(label.split("(")[0].strip())
    ax_tod.set_title("Mean profile — Hour of day")
    ax_tod.set_xticks(range(0, 25, 3))
    ax_tod.legend(fontsize=8)
    ax_tod.grid(True, alpha=0.3)

    # ── Day of week ───────────────────────────────────────────────────────────
    mu_d, std_d = mean_profile(df.dropna(subset=[col]), "dayofweek", col)
    ax_dow.bar(mu_d.index, mu_d.values, color=color, alpha=0.7, label="Mean")
    ax_dow.errorbar(mu_d.index, mu_d.values, yerr=std_d.values,
                    fmt="none", color="black", capsize=4, linewidth=1)
    ax_dow.set_xticks(range(7))
    ax_dow.set_xticklabels(DAY_NAMES)
    ax_dow.set_xlabel("Day of week")
    ax_dow.set_ylabel(label.split("(")[0].strip())
    ax_dow.set_title("Mean profile — Day of week")
    ax_dow.grid(True, alpha=0.3, axis="y")

    # ── Month of year ─────────────────────────────────────────────────────────
    mu_m, std_m = mean_profile(df.dropna(subset=[col]), "month", col)
    ax_moy.bar(mu_m.index, mu_m.values, color=color, alpha=0.7, label="Mean")
    ax_moy.errorbar(mu_m.index, mu_m.values, yerr=std_m.values,
                    fmt="none", color="black", capsize=4, linewidth=1)
    ax_moy.set_xticks(range(1, 13))
    ax_moy.set_xticklabels(MONTH_NAMES, rotation=30, ha="right")
    ax_moy.set_xlabel("Month")
    ax_moy.set_ylabel(label.split("(")[0].strip())
    ax_moy.set_title("Mean profile — Month of year")
    ax_moy.grid(True, alpha=0.3, axis="y")

    # ── PACF ──────────────────────────────────────────────────────────────────
    clean = series.interpolate(method="linear").ffill().bfill()
    plot_pacf(clean, lags=PACF_LAGS, method=PACF_METHOD, ax=ax_pacf,
              color=color, alpha=0.6, title="")
    ax_pacf.set_title(f"Partial ACF  (lags 0–{PACF_LAGS}, Δt = 15 min)")
    ax_pacf.set_xlabel("Lag (15-min steps)")
    ax_pacf.set_ylabel("Partial autocorrelation")
    # Mark diurnal and weekly lag lines
    for lag, lbl in [(48, "12 h"), (96, "24 h"), (672, "1 week")]:
        if lag <= PACF_LAGS:
            ax_pacf.axvline(lag, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_pacf.text(lag + 0.5, ax_pacf.get_ylim()[1] * 0.95, lbl,
                         fontsize=7, color="gray", va="top")
    ax_pacf.grid(True, alpha=0.3)

    out_path = f"{OUTPUT_DIR}/{fname}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows  ({df[TIMESTAMP].min().date()} – {df[TIMESTAMP].max().date()})")

    for sig in SIGNALS:
        if sig["col"] not in df.columns:
            print(f"  Skipping '{sig['col']}' — column not found in dataset.")
            continue
        n_valid = df[sig["col"]].notna().sum()
        print(f"\nAnalysing: {sig['label']}  ({n_valid} non-null values)")
        plot_signal(df, sig)

    print("\nDone.")
