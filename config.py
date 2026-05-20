# Configuration for DLA stochastic optimization system
# All tunable hyperparameters, paths, and method switches live here.

import os

# ── Output directories ────────────────────────────────────────────────────────
# Trained model weights (.pt) and normalization stats (.npz) are stored in
# separate subdirectories so the project root stays clean.
MODELS_DIR = "models"
NORMS_DIR  = "norms"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NORMS_DIR,  exist_ok=True)

# ── Seed ────────────────────────────────────────────────────────────
SEED = 45313044

# ── Evaluation window ────────────────────────────────────────────────────────
# Start date (inclusive) and length for run_week_visualization.
# Any period within 2023 works; EVAL_N_DAYS can be shorter or longer than 7.
EVAL_START_DATE = "2023-09-27"   # ISO date string, e.g. '2023-11-27' for week 48 of 2023
EVAL_N_DAYS     = 1              # number of calendar days to evaluate

# ── Battery parameters ────────────────────────────────────────────────────────
BATTERY_CAPACITY_KWH = 500.0
BATTERY_MAX_POWER_KW = 100.0
BATTERY_ROUND_TRIP_EFFICIENCY = 0.9

# ── MPC / optimization ────────────────────────────────────────────────────────
HORIZON_HOURS = 12                          # look-ahead window for the LP
OPTIMIZATION_STEPS_PER_HOUR = 4            # 15-minute intervals
N_SCENARIOS = 20                            # Monte-Carlo scenarios per LP solve
OPT_FREQUENCY = 3 * OPTIMIZATION_STEPS_PER_HOUR # re-optimize every (eg 3 hours (12 steps))

# Optimisation mode — choose one:
#   "extensive"         — hard SOC constraints, all scenarios share one schedule (recommended)
#   "mean_scenario"     — single LP on scenario means (fastest, ignores uncertainty)
#   "scenario_avg"      — independent LP per scenario, schedules averaged (legacy)
#   "cvar"              — CVaR-relaxed SOC constraints at risk level RISK_EPSILON
#   "scenario_approach" — auto-compute N via Campi-Garatti bound, hard constraints
SO_MODE = "cvar"

# ── Probabilistic constraint settings ────────────────────────────────────────
# Used by SO_MODE "cvar" and "scenario_approach".
RISK_EPSILON = 0.5   # max allowed SOC violation probability  (e.g. 0.05 = 5%)
RISK_DELTA   = 0.1   # confidence level for Campi-Garatti N   (e.g. 0.01 = 99% confidence)

# ── Model hyperparameters ─────────────────────────────────────────────────────
HIDDEN_DIMS = [64, 64, 32]                 # BNN / MC-Dropout hidden layer sizes
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_MC_SAMPLES = 50                        # forward passes for MC uncertainty

# ── Uncertainty method ────────────────────────────────────────────────────────
#   "mcd" = MC Dropout (fast, approximate epistemic uncertainty)
#   "bnn" = Bayes by Backprop (true weight posteriors, slower)
UNCERTAINTY_METHOD = "mcd"

# ── Scenario generation method ────────────────────────────────────────────────
#   "cholesky_decomp_corr"   – sample correlated Gaussian noise via Cholesky of
#                              the residual correlation matrix estimated on training data
#   "full_norm_flow_fixed"   – use a conditional affine normalizing flow (fixed weights)
#   "full_norm_flow_bayes"   – Bayesian version (not yet implemented)
SCENARIO_METHOD = "cholesky_decomp_corr"

# Hidden layer sizes for the normalizing flow encoder MLP.
# Wider than HIDDEN_DIMS because the output head (Cholesky entries) is much larger.
FLOW_HIDDEN_DIMS = [128, 128, 64]

# ── Data paths ────────────────────────────────────────────────────────────────
FUNDIUM_DATA_PATH = "fondium_15_min_data_2023.csv"
PRICE_DATA_PATH = (
    "energy-charts_Stromproduktion_und_Börsenstrompreise_in_Deutschland_2023.csv"
)

# ── Train / test split ────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.5

# ── Lag feature steps (in 15-min intervals) ───────────────────────────────────
# 48 = 12 h,  96 = 24 h,  672 = 1 week
# Must be >= HORIZON_HOURS * OPTIMIZATION_STEPS_PER_HOUR to avoid data leakage.
ABWAERME_LAG_STEPS = [48, 96, 672]
DLA_LAG_STEPS      = [48, 96, 672]
PL2_LAG_STEPS      = [48, 96, 672]
PRICE_LAG_STEPS    = [48, 96, 672]

_horizon = HORIZON_HOURS * OPTIMIZATION_STEPS_PER_HOUR
# Check lag steps against optimization horizon 
# and warn if any are smaller (potential data leakage)
for _name, _steps in [("ABWAERME", ABWAERME_LAG_STEPS), ("DLA", DLA_LAG_STEPS), ("PL2", PL2_LAG_STEPS), ("PRICE", PRICE_LAG_STEPS)]:
    if any(s < _horizon for s in _steps):
        print(f"Warning: {_name} lag steps contain values smaller than the optimization horizon ({_horizon}). Data leakage may occur.")
