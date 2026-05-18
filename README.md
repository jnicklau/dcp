# DeepCarbPlanner

Receding-horizon MPC battery scheduler with probabilistic load and price forecasts (MC Dropout or Bayes by Backprop) feeding a stochastic LP solved via CVXPY/ECOS.

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt                 # Python 3.12.0

python train_simple.py    # train BNNs, save weights + Cholesky matrices
python eval_visualize.py  # run MPC for week 48, save plots
```

---

## Configuration (`config.py`)

All behaviour is controlled from `config.py`.

### Battery
| Parameter | Default | Description |
|---|---|---|
| `BATTERY_CAPACITY_KWH` | `500.0` | Usable capacity (kWh) |
| `BATTERY_MAX_POWER_KW` | `100.0` | Peak charge/discharge power (kW) |
| `BATTERY_ROUND_TRIP_EFFICIENCY` | `0.90` | Round-trip efficiency η (applied as √η per side) |

### MPC / optimisation
| Parameter | Default | Description |
|---|---|---|
| `HORIZON_HOURS` | `12` | LP look-ahead window |
| `OPT_FREQUENCY` | `12` | Steps committed per iteration (3 h at 15-min resolution) |
| `N_SCENARIOS` | `10` | Scenarios per LP solve (ignored by `scenario_approach`) |
| `SO_MODE` | `"extensive"` | Optimisation mode — see table below |
| `RISK_EPSILON` | `0.05` | Max SOC violation probability for `cvar` / `scenario_approach` |
| `RISK_DELTA` | `0.10` | Campi-Garatti confidence level for `scenario_approach` |

### Optimisation modes
| `SO_MODE` | Description |
|---|---|
| `"extensive"` | All scenarios share one schedule; hard SOC constraints per scenario **(recommended)** |
| `"mean_scenario"` | Single LP on expected inputs — fastest, ignores uncertainty |
| `"scenario_avg"` | Independent LP per scenario, schedules averaged — legacy |
| `"cvar"` | CVaR-relaxed SOC bounds at level `RISK_EPSILON`; soft tail constraints instead of hard limits |
| `"scenario_approach"` | Auto-samples N scenarios via Campi-Garatti bound; guarantees (1−ε) feasibility with (1−δ) confidence |

> **Note:** `scenario_approach` requires O(1000) scenarios for typical horizons and is slow. `cvar` with small `N_SCENARIOS` is the practical alternative.

### Forecasting
| Parameter | Default | Description |
|---|---|---|
| `UNCERTAINTY_METHOD` | `"mcd"` | `"mcd"` = MC Dropout, `"bnn"` = Bayes by Backprop |
| `NUM_MC_SAMPLES` | `50` | Forward passes for MC uncertainty |
| `HIDDEN_DIMS` | `[64, 64, 32]` | Hidden layer widths |
| `NUM_EPOCHS` | `10` | Training epochs |
| `SCENARIO_METHOD` | `"cholesky_decomp_corr"` | Scenario sampling (Cholesky-correlated Gaussian) |

### Lag features
All signals use lags at 48 / 96 / 672 steps (12 h, 24 h, 1 week). Configured via `DLA_LAG_STEPS`, `PRICE_LAG_STEPS`, `ABWAERME_LAG_STEPS`.

---

## Optimizer API

Scenario generation and scheduling are decoupled — any `(prices, consumption)` trajectory source works.

```python
from stoch_optimizer import StochasticOptimizer

opt = StochasticOptimizer(battery_capacity=500, max_power=100, efficiency=0.9)

# Option A — bring your own scenarios
result = opt.optimize(my_scenarios, current_soc=250.0, discharge_allowed=True, mode="extensive")

# Option B — sample via built-in Cholesky sampler, then optimise
scenarios = opt.sample_scenarios(price_mean, price_std, dla_mean, dla_std)
result = opt.optimize(scenarios, current_soc=250.0)

# Option C — one-liner; required for scenario_approach (auto-computes N)
result = opt.optimize_from_stats(price_mean, price_std, dla_mean, dla_std,
                                  current_soc=250.0, mode="scenario_approach")
```

`result` keys: `expected_cost`, `cost_std`, `avg_soc`, `avg_charge`, `avg_discharge`.
`cvar` adds `epsilon`; `scenario_approach` adds `campi_garatti_n_required`, `campi_garatti_satisfied`, `epsilon`, `delta`.

---

## Repository structure

```
config.py            # all hyperparameters and method switches
train_simple.py      # BNN training + Cholesky estimation
eval_visualize.py    # MPC evaluation pipeline → plots
stoch_optimizer.py   # stochastic LP (all modes)
data_loader.py       # CSV loading, feature engineering
plots.py             # plotting helpers
models/              # trained weights *.pt  (git-ignored)
norms/               # normalisation stats *.npz  (git-ignored)
```

---
