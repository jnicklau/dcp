import numpy as np
import cvxpy as cp

from config import (
    BATTERY_CAPACITY_KWH,
    BATTERY_MAX_POWER_KW,
    BATTERY_ROUND_TRIP_EFFICIENCY,
    HORIZON_HOURS,
    N_SCENARIOS,
)


class StochasticOptimizer:
    """
    Receding-horizon stochastic battery scheduler.

    At each MPC step, N_SCENARIOS LP instances are solved over the look-ahead
    horizon.  Scenarios are drawn from the BNN posterior predictive distributions,
    optionally with Cholesky-correlated temporal noise.  Only the mean schedule
    across all scenarios is returned; the caller commits the first OPT_FREQUENCY
    steps before re-solving.
    """

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

    # ── Single-scenario LP ────────────────────────────────────────────────────

    def optimize_single_scenario(
        self, prices, consumption, current_soc, discharge_allowed=False
    ):
        """Solve one LP for a deterministic (price, consumption) scenario.

        Objective: minimise total energy cost over the horizon.
          cost = Σ price_t * (consumption_t + charge_t - discharge_t) / 1000

        The round-trip efficiency η is split symmetrically as √η on each side:
          energy stored  = charge   * √η
          energy released = discharge / √η
        """
        n = len(prices)
        charge = cp.Variable(n, nonneg=True)
        if discharge_allowed:
            discharge = cp.Variable(n, nonneg=True)
        soc = cp.Variable(n + 1, nonneg=True)

        # Baseline consumption cost (price × energy in MWh)
        cost = cp.sum(cp.multiply(prices, consumption / 1000))
        # Charging raises cost (buy electricity to store, incur efficiency loss)
        cost += cp.sum(cp.multiply(prices, charge / 1000)) * self.inv_sqrt_eff
        if discharge_allowed:
            # Discharging reduces cost (energy recovered from battery offsets grid draw)
            cost -= cp.sum(cp.multiply(prices, discharge / 1000)) * self.inv_sqrt_eff

        # SOC dynamics: soc[t+1] = soc[t] + sqrt_eff*charge[t] - discharge[t]/sqrt_eff
        constraints = [soc[0] == current_soc, soc <= self.capacity]
        for t in range(n):
            constraints.append(charge[t] <= self.max_power * 0.25)  # 15-min power cap
            if discharge_allowed:
                constraints.append(discharge[t] <= self.max_power * 0.25)
                energy_delta = (
                    charge[t] * self.sqrt_eff - discharge[t] * self.inv_sqrt_eff
                )
            else:
                energy_delta = charge[t] * self.sqrt_eff
            constraints.append(soc[t + 1] == soc[t] + energy_delta)

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

    # ── Scenario sampling + averaging ─────────────────────────────────────────

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
        """Sample N_SCENARIOS futures, solve an LP for each, and average the schedules.

        Correlated samples: z ~ N(0,I), noise = std * L @ z, where L is the lower
        Cholesky factor of the residual correlation matrix estimated on training data.
        Without a Cholesky factor, independent Gaussian noise is used instead.
        """
        costs, all_soc, all_charge, all_discharge = [], [], [], []
        n = len(price_mean)
        for _ in range(self.n_scenarios):
            # ── Draw temporally correlated noise ─────────────────────────────
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

            # Clip sampled prices to physical bounds; extreme outliers cause ECOS
            # numerical failures (German day-ahead: historically −500 to 3000 EUR/MWh)
            prices = np.clip(price_mean + price_noise, -500.0, 3000.0)

            # # Consumption must be non-negative; cap at 5x the horizon mean as sanity bound
            # max_consumption = max(np.mean(dla_mean) * 5, 1.0)
            # consumption = np.clip(dla_mean + dla_noise, 0.0, max_consumption)
            consumption = dla_mean + dla_noise

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
