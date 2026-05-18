import numpy as np
import cvxpy as cp

from config import (
    BATTERY_CAPACITY_KWH,
    BATTERY_MAX_POWER_KW,
    BATTERY_ROUND_TRIP_EFFICIENCY,
    HORIZON_HOURS,
    N_SCENARIOS,
    RISK_EPSILON,
    RISK_DELTA,
)


class StochasticOptimizer:
    """
    Receding-horizon stochastic battery scheduler.

    Primary public API
    ------------------
    sample_scenarios(price_mean, price_std, dla_mean, dla_std)
        Draw N_SCENARIOS (prices, consumption) tuples using the Cholesky-correlated
        Gaussian sampler.  Returns a plain list so any external source of scenarios
        (normalizing flows, ensemble models, expert draws, …) can be used instead.

    optimize(scenarios, current_soc, discharge_allowed, mode, epsilon, delta)
        Solve the battery scheduling LP given a list of (prices, consumption) tuples.
        Modes:
          "extensive"         — hard SOC constraints, shared schedule (recommended)
          "mean_scenario"     — single LP on scenario means (fastest)
          "scenario_avg"      — independent LP per scenario, schedules averaged (legacy)
          "cvar"              — CVaR-relaxed SOC constraints at level epsilon
          "scenario_approach" — uses provided scenarios as the scenario-approach set;
                                reports whether N satisfies the Campi-Garatti bound

    optimize_from_stats(price_mean, price_std, dla_mean, dla_std, ...)
        Calls sample_scenarios then optimize — preserves the original interface.
        For mode="scenario_approach" this auto-computes the required N via the
        Campi-Garatti bound and draws that many scenarios before solving.
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

    # ── Public scenario sampling ───────────────────────────────────────────────

    def sample_scenarios(self, price_mean, price_std, dla_mean, dla_std):
        """Draw self.n_scenarios (prices, consumption) pairs with correlated noise.

        Returns
        -------
        list of (prices, consumption) tuples, each a 1-D numpy array of length
        horizon_steps.  Any external scenario source can produce the same format.
        """
        return self._draw_scenarios(price_mean, price_std, dla_mean, dla_std)

    # ── Primary optimizer ─────────────────────────────────────────────────────

    def optimize(self, scenarios, current_soc, discharge_allowed=False, mode="extensive",
                 epsilon=RISK_EPSILON, delta=RISK_DELTA):
        """Solve the battery scheduling problem given a list of scenarios.

        Parameters
        ----------
        scenarios : list of (prices, consumption) tuples
            Each tuple contains two 1-D arrays of length horizon_steps.
        current_soc : float
            Battery state of charge at the start of the horizon (kWh).
        mode : str
            "extensive"         — hard SOC constraints, shared first-stage schedule.
            "mean_scenario"     — single LP on scenario means.
            "scenario_avg"      — independent LP per scenario, schedules averaged.
            "cvar"              — CVaR-relaxed SOC constraints at risk level epsilon.
            "scenario_approach" — extensive form on provided scenarios; result dict
                                  includes whether N satisfies Campi-Garatti bound.
        epsilon : float
            Acceptable SOC violation probability for "cvar" / "scenario_approach".
        delta : float
            Confidence level for Campi-Garatti N check in "scenario_approach".
        """
        if mode == "extensive":
            return self._extensive_form(scenarios, current_soc, discharge_allowed)
        elif mode == "mean_scenario":
            prices_mean = np.mean([s[0] for s in scenarios], axis=0)
            dla_mean    = np.mean([s[1] for s in scenarios], axis=0)
            return self._mean_scenario(prices_mean, dla_mean, current_soc, discharge_allowed)
        elif mode == "scenario_avg":
            return self._scenario_avg(scenarios, current_soc, discharge_allowed)
        elif mode == "cvar":
            return self._extensive_form_cvar(scenarios, current_soc, discharge_allowed, epsilon)
        elif mode == "scenario_approach":
            # Use the provided scenarios as-is; report Campi-Garatti compliance
            n_required = self._campi_garatti_n(discharge_allowed, epsilon, delta)
            result = self._extensive_form(scenarios, current_soc, discharge_allowed)
            if result is not None:
                result["campi_garatti_n_required"] = n_required
                result["campi_garatti_satisfied"]  = len(scenarios) >= n_required
                result["epsilon"] = epsilon
                result["delta"]   = delta
            return result
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose 'extensive', 'mean_scenario', "
                "'scenario_avg', 'cvar', or 'scenario_approach'."
            )

    # ── Convenience wrapper (backwards compatible) ────────────────────────────

    def optimize_from_stats(
        self, price_mean, price_std, dla_mean, dla_std,
        current_soc, discharge_allowed=False, mode="extensive",
        epsilon=RISK_EPSILON, delta=RISK_DELTA,
    ):
        """Sample scenarios from mean/std then optimize — preserves original interface.

        For mode="scenario_approach" this automatically computes the required number
        of scenarios N via the Campi-Garatti bound and draws exactly that many,
        giving a probabilistic feasibility guarantee of (1-epsilon) with confidence
        (1-delta).
        """
        if mode == "scenario_approach":
            return self._scenario_approach(
                price_mean, price_std, dla_mean, dla_std,
                current_soc, discharge_allowed, epsilon, delta,
            )
        scenarios = self.sample_scenarios(price_mean, price_std, dla_mean, dla_std)
        return self.optimize(scenarios, current_soc, discharge_allowed, mode, epsilon, delta)

    def stochastic_optimize(
        self, price_mean, price_std, dla_mean, dla_std,
        current_soc, discharge_allowed=False, verbose=False, mode="extensive",
    ):
        """Deprecated convenience wrapper — use optimize_from_stats() instead."""
        return self.optimize_from_stats(
            price_mean, price_std, dla_mean, dla_std,
            current_soc, discharge_allowed, mode,
        )

    # ── Probabilistic helpers ─────────────────────────────────────────────────

    def _campi_garatti_n(self, discharge_allowed, epsilon, delta):
        """Minimum scenarios for (1-epsilon) feasibility guarantee with confidence (1-delta).

        Campi & Garatti (2008): N >= (2/epsilon) * (ln(1/delta) + n_d)
        where n_d = number of first-stage decision variables.
        """
        n_d = 2 * self.horizon_steps if discharge_allowed else self.horizon_steps
        return int(np.ceil((2 / epsilon) * (np.log(1.0 / delta) + n_d)))

    def _scenario_approach(
        self, price_mean, price_std, dla_mean, dla_std,
        current_soc, discharge_allowed, epsilon, delta,
    ):
        """Draw Campi-Garatti N scenarios and solve the extensive-form LP.

        The resulting schedule is feasible for any unseen scenario with probability
        >= (1 - epsilon), with confidence >= (1 - delta).
        """
        n_required = self._campi_garatti_n(discharge_allowed, epsilon, delta)
        # Temporarily override n_scenarios to draw the required number
        orig_n = self.n_scenarios
        self.n_scenarios = n_required
        scenarios = self._draw_scenarios(price_mean, price_std, dla_mean, dla_std)
        self.n_scenarios = orig_n

        if self.verbose:
            print(f"  [scenario_approach] N_required={n_required} "
                  f"(epsilon={epsilon}, delta={delta})")

        result = self._extensive_form(scenarios, current_soc, discharge_allowed)
        if result is not None:
            result["campi_garatti_n_required"] = n_required
            result["campi_garatti_satisfied"]  = True
            result["epsilon"] = epsilon
            result["delta"]   = delta
        return result

    # ── Mode implementations ──────────────────────────────────────────────────

    def _draw_scenarios(self, price_mean, price_std, dla_mean, dla_std):
        """Draw self.n_scenarios (prices, consumption) pairs with correlated noise."""
        n = len(price_mean)
        scenarios = []
        for _ in range(self.n_scenarios):
            if self.price_chol is not None and self.price_chol.shape[0] == n:
                price_noise = price_std * (self.price_chol @ np.random.randn(n))
            else:
                price_noise = price_std * np.random.randn(n)

            if self.dla_chol is not None and self.dla_chol.shape[0] == n:
                dla_noise = dla_std * (self.dla_chol @ np.random.randn(n))
            else:
                dla_noise = dla_std * np.random.randn(n)

            prices      = np.clip(price_mean + price_noise, -500.0, 3000.0)
            consumption = dla_mean + dla_noise
            scenarios.append((prices, consumption))
        return scenarios

    def _mean_scenario(self, price_mean, dla_mean, current_soc, discharge_allowed):
        """Deterministic LP on expected inputs — fastest option."""
        result = self.optimize_single_scenario(
            np.clip(price_mean, -500.0, 3000.0), dla_mean, current_soc, discharge_allowed
        )
        if result is None:
            return None
        return {
            "expected_cost": result["cost"],
            "cost_std":      0.0,
            "avg_soc":       result["soc"],
            "avg_charge":    result["charge"],
            "avg_discharge": result["discharge"],
        }

    def _scenario_avg(self, scenarios, current_soc, discharge_allowed):
        """Solve one LP per scenario independently, then average schedules."""
        costs, all_soc, all_charge, all_discharge = [], [], [], []
        for prices, consumption in scenarios:
            result = self.optimize_single_scenario(
                prices, consumption, current_soc, discharge_allowed
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
            "cost_std":      np.std(costs),
            "avg_soc":       np.mean(all_soc, axis=0),
            "avg_charge":    np.mean(all_charge, axis=0),
            "avg_discharge": np.mean(all_discharge, axis=0),
        }

    def _extensive_form(self, scenarios, current_soc, discharge_allowed):
        """Extensive-form stochastic LP.

        All scenarios share a single first-stage schedule (charge/discharge for
        every time step), which is what actually gets committed.  Each scenario
        has its own SOC trajectory as a second-stage variable, so the SOC
        constraints are enforced per-scenario rather than only on average.

        Objective: minimise expected cost across all scenarios.
          min  (1/S) Σ_s Σ_t price_t^s * (consumption_t^s + charge_t - discharge_t) / 1000
        subject to:
          per-scenario SOC dynamics and capacity bounds
          shared power limits on charge / discharge
        """
        S = len(scenarios)
        n = len(scenarios[0][0])

        # First-stage (shared) decision variables
        charge = cp.Variable(n, nonneg=True)
        discharge = cp.Variable(n, nonneg=True) if discharge_allowed else None

        # Second-stage: one SOC trajectory per scenario
        soc_vars = [cp.Variable(n + 1, nonneg=True) for _ in range(S)]

        total_cost = 0
        constraints = []
        for s, ((prices, consumption), soc) in enumerate(zip(scenarios, soc_vars)):
            # SOC dynamics for scenario s
            constraints.append(soc[0] == current_soc)
            constraints.append(soc <= self.capacity)
            for t in range(n):
                if discharge_allowed:
                    energy_delta = charge[t] * self.sqrt_eff - discharge[t] * self.inv_sqrt_eff
                else:
                    energy_delta = charge[t] * self.sqrt_eff
                constraints.append(soc[t + 1] == soc[t] + energy_delta)

            # Scenario cost
            scenario_cost = cp.sum(cp.multiply(prices, consumption / 1000))
            scenario_cost += cp.sum(cp.multiply(prices, charge / 1000)) * self.inv_sqrt_eff
            if discharge_allowed:
                scenario_cost -= cp.sum(cp.multiply(prices, discharge / 1000)) * self.inv_sqrt_eff
            total_cost += scenario_cost

        # Shared power limits
        constraints += [charge <= self.max_power * 0.25]
        if discharge_allowed:
            constraints += [discharge <= self.max_power * 0.25]

        problem = cp.Problem(cp.Minimize(total_cost / S), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            return None

        charge_val    = np.array(charge.value).flatten()
        discharge_val = np.array(discharge.value).flatten() if discharge_allowed else np.zeros(n)

        # Reconstruct the shared SOC trajectory from the committed schedule
        soc_traj = np.empty(n + 1)
        soc_traj[0] = current_soc
        for t in range(n):
            delta = charge_val[t] * self.sqrt_eff - discharge_val[t] * self.inv_sqrt_eff
            soc_traj[t + 1] = np.clip(soc_traj[t] + delta, 0, self.capacity)

        # Expected cost: average per-scenario cost using committed schedule
        scenario_costs = []
        for prices, consumption in scenarios:
            c = np.sum(prices * consumption / 1000)
            c += np.sum(prices * charge_val / 1000) * self.inv_sqrt_eff
            if discharge_allowed:
                c -= np.sum(prices * discharge_val / 1000) * self.inv_sqrt_eff
            scenario_costs.append(c)

        return {
            "expected_cost": np.mean(scenario_costs),
            "cost_std":      np.std(scenario_costs),
            "avg_soc":       soc_traj,
            "avg_charge":    charge_val,
            "avg_discharge": discharge_val,
        }

    def _extensive_form_cvar(self, scenarios, current_soc, discharge_allowed, epsilon):
        """Extensive-form LP with CVaR-relaxed SOC capacity constraints.

        Instead of enforcing soc[t,s] in [0, C] for every scenario s and
        timestep t (hard constraints), we require:

            CVaR_epsilon( soc[t,s] - C ) <= 0   (upper bound in expectation)
            CVaR_epsilon( -soc[t,s]    ) <= 0   (lower bound in expectation)

        This allows up to fraction epsilon of scenarios to violate capacity in
        each timestep, but controls the expected excess in the tail.

        Linearisation via auxiliary variables eta (VaR threshold) and xi >= 0
        (exceedance):
            eta_t + (1 / epsilon*S) * sum_s xi[s,t] <= 0
            xi[s,t] >= violation[s,t] - eta_t
        """
        S = len(scenarios)
        n = len(scenarios[0][0])

        charge    = cp.Variable(n, nonneg=True)
        discharge = cp.Variable(n, nonneg=True) if discharge_allowed else None

        # SOC unrestricted (bounds enforced softly via CVaR)
        soc_vars = [cp.Variable(n + 1) for _ in range(S)]

        # CVaR auxiliary variables — one VaR threshold and exceedance per timestep
        eta_up = cp.Variable(n + 1)                       # upper bound VaR
        eta_lo = cp.Variable(n + 1)                       # lower bound VaR
        xi_up  = cp.Variable((S, n + 1), nonneg=True)     # upper exceedance
        xi_lo  = cp.Variable((S, n + 1), nonneg=True)     # lower exceedance

        total_cost  = 0
        constraints = []

        for s, ((prices, consumption), soc) in enumerate(zip(scenarios, soc_vars)):
            constraints.append(soc[0] == current_soc)
            for t in range(n):
                if discharge_allowed:
                    energy_delta = charge[t] * self.sqrt_eff - discharge[t] * self.inv_sqrt_eff
                else:
                    energy_delta = charge[t] * self.sqrt_eff
                constraints.append(soc[t + 1] == soc[t] + energy_delta)

            # CVaR linearisation — exceedance per scenario
            constraints.append(xi_up[s] >= soc - self.capacity - eta_up)
            constraints.append(xi_lo[s] >= -soc - eta_lo)

            scenario_cost  = cp.sum(cp.multiply(prices, consumption / 1000))
            scenario_cost += cp.sum(cp.multiply(prices, charge / 1000)) * self.inv_sqrt_eff
            if discharge_allowed:
                scenario_cost -= cp.sum(cp.multiply(prices, discharge / 1000)) * self.inv_sqrt_eff
            total_cost += scenario_cost

        # CVaR constraints: expected tail excess <= 0
        constraints.append(eta_up + cp.sum(xi_up, axis=0) / (epsilon * S) <= 0)
        constraints.append(eta_lo + cp.sum(xi_lo, axis=0) / (epsilon * S) <= 0)

        # Shared power limits
        constraints += [charge <= self.max_power * 0.25]
        if discharge_allowed:
            constraints += [discharge <= self.max_power * 0.25]

        problem = cp.Problem(cp.Minimize(total_cost / S), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            return None

        charge_val    = np.array(charge.value).flatten()
        discharge_val = np.array(discharge.value).flatten() if discharge_allowed else np.zeros(n)

        soc_traj    = np.empty(n + 1)
        soc_traj[0] = current_soc
        for t in range(n):
            delta = charge_val[t] * self.sqrt_eff - discharge_val[t] * self.inv_sqrt_eff
            soc_traj[t + 1] = np.clip(soc_traj[t] + delta, 0, self.capacity)

        scenario_costs = []
        for prices, consumption in scenarios:
            c  = np.sum(prices * consumption / 1000)
            c += np.sum(prices * charge_val / 1000) * self.inv_sqrt_eff
            if discharge_allowed:
                c -= np.sum(prices * discharge_val / 1000) * self.inv_sqrt_eff
            scenario_costs.append(c)

        return {
            "expected_cost": np.mean(scenario_costs),
            "cost_std":      np.std(scenario_costs),
            "avg_soc":       soc_traj,
            "avg_charge":    charge_val,
            "avg_discharge": discharge_val,
            "epsilon":       epsilon,
        }
