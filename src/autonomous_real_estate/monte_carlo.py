"""Stochastic risk simulation engine for real estate cash flows and valuations."""

from __future__ import annotations

import numpy as np
from autonomous_real_estate.engine import CashFlowEngine
from autonomous_real_estate.models import (
    MonteCarloParams,
    MonteCarloResult,
    PropertyInput,
)


class MonteCarloSimulator:
    """Executes stochastic distributions over underwriting metrics."""

    @classmethod
    def sample_irrs(
        cls,
        prop: PropertyInput,
        params: MonteCarloParams | None = None,
    ) -> tuple[np.ndarray, int]:
        """Sample Monte Carlo IRR distribution and count DSCR breaches (< 1.20)."""
        p = params or MonteCarloParams()
        rng = np.random.default_rng(p.seed)

        # Sample distributions
        vacancies = np.clip(
            rng.normal(loc=prop.vacancy_rate, scale=p.vacancy_std, size=p.iterations),
            0.0,
            0.95,
        )
        rent_growths = rng.normal(
            loc=prop.annual_rent_growth, scale=p.rent_growth_std, size=p.iterations
        )
        exit_caps = np.clip(
            rng.normal(loc=prop.exit_cap_rate, scale=p.exit_cap_std, size=p.iterations),
            0.02,
            0.25,
        )

        irrs: list[float] = []
        dscr_breaches = 0

        for i in range(p.iterations):
            proj = CashFlowEngine.project_cash_flows(
                prop=prop,
                vacancy_override=float(vacancies[i]),
                rent_growth_override=float(rent_growths[i]),
                exit_cap_override=float(exit_caps[i]),
            )
            irrs.append(proj.levered_irr)
            if proj.min_dscr < 1.20:
                dscr_breaches += 1

        return np.array(irrs), dscr_breaches

    @classmethod
    def simulate(
        cls,
        prop: PropertyInput,
        params: MonteCarloParams | None = None,
    ) -> MonteCarloResult:
        """Run N Monte Carlo iterations by shocking vacancy, rent growth, and exit caps."""
        p = params or MonteCarloParams()
        irr_array, dscr_breaches = cls.sample_irrs(prop, p)

        mean_irr = float(np.mean(irr_array))
        median_irr = float(np.median(irr_array))
        std_irr = float(np.std(irr_array))

        p5_irr = float(np.percentile(irr_array, 5))
        p95_irr = float(np.percentile(irr_array, 95))

        # VaR (95% confidence): loss relative to mean IRR
        var_95 = max(0.0, mean_irr - p5_irr)
        cvar_tail = irr_array[irr_array <= p5_irr]
        cvar_95 = max(0.0, mean_irr - float(np.mean(cvar_tail))) if len(cvar_tail) > 0 else var_95

        prob_neg = float(np.mean(irr_array < 0.0))
        prob_dscr = float(dscr_breaches / p.iterations)

        return MonteCarloResult(
            iterations=p.iterations,
            mean_irr=round(mean_irr, 4),
            median_irr=round(median_irr, 4),
            std_irr=round(std_irr, 4),
            percentile_5_irr=round(p5_irr, 4),
            percentile_95_irr=round(p95_irr, 4),
            value_at_risk_95=round(var_95, 4),
            conditional_var_95=round(cvar_95, 4),
            probability_negative_irr=round(prob_neg, 4),
            probability_dscr_breach=round(prob_dscr, 4),
        )
