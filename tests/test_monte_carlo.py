"""Unit tests for Monte Carlo stochastic risk simulation."""

from autonomous_real_estate.models import MonteCarloParams, PropertyInput
from autonomous_real_estate.monte_carlo import MonteCarloSimulator


def test_monte_carlo_distribution_properties():
    prop = PropertyInput(
        purchase_price=1_500_000.0,
        gross_potential_rent=180_000.0,
        vacancy_rate=0.06,
        operating_expense_ratio=0.38,
    )
    params = MonteCarloParams(iterations=200, seed=42)
    res = MonteCarloSimulator.simulate(prop, params)

    assert res.iterations == 200
    assert res.mean_irr > -0.5
    assert res.percentile_5_irr <= res.median_irr <= res.percentile_95_irr
    assert res.value_at_risk_95 >= 0.0
    assert res.conditional_var_95 >= res.value_at_risk_95
    assert 0.0 <= res.probability_negative_irr <= 1.0
    assert 0.0 <= res.probability_dscr_breach <= 1.0


def test_monte_carlo_reproducibility():
    prop = PropertyInput(
        purchase_price=1_000_000.0,
        gross_potential_rent=120_000.0,
    )
    params1 = MonteCarloParams(iterations=100, seed=99)
    res1 = MonteCarloSimulator.simulate(prop, params1)

    params2 = MonteCarloParams(iterations=100, seed=99)
    res2 = MonteCarloSimulator.simulate(prop, params2)

    assert res1.mean_irr == res2.mean_irr
    assert res1.value_at_risk_95 == res2.value_at_risk_95
