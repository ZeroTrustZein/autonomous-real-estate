"""Unit tests for underwriting domain models and validation."""

import pytest
from pydantic import ValidationError

from autonomous_real_estate.models import (
    AnnualCashFlow,
    CashFlowProjection,
    LoanTerms,
    MonteCarloParams,
    PropertyInput,
    StressScenario,
)


def test_loan_terms_defaults_and_properties():
    loan = LoanTerms()
    assert loan.ltv == 0.75
    assert loan.interest_rate == 0.065
    assert loan.amortization_years == 30
    assert pytest.approx(loan.monthly_rate, rel=1e-5) == 0.065 / 12.0
    assert loan.total_payments == 360


def test_loan_terms_validation_bounds():
    with pytest.raises(ValidationError):
        LoanTerms(ltv=1.2)  # Exceeds max LTV 0.95

    with pytest.raises(ValidationError):
        LoanTerms(interest_rate=-0.01)


def test_property_input_validation():
    prop = PropertyInput(
        purchase_price=2_000_000.0,
        gross_potential_rent=240_000.0,
    )
    assert prop.property_id == "PROP-001"
    assert prop.vacancy_rate == 0.05
    assert prop.holding_period_years == 5

    with pytest.raises(ValidationError):
        PropertyInput(purchase_price=-100.0, gross_potential_rent=50_000.0)


def test_stress_scenario_defaults():
    scenario = StressScenario(name="Downturn", vacancy_bump=0.05, rent_decline=0.03)
    assert scenario.name == "Downturn"
    assert scenario.vacancy_bump == 0.05
    assert scenario.rent_decline == 0.03
    assert scenario.cap_rate_expansion_bps == 0.0


def test_monte_carlo_params_validation():
    params = MonteCarloParams(iterations=500, seed=123)
    assert params.iterations == 500
    assert params.seed == 123

    with pytest.raises(ValidationError):
        MonteCarloParams(iterations=10)  # ge=100
