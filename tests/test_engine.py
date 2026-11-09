"""Unit tests for financial calculations, DCF engine, and IRR solver."""

import pytest
from autonomous_real_estate.engine import CashFlowEngine
from autonomous_real_estate.models import LoanTerms, PropertyInput


def test_annual_debt_service_calculation():
    loan_amount = 1_000_000.0
    terms = LoanTerms(ltv=0.75, interest_rate=0.06, amortization_years=30)
    payment = CashFlowEngine.calculate_annual_debt_service(loan_amount, terms)
    # Monthly payment for 1M at 6% 30yr is approx 5995.51 -> annual ~ 71946.06
    assert 71000.0 < payment < 73000.0


def test_zero_loan_debt_service():
    payment = CashFlowEngine.calculate_annual_debt_service(0.0, LoanTerms(ltv=0.0))
    assert payment == 0.0


def test_remaining_balance_amortization():
    loan_amount = 1_000_000.0
    terms = LoanTerms(interest_rate=0.06, amortization_years=30)
    # Year 0 balance is 1M
    bal_0 = CashFlowEngine.calculate_remaining_balance(loan_amount, terms, 0)
    assert pytest.approx(bal_0, rel=1e-3) == loan_amount

    # Year 5 balance
    bal_5 = CashFlowEngine.calculate_remaining_balance(loan_amount, terms, 5)
    assert 900_000.0 < bal_5 < 960_000.0

    # Year 30 balance is 0
    bal_30 = CashFlowEngine.calculate_remaining_balance(loan_amount, terms, 30)
    assert bal_30 == 0.0


def test_irr_solver_accuracy():
    # Simple investment: -100, then 110 in year 1 -> IRR = 10%
    flows = [-100.0, 110.0]
    irr = CashFlowEngine.calculate_irr(flows)
    assert pytest.approx(irr, abs=1e-4) == 0.10

    # Multiple periods: -100, 20, 20, 120 -> approx 20%
    flows_multi = [-100.0, 20.0, 20.0, 120.0]
    irr_multi = CashFlowEngine.calculate_irr(flows_multi)
    assert pytest.approx(irr_multi, abs=1e-3) == 0.20


def test_irr_all_positive_or_negative_returns_zero():
    assert CashFlowEngine.calculate_irr([10.0, 20.0, 30.0]) == 0.0
    assert CashFlowEngine.calculate_irr([-10.0, -20.0]) == 0.0


def test_project_cash_flows_end_to_end():
    prop = PropertyInput(
        purchase_price=1_000_000.0,
        gross_potential_rent=120_000.0,
        vacancy_rate=0.05,
        operating_expense_ratio=0.35,
        holding_period_years=5,
        exit_cap_rate=0.065,
        annual_rent_growth=0.03,
        loan=LoanTerms(ltv=0.70, interest_rate=0.06, amortization_years=30),
    )

    proj = CashFlowEngine.project_cash_flows(prop)
    assert len(proj.annual_flows) == 5
    assert proj.initial_equity == 300_000.0
    assert proj.loan_amount == 700_000.0
    assert proj.exit_sale_price > 1_000_000.0
    assert proj.net_terminal_proceeds > 0.0
    assert proj.levered_irr > 0.0
    assert proj.unlevered_irr > 0.0
    assert proj.equity_multiple > 1.0
    assert proj.min_dscr > 1.0
