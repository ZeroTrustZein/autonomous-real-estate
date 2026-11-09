"""Pydantic schemas and domain models for real estate underwriting."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class LoanTerms(BaseModel):
    """Loan parameters for debt financing."""
    ltv: float = Field(default=0.75, ge=0.0, le=0.95, description="Loan to Value ratio")
    interest_rate: float = Field(default=0.065, ge=0.0, le=0.25, description="Annual nominal interest rate")
    amortization_years: int = Field(default=30, ge=1, le=50, description="Amortization period in years")

    @property
    def monthly_rate(self) -> float:
        return self.interest_rate / 12.0

    @property
    def total_payments(self) -> int:
        return self.amortization_years * 12


class PropertyInput(BaseModel):
    """Property specifications and baseline operational metrics."""
    property_id: str = Field(default="PROP-001", description="Identifier for property")
    name: str = Field(default="Commercial Asset", description="Asset descriptor")
    purchase_price: float = Field(gt=0.0, description="Acquisition purchase price")
    gross_potential_rent: float = Field(gt=0.0, description="Year 1 Gross Potential Rent (GPR)")
    vacancy_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Expected vacancy allowance")
    operating_expense_ratio: float = Field(default=0.35, ge=0.0, le=0.90, description="Operating expenses as % of EGI")
    other_income: float = Field(default=0.0, ge=0.0, description="Ancillary annual revenue")
    holding_period_years: int = Field(default=5, ge=1, le=30, description="Investment hold horizon")
    exit_cap_rate: float = Field(default=0.065, gt=0.0, le=0.30, description="Terminal exit cap rate")
    annual_rent_growth: float = Field(default=0.03, ge=-0.20, le=0.30, description="Annual rent growth rate")
    annual_expense_growth: float = Field(default=0.025, ge=-0.10, le=0.25, description="Annual expense inflation")
    loan: Optional[LoanTerms] = Field(default_factory=LoanTerms, description="Debt terms if levered")


class AnnualCashFlow(BaseModel):
    """Single period cash flow breakdown."""
    year: int
    gross_potential_revenue: float
    vacancy_loss: float
    effective_gross_income: float
    operating_expenses: float
    net_operating_income: float
    debt_service: float
    cash_flow_after_debt: float
    dscr: float


class CashFlowProjection(BaseModel):
    """Full multi-year underwriting cash flow waterfall."""
    annual_flows: List[AnnualCashFlow]
    initial_equity: float
    loan_amount: float
    exit_sale_price: float
    exit_debt_balance: float
    net_terminal_proceeds: float
    unlevered_irr: float
    levered_irr: float
    equity_multiple: float
    avg_dscr: float
    min_dscr: float


class MonteCarloParams(BaseModel):
    """Configuration for stochastic risk simulation."""
    iterations: int = Field(default=1000, ge=100, le=50000)
    seed: Optional[int] = Field(default=42)
    vacancy_std: float = Field(default=0.02, ge=0.0)
    rent_growth_std: float = Field(default=0.015, ge=0.0)
    exit_cap_std: float = Field(default=0.0075, ge=0.0)
    interest_rate_shock_bps: float = Field(default=100.0, ge=0.0)


class MonteCarloResult(BaseModel):
    """Aggregated risk distributions from Monte Carlo trials."""
    iterations: int
    mean_irr: float
    median_irr: float
    std_irr: float
    percentile_5_irr: float
    percentile_95_irr: float
    value_at_risk_95: float
    conditional_var_95: float
    probability_negative_irr: float
    probability_dscr_breach: float


class StressScenario(BaseModel):
    """Deterministic macroeconomic stress shocks."""
    name: str
    vacancy_bump: float = 0.0
    rent_decline: float = 0.0
    cap_rate_expansion_bps: float = 0.0


class UnderwritingReport(BaseModel):
    """Comprehensive deal synthesis produced by the swarm."""
    property_id: str
    property_name: str
    decision: str  # "APPROVED", "REJECTED", "CAUTION"
    executive_summary: str
    base_case: CashFlowProjection
    monte_carlo: MonteCarloResult
    stress_results: dict[str, float]
    agent_findings: List[str]
