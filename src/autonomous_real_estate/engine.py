"""Deterministic financial engine for DCF, debt service, and IRR calculations."""

from __future__ import annotations

import math
from typing import List, Optional
import numpy as np

from autonomous_real_estate.models import (
    AnnualCashFlow,
    CashFlowProjection,
    LoanTerms,
    PropertyInput,
)


class CashFlowEngine:
    """Calculates deterministic property cash flows, mortgage amortizations, and IRRs."""

    @staticmethod
    def calculate_annual_debt_service(loan_amount: float, terms: LoanTerms) -> float:
        """Calculate annual debt payment for amortizing mortgage."""
        if loan_amount <= 0.0 or terms.ltv <= 0.0 or terms.interest_rate <= 0.0:
            return 0.0

        r = terms.monthly_rate
        n = terms.total_payments
        factor = (1.0 + r) ** n
        monthly_payment = loan_amount * (r * factor) / (factor - 1.0)
        return monthly_payment * 12.0

    @staticmethod
    def calculate_remaining_balance(
        loan_amount: float, terms: LoanTerms, elapsed_years: int
    ) -> float:
        """Compute remaining principal balance after elapsed years."""
        if loan_amount <= 0.0 or terms.interest_rate <= 0.0:
            return 0.0

        r = terms.monthly_rate
        n = terms.total_payments
        p = elapsed_years * 12
        if p >= n:
            return 0.0

        numerator = ((1.0 + r) ** n) - ((1.0 + r) ** p)
        denominator = ((1.0 + r) ** n) - 1.0
        return loan_amount * (numerator / denominator)

    @classmethod
    def calculate_irr(cls, cash_flows: List[float], max_iter: int = 100, tol: float = 1e-6) -> float:
        """Calculate Internal Rate of Return (IRR) via Newton-Raphson with bisection fallback."""
        if not cash_flows or len(cash_flows) < 2:
            return 0.0

        # Check if all cash flows have the same sign
        if all(cf >= 0 for cf in cash_flows) or all(cf <= 0 for cf in cash_flows):
            return 0.0

        # Try Newton-Raphson starting at 0.10
        rate = 0.10
        for _ in range(max_iter):
            if rate <= -0.999:
                rate = -0.999
            npv = 0.0
            d_npv = 0.0
            for t, cf in enumerate(cash_flows):
                disc = (1.0 + rate) ** t
                npv += cf / disc
                if t > 0:
                    d_npv -= t * cf / ((1.0 + rate) ** (t + 1))

            if abs(npv) < tol:
                return float(rate)

            if abs(d_npv) < 1e-12:
                break

            step = npv / d_npv
            rate -= step
            if rate < -0.99 or rate > 10.0:
                break

        # Fallback: Bisection search between -0.90 and 5.0
        low = -0.90
        high = 5.0
        npv_low = sum(cf / ((1.0 + low) ** t) for t, cf in enumerate(cash_flows))
        npv_high = sum(cf / ((1.0 + high) ** t) for t, cf in enumerate(cash_flows))

        if npv_low * npv_high > 0:
            return float(np.nan_to_num(rate, nan=0.0))

        for _ in range(100):
            mid = (low + high) / 2.0
            npv_mid = sum(cf / ((1.0 + mid) ** t) for t, cf in enumerate(cash_flows))
            if abs(npv_mid) < tol or (high - low) / 2.0 < tol:
                return float(mid)
            if (npv_low * npv_mid) <= 0:
                high = mid
                npv_high = npv_mid
            else:
                low = mid
                npv_low = npv_mid

        return float((low + high) / 2.0)

    @classmethod
    def project_cash_flows(
        cls,
        prop: PropertyInput,
        vacancy_override: Optional[float] = None,
        rent_growth_override: Optional[float] = None,
        exit_cap_override: Optional[float] = None,
    ) -> CashFlowProjection:
        """Generate multi-year cash flow waterfall and returns."""
        loan_terms = prop.loan or LoanTerms(ltv=0.0)
        loan_amount = prop.purchase_price * loan_terms.ltv
        initial_equity = prop.purchase_price - loan_amount

        annual_debt_service = cls.calculate_annual_debt_service(loan_amount, loan_terms)

        vacancy_rate = vacancy_override if vacancy_override is not None else prop.vacancy_rate
        rent_growth = rent_growth_override if rent_growth_override is not None else prop.annual_rent_growth
        exit_cap = exit_cap_override if exit_cap_override is not None else prop.exit_cap_rate

        annual_flows: List[AnnualCashFlow] = []
        current_gpr = prop.gross_potential_rent
        current_opex_rate = prop.operating_expense_ratio

        dscrs: List[float] = []

        for yr in range(1, prop.holding_period_years + 1):
            if yr > 1:
                current_gpr *= (1.0 + rent_growth)

            gross_rev = current_gpr + prop.other_income
            vac_loss = gross_rev * vacancy_rate
            egi = gross_rev - vac_loss
            opex = egi * current_opex_rate
            noi = egi - opex
            cash_flow_after_debt = noi - annual_debt_service

            dscr = noi / annual_debt_service if annual_debt_service > 0 else 999.0
            dscrs.append(dscr)

            annual_flows.append(
                AnnualCashFlow(
                    year=yr,
                    gross_potential_revenue=round(gross_rev, 2),
                    vacancy_loss=round(vac_loss, 2),
                    effective_gross_income=round(egi, 2),
                    operating_expenses=round(opex, 2),
                    net_operating_income=round(noi, 2),
                    debt_service=round(annual_debt_service, 2),
                    cash_flow_after_debt=round(cash_flow_after_debt, 2),
                    dscr=round(dscr, 2),
                )
            )

        # Terminal Year calculations
        terminal_forward_noi = annual_flows[-1].net_operating_income * (1.0 + rent_growth)
        exit_sale_price = terminal_forward_noi / exit_cap if exit_cap > 0 else 0.0
        exit_debt_balance = cls.calculate_remaining_balance(
            loan_amount, loan_terms, prop.holding_period_years
        )
        net_terminal_proceeds = max(0.0, exit_sale_price - exit_debt_balance)

        # Levered cash flows: -Equity, CF1, CF2, ..., CFn + Terminal Equity
        levered_flows = [-initial_equity] + [f.cash_flow_after_debt for f in annual_flows]
        levered_flows[-1] += net_terminal_proceeds

        # Unlevered cash flows: -PurchasePrice, NOI1, NOI2, ..., NOIn + ExitPrice
        unlevered_flows = [-prop.purchase_price] + [f.net_operating_income for f in annual_flows]
        unlevered_flows[-1] += exit_sale_price

        levered_irr = cls.calculate_irr(levered_flows)
        unlevered_irr = cls.calculate_irr(unlevered_flows)

        total_distributions = sum(levered_flows[1:])
        equity_multiple = total_distributions / initial_equity if initial_equity > 0 else 0.0

        return CashFlowProjection(
            annual_flows=annual_flows,
            initial_equity=round(initial_equity, 2),
            loan_amount=round(loan_amount, 2),
            exit_sale_price=round(exit_sale_price, 2),
            exit_debt_balance=round(exit_debt_balance, 2),
            net_terminal_proceeds=round(net_terminal_proceeds, 2),
            unlevered_irr=round(unlevered_irr, 4),
            levered_irr=round(levered_irr, 4),
            equity_multiple=round(equity_multiple, 3),
            avg_dscr=round(float(np.mean(dscrs)), 2) if dscrs else 0.0,
            min_dscr=round(float(np.min(dscrs)), 2) if dscrs else 0.0,
        )
