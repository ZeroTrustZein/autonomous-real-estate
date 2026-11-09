"""Multi-agent underwriting swarm coordinating deal evaluation."""

from __future__ import annotations

from typing import List, Tuple
from autonomous_real_estate.engine import CashFlowEngine
from autonomous_real_estate.models import (
    CashFlowProjection,
    MonteCarloParams,
    MonteCarloResult,
    PropertyInput,
    StressScenario,
    UnderwritingReport,
)
from autonomous_real_estate.monte_carlo import MonteCarloSimulator


class AcquisitionAgent:
    """Agent evaluating deal baseline metrics and returns hurdle."""

    def evaluate(self, prop: PropertyInput, proj: CashFlowProjection) -> Tuple[bool, List[str]]:
        findings = []
        approved = True

        entry_cap = (
            (prop.gross_potential_rent * (1.0 - prop.vacancy_rate) * (1.0 - prop.operating_expense_ratio))
            / prop.purchase_price
        )
        findings.append(f"[AcquisitionAgent] Unlevered Day-1 Cap Rate: {entry_cap:.2%}")

        if proj.levered_irr < 0.10:
            approved = False
            findings.append(f"[AcquisitionAgent] FAIL: Levered IRR {proj.levered_irr:.2%} below hurdle (10.0%)")
        else:
            findings.append(f"[AcquisitionAgent] PASS: Levered IRR {proj.levered_irr:.2%} meets hurdle")

        if proj.equity_multiple < 1.30:
            approved = False
            findings.append(f"[AcquisitionAgent] FAIL: Equity Multiple {proj.equity_multiple:.2f}x below 1.30x")

        return approved, findings


class RiskAgent:
    """Agent running stochastic simulation and macro stress tests."""

    def evaluate(
        self,
        prop: PropertyInput,
        mc: MonteCarloResult,
        stress_scenarios: List[StressScenario],
    ) -> Tuple[bool, dict[str, float], List[str]]:
        findings = []
        approved = True

        findings.append(f"[RiskAgent] Monte Carlo 95% VaR: {mc.value_at_risk_95:.2%}, CVaR: {mc.conditional_var_95:.2%}")
        if mc.probability_dscr_breach > 0.20:
            approved = False
            findings.append(f"[RiskAgent] FAIL: DSCR breach probability {mc.probability_dscr_breach:.1%} exceeds 20%")

        if mc.probability_negative_irr > 0.10:
            approved = False
            findings.append(f"[RiskAgent] FAIL: Negative IRR probability {mc.probability_negative_irr:.1%} exceeds 10%")

        stress_results: dict[str, float] = {}
        for scenario in stress_scenarios:
            stressed_proj = CashFlowEngine.project_cash_flows(
                prop=prop,
                vacancy_override=prop.vacancy_rate + scenario.vacancy_bump,
                rent_growth_override=prop.annual_rent_growth - scenario.rent_decline,
                exit_cap_override=prop.exit_cap_rate + (scenario.cap_rate_expansion_bps / 10000.0),
            )
            stress_results[scenario.name] = stressed_proj.levered_irr
            findings.append(
                f"[RiskAgent] Scenario '{scenario.name}': Levered IRR {stressed_proj.levered_irr:.2%}, Min DSCR {stressed_proj.min_dscr:.2f}"
            )

        return approved, stress_results, findings


class UnderwritingSwarm:
    """Orchestrator synthesizing multi-agent findings into institutional investment memo."""

    def __init__(self, mc_params: MonteCarloParams | None = None) -> None:
        self.mc_params = mc_params or MonteCarloParams()
        self.acquisition_agent = AcquisitionAgent()
        self.risk_agent = RiskAgent()

    def run_underwriting(
        self,
        prop: PropertyInput,
        stress_scenarios: List[StressScenario] | None = None,
    ) -> UnderwritingReport:
        """Run complete collaborative underwriting workflow."""
        scenarios = stress_scenarios or [
            StressScenario(name="Mild Recession", vacancy_bump=0.04, rent_decline=0.02, cap_rate_expansion_bps=50),
            StressScenario(name="Severe Stagflation", vacancy_bump=0.08, rent_decline=0.05, cap_rate_expansion_bps=125),
        ]

        # 1. Base case DCF
        base_proj = CashFlowEngine.project_cash_flows(prop)

        # 2. Agent 1: Acquisition evaluation
        acq_ok, acq_logs = self.acquisition_agent.evaluate(prop, base_proj)

        # 3. Agent 2: Monte Carlo Risk evaluation
        mc_result = MonteCarloSimulator.simulate(prop, self.mc_params)
        risk_ok, stress_dict, risk_logs = self.risk_agent.evaluate(prop, mc_result, scenarios)

        # 4. Committee Decision Synthesis
        all_logs = acq_logs + risk_logs
        if acq_ok and risk_ok:
            decision = "APPROVED"
            summary = "Deal satisfies baseline yield hurdles and demonstrates strong resilience against stochastic shocks."
        elif acq_ok and not risk_ok:
            decision = "CAUTION"
            summary = "Deal meets baseline returns but exhibits elevated tail-risk or debt-service vulnerability under stress."
        else:
            decision = "REJECTED"
            summary = "Deal fails primary return thresholds or has unacceptable probability of capital loss."

        return UnderwritingReport(
            property_id=prop.property_id,
            property_name=prop.name,
            decision=decision,
            executive_summary=summary,
            base_case=base_proj,
            monte_carlo=mc_result,
            stress_results=stress_dict,
            agent_findings=all_logs,
        )
