"""Unit tests for multi-agent underwriting swarm collaboration."""

from autonomous_real_estate.models import (
    MonteCarloParams,
    PropertyInput,
    StressScenario,
)
from autonomous_real_estate.swarm import AcquisitionAgent, RiskAgent, UnderwritingSwarm


def test_acquisition_agent_evaluation():
    agent = AcquisitionAgent()
    prop = PropertyInput(
        purchase_price=1_000_000.0,
        gross_potential_rent=130_000.0,
        vacancy_rate=0.05,
        operating_expense_ratio=0.35,
    )
    from autonomous_real_estate.engine import CashFlowEngine
    proj = CashFlowEngine.project_cash_flows(prop)

    approved, logs = agent.evaluate(prop, proj)
    assert isinstance(approved, bool)
    assert len(logs) >= 2


def test_swarm_underwriting_approval_and_caution():
    # Strong deal -> APPROVED
    strong_prop = PropertyInput(
        property_id="DEAL-STRONG",
        name="High Yield Multifamily",
        purchase_price=1_000_000.0,
        gross_potential_rent=160_000.0,
        vacancy_rate=0.04,
        operating_expense_ratio=0.30,
        annual_rent_growth=0.04,
    )
    swarm = UnderwritingSwarm(mc_params=MonteCarloParams(iterations=100, seed=42))
    report = swarm.run_underwriting(strong_prop)

    assert report.property_id == "DEAL-STRONG"
    assert report.decision in ["APPROVED", "CAUTION"]
    assert len(report.agent_findings) > 0
    assert "Mild Recession" in report.stress_results

    # Unprofitable deal -> REJECTED
    bad_prop = PropertyInput(
        property_id="DEAL-BAD",
        name="Overpriced Low Yield",
        purchase_price=3_000_000.0,
        gross_potential_rent=60_000.0,
        vacancy_rate=0.15,
        operating_expense_ratio=0.55,
    )
    report_bad = swarm.run_underwriting(bad_prop)
    assert report_bad.decision == "REJECTED"
