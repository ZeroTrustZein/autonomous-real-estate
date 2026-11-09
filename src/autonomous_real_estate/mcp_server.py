"""Model Context Protocol (MCP) server exposing real estate underwriting tools."""

from __future__ import annotations

import json
from typing import Any, Dict

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError:
        MCPServer = None

from autonomous_real_estate.engine import CashFlowEngine
from autonomous_real_estate.models import (
    LoanTerms,
    MonteCarloParams,
    PropertyInput,
    StressScenario,
)
from autonomous_real_estate.monte_carlo import MonteCarloSimulator
from autonomous_real_estate.swarm import UnderwritingSwarm


def create_mcp_server() -> Any:
    """Create and configure MCP server instance."""
    if MCPServer is None:
        raise RuntimeError("mcp package is not installed.")

    server = MCPServer("autonomous-real-estate")

    @server.tool()
    def project_cash_flows(
        property_id: str,
        purchase_price: float,
        gross_potential_rent: float,
        vacancy_rate: float = 0.05,
        operating_expense_ratio: float = 0.35,
        holding_period_years: int = 5,
        exit_cap_rate: float = 0.065,
        annual_rent_growth: float = 0.03,
        ltv: float = 0.75,
        interest_rate: float = 0.065,
    ) -> str:
        """Project multi-year pro-forma cash flows, DSCR, and levered/unlevered IRR."""
        prop = PropertyInput(
            property_id=property_id,
            purchase_price=purchase_price,
            gross_potential_rent=gross_potential_rent,
            vacancy_rate=vacancy_rate,
            operating_expense_ratio=operating_expense_ratio,
            holding_period_years=holding_period_years,
            exit_cap_rate=exit_cap_rate,
            annual_rent_growth=annual_rent_growth,
            loan=LoanTerms(ltv=ltv, interest_rate=interest_rate),
        )
        proj = CashFlowEngine.project_cash_flows(prop)
        return proj.model_dump_json(indent=2)

    @server.tool()
    def run_monte_carlo(
        purchase_price: float,
        gross_potential_rent: float,
        iterations: int = 500,
        vacancy_std: float = 0.02,
        rent_growth_std: float = 0.015,
        exit_cap_std: float = 0.0075,
    ) -> str:
        """Run stochastic Monte Carlo risk simulation across vacancy and rent growth distributions."""
        prop = PropertyInput(
            purchase_price=purchase_price,
            gross_potential_rent=gross_potential_rent,
        )
        params = MonteCarloParams(
            iterations=iterations,
            vacancy_std=vacancy_std,
            rent_growth_std=rent_growth_std,
            exit_cap_std=exit_cap_std,
        )
        result = MonteCarloSimulator.simulate(prop, params)
        return result.model_dump_json(indent=2)

    @server.tool()
    def evaluate_deal(
        property_id: str,
        name: str,
        purchase_price: float,
        gross_potential_rent: float,
        vacancy_rate: float = 0.05,
        operating_expense_ratio: float = 0.35,
    ) -> str:
        """Synthesize full multi-agent underwriting report (Acquisition, Risk, and Committee decision)."""
        prop = PropertyInput(
            property_id=property_id,
            name=name,
            purchase_price=purchase_price,
            gross_potential_rent=gross_potential_rent,
            vacancy_rate=vacancy_rate,
            operating_expense_ratio=operating_expense_ratio,
        )
        swarm = UnderwritingSwarm(mc_params=MonteCarloParams(iterations=300))
        report = swarm.run_underwriting(prop)
        return report.model_dump_json(indent=2)

    return server


def run_server() -> None:
    """Run MCP server over stdio."""
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    run_server()
