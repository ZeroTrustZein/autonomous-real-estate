"""Command-line interface (CLI) for Autonomous Real Estate Underwriting Swarm."""

from __future__ import annotations

import argparse
import json
import sys

from autonomous_real_estate.engine import CashFlowEngine
from autonomous_real_estate.models import LoanTerms, MonteCarloParams, PropertyInput
from autonomous_real_estate.monte_carlo import MonteCarloSimulator
from autonomous_real_estate.swarm import UnderwritingSwarm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="are-underwrite",
        description="Autonomous Real Estate Underwriting Swarm & Risk Simulation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run multi-agent deal evaluation")
    eval_parser.add_argument("--id", default="PROP-101", help="Property ID")
    eval_parser.add_argument("--name", default="Commercial Asset", help="Property name")
    eval_parser.add_argument("--price", type=float, required=True, help="Purchase price ($)")
    eval_parser.add_argument("--rent", type=float, required=True, help="Gross potential rent ($)")
    eval_parser.add_argument("--vacancy", type=float, default=0.05, help="Vacancy rate (default: 0.05)")
    eval_parser.add_argument("--opex", type=float, default=0.35, help="Operating expense ratio (default: 0.35)")
    eval_parser.add_argument("--ltv", type=float, default=0.75, help="Loan to value (default: 0.75)")
    eval_parser.add_argument("--rate", type=float, default=0.065, help="Interest rate (default: 0.065)")
    eval_parser.add_argument("--json", action="store_true", help="Output JSON format")

    # Command: simulate
    sim_parser = subparsers.add_parser("simulate", help="Run Monte Carlo stress simulation")
    sim_parser.add_argument("--price", type=float, required=True, help="Purchase price ($)")
    sim_parser.add_argument("--rent", type=float, required=True, help="Gross potential rent ($)")
    sim_parser.add_argument("--iterations", type=int, default=1000, help="Number of simulations")
    sim_parser.add_argument("--json", action="store_true", help="Output JSON format")

    # Command: mcp
    subparsers.add_parser("mcp", help="Launch Model Context Protocol (MCP) server")

    return parser


def handle_evaluate(args: argparse.Namespace) -> int:
    prop = PropertyInput(
        property_id=args.id,
        name=args.name,
        purchase_price=args.price,
        gross_potential_rent=args.rent,
        vacancy_rate=args.vacancy,
        operating_expense_ratio=args.opex,
        loan=LoanTerms(ltv=args.ltv, interest_rate=args.rate),
    )
    swarm = UnderwritingSwarm()
    report = swarm.run_underwriting(prop)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print("==================================================")
        print(f" UNDERWRITING REPORT: {report.property_name} ({report.property_id})")
        print("==================================================")
        print(f"DECISION: {report.decision}")
        print(f"SUMMARY:  {report.executive_summary}\n")
        print("--- BASE CASE METRICS ---")
        print(f"Purchase Price:       ${prop.purchase_price:,.2f}")
        print(f"Initial Equity:       ${report.base_case.initial_equity:,.2f}")
        print(f"Exit Sale Price:      ${report.base_case.exit_sale_price:,.2f}")
        print(f"Unlevered IRR:        {report.base_case.unlevered_irr:.2%}")
        print(f"Levered IRR:          {report.base_case.levered_irr:.2%}")
        print(f"Equity Multiple:      {report.base_case.equity_multiple:.2f}x")
        print(f"Min DSCR:             {report.base_case.min_dscr:.2f}\n")
        print("--- MONTE CARLO RISK (95% CI) ---")
        print(f"Mean Levered IRR:     {report.monte_carlo.mean_irr:.2%}")
        print(f"Value at Risk (95%):  {report.monte_carlo.value_at_risk_95:.2%}")
        print(f"Probability Neg IRR:  {report.monte_carlo.probability_negative_irr:.1%}")
        print(f"Probability DSCR <1.2:{report.monte_carlo.probability_dscr_breach:.1%}\n")
        print("--- AGENT SYNTHESIS LOGS ---")
        for log in report.agent_findings:
            print(f" * {log}")
        print("==================================================")

    return 0


def handle_simulate(args: argparse.Namespace) -> int:
    prop = PropertyInput(
        purchase_price=args.price,
        gross_potential_rent=args.rent,
    )
    result = MonteCarloSimulator.simulate(prop, MonteCarloParams(iterations=args.iterations))
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"Monte Carlo ({result.iterations} trials):")
        print(f"  Mean IRR:     {result.mean_irr:.2%}")
        print(f"  Median IRR:   {result.median_irr:.2%}")
        print(f"  VaR 95%:      {result.value_at_risk_95:.2%}")
        print(f"  CVaR 95%:     {result.conditional_var_95:.2%}")
        print(f"  Neg IRR Prob: {result.probability_negative_irr:.1%}")
    return 0


def handle_mcp() -> int:
    from autonomous_real_estate.mcp_server import run_server
    run_server()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "evaluate":
        return handle_evaluate(args)
    elif args.command == "simulate":
        return handle_simulate(args)
    elif args.command == "mcp":
        return handle_mcp()

    return 0


if __name__ == "__main__":
    sys.exit(main())
