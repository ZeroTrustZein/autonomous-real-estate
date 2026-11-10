"""Build deterministic README assets into docs/assets/.

Every number in every SVG comes from running the real package code with a fixed seed.
Run from repo root:  python scripts/make_docs_assets.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.text import Text  # noqa: E402

from autonomous_real_estate import cli  # noqa: E402
from autonomous_real_estate.engine import CashFlowEngine  # noqa: E402
from autonomous_real_estate.models import MonteCarloParams, PropertyInput, StressScenario  # noqa: E402
from autonomous_real_estate.monte_carlo import MonteCarloSimulator  # noqa: E402
from autonomous_real_estate.swarm import UnderwritingSwarm  # noqa: E402

SEED = 42
OUT = ROOT / "docs" / "assets"
PRICE, RENT = 5_000_000.0, 600_000.0

# Slate palette
BG, PANEL, GRID, TEXT, MUTED = "#0f172a", "#1e293b", "#334155", "#e2e8f0", "#94a3b8"
BLUE, GREEN, AMBER, RED, VIOLET = "#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa"

SCENARIOS = [
    StressScenario(name="Moderate Recession", vacancy_bump=0.04, rent_decline=0.02, cap_rate_expansion_bps=50),
    StressScenario(name="Severe Stagflation", vacancy_bump=0.08, rent_decline=0.05, cap_rate_expansion_bps=125),
    StressScenario(name="Supply Shock", vacancy_bump=0.06, rent_decline=0.03, cap_rate_expansion_bps=25),
]

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "svg.hashsalt": str(SEED),
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    # Strip the date so re-runs produce byte-identical files.
    fig.savefig(OUT / name, format="svg", metadata={"Date": None})
    plt.close(fig)
    print(f"wrote docs/assets/{name}")


def pct(v: float, _pos: object = None) -> str:
    return f"{v:.0%}"


def cli_evaluate() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(["evaluate", "--price", str(int(PRICE)), "--rent", str(int(RENT))])

    console = Console(record=True, width=88, file=io.StringIO(), force_terminal=True, color_system="truecolor")
    console.print(Text("$ are-underwrite evaluate --price 5000000 --rent 600000", style="bold #38bdf8"))
    for line in buf.getvalue().splitlines():
        style = ""
        if line.startswith("=") or line.startswith("---"):
            style = "#64748b"
        elif line.startswith("DECISION:"):
            style = "bold #34d399" if "APPROVED" in line else "bold #f87171"
        elif "FAIL" in line or "IRR -" in line:
            style = "#f87171"
        elif "PASS" in line:
            style = "#34d399"
        elif line.startswith(" * "):
            style = "#cbd5e1"
        console.print(Text(line, style=style), highlight=False)
    console.save_svg(str(OUT / "cli-evaluate.svg"), title="are-underwrite evaluate")
    print("wrote docs/assets/cli-evaluate.svg")


def irr_distribution(prop: PropertyInput) -> None:
    params = MonteCarloParams(iterations=5000, seed=SEED)
    irrs, _ = MonteCarloSimulator.sample_irrs(prop, params)
    mc = MonteCarloSimulator.simulate(prop, params)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    counts, edges, _ = ax.hist(irrs, bins=60, density=True, color=BLUE, alpha=0.35, edgecolor=BLUE, linewidth=0.4)

    # Gaussian KDE (Silverman bandwidth), numpy only.
    grid = np.linspace(irrs.min(), irrs.max(), 400)
    bw = 1.06 * irrs.std() * len(irrs) ** (-1 / 5)
    kde = np.exp(-0.5 * ((grid[:, None] - irrs[None, :]) / bw) ** 2).sum(1) / (len(irrs) * bw * np.sqrt(2 * np.pi))
    ax.plot(grid, kde, color=BLUE, linewidth=2)

    tail = grid <= mc.percentile_5_irr
    ax.fill_between(grid[tail], kde[tail], color=RED, alpha=0.45, label=f"5% tail (CVaR {mc.conditional_var_95:.2%})")
    ax.axvline(mc.mean_irr, color=GREEN, linewidth=2, label=f"Mean {mc.mean_irr:.2%}")
    ax.axvline(mc.percentile_5_irr, color=RED, linewidth=2, linestyle="--", label=f"P5 {mc.percentile_5_irr:.2%}")

    ymax = max(counts.max(), kde.max()) * 1.08
    ax.annotate(
        "",
        xy=(mc.percentile_5_irr, ymax * 0.9),
        xytext=(mc.mean_irr, ymax * 0.9),
        arrowprops={"arrowstyle": "<->", "color": AMBER, "linewidth": 1.6},
    )
    ax.text(
        (mc.mean_irr + mc.percentile_5_irr) / 2, ymax * 0.93, f"VaR(95%) = {mc.value_at_risk_95:.2%}",
        color=AMBER, ha="center", va="bottom", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": PANEL, "edgecolor": "none"},
    )
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_formatter(pct)
    ax.set_yticks([])
    ax.set_xlabel("Levered IRR")
    ax.set_title(f"Monte Carlo levered IRR  ·  {mc.iterations:,} trials  ·  seed {SEED}")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "irr-distribution.svg")


def stress_scenarios(prop: PropertyInput) -> None:
    report = UnderwritingSwarm(MonteCarloParams(seed=SEED)).run_underwriting(prop, SCENARIOS)
    names = ["Base Case"] + list(report.stress_results)
    vals = [report.base_case.levered_irr] + list(report.stress_results.values())
    colors = [GREEN if v >= 0.10 else AMBER if v >= 0 else RED for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(names, vals, color=colors, width=0.6, zorder=3)
    ax.axhline(0.10, color=VIOLET, linestyle="--", linewidth=1.5, label="10% IRR hurdle", zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=4)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.01), f"{v:.2%}",
            ha="center", va="bottom" if v >= 0 else "top", color=TEXT, fontweight="bold",
        )
    lo, hi = min(vals + [0]), max(vals)
    ax.set_ylim(lo - 0.06, hi + 0.06)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("Levered IRR")
    ax.set_title("Deterministic stress scenarios  ·  RiskAgent")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "stress-scenarios.svg")


def cashflow_dscr() -> None:
    prop = PropertyInput(purchase_price=PRICE, gross_potential_rent=RENT, holding_period_years=10)
    flows = CashFlowEngine.project_cash_flows(prop).annual_flows
    years = np.array([f.year for f in flows])
    noi = np.array([f.net_operating_income for f in flows]) / 1e3
    ds = np.array([f.debt_service for f in flows]) / 1e3
    dscr = [f.dscr for f in flows]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.38
    ax.bar(years - w / 2, noi, w, color=BLUE, label="NOI", zorder=3)
    ax.bar(years + w / 2, ds, w, color=MUTED, label="Debt service", zorder=3)
    ax.set_xticks(years, [f"Y{y}" for y in years])
    ax.set_ylabel("$ thousands")
    ax.grid(axis="x", visible=False)

    ax2 = ax.twinx()
    ax2.plot(years, dscr, color=AMBER, marker="o", linewidth=2.2, label="DSCR", zorder=5)
    ax2.axhline(1.20, color=RED, linestyle="--", linewidth=1.5, label="1.20x covenant")
    for y, d in zip(years, dscr):
        ax2.text(y, d + 0.03, f"{d:.2f}x", ha="center", va="bottom", color=AMBER, fontsize=9,
                 bbox={"boxstyle": "round,pad=0.15", "facecolor": BG, "edgecolor": "none", "alpha": 0.85})
    ax2.set_ylim(0.9, max(dscr) + 0.25)
    ax2.set_ylabel("DSCR", color=AMBER)
    ax2.tick_params(axis="y", colors=AMBER)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GRID)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, ncol=4)
    ax.set_ylim(0, noi.max() * 1.3)
    ax.set_title("10-year NOI vs debt service  ·  75% LTV @ 6.5%, 30y amortization")
    save(fig, "cashflow-dscr.svg")


def main() -> int:
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    prop = PropertyInput(purchase_price=PRICE, gross_potential_rent=RENT)
    cli_evaluate()
    irr_distribution(prop)
    stress_scenarios(prop)
    cashflow_dscr()
    return 0


if __name__ == "__main__":
    sys.exit(main())
