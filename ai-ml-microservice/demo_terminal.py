"""


Run:
    python demo_terminal.py
    python demo_terminal.py --scenario all     (default)
    python demo_terminal.py --scenario single
    python demo_terminal.py --scenario batch
    python demo_terminal.py --scenario db

Requirements:
    - Train the model first: python train_clv_model.py
    - The 'rich' package must be installed (pip install rich)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# ── Ensure project root is on sys.path ─────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from app.clv_schemas import CLVInputPayload, IndustryType
from pipelines.clv_predictor import CLVPredictor

console = Console()

# ── Demo client scenarios (diverse industries, engagement levels, tenures) ──
DEMO_CLIENTS: list[dict] = [
    {
        "label":                        "TechCorp India (High-Value)",
        "company_id":                   "client-tech-001",
        "historical_spend_amount":      22000.0,
        "purchase_frequency_yearly":    8,
        "industry_type":                IndustryType.TECHNOLOGY,
        "rolling_engagement_score":     88.5,
        "contract_tenure_months":       28,
        "contract_start_date":          "2024-06-15",
    },
    {
        "label":                        "FinEdge Pvt Ltd (Finance, Mid-Tier)",
        "company_id":                   "lead-fin-022",
        "historical_spend_amount":      35000.0,
        "purchase_frequency_yearly":    4,
        "industry_type":                IndustryType.FINANCE,
        "rolling_engagement_score":     63.0,
        "contract_tenure_months":       36,
        "contract_start_date":          "2023-10-01",
    },
    {
        "label":                        "MediCare Solutions (Healthcare, New)",
        "company_id":                   "client-health-004",
        "historical_spend_amount":      18500.0,
        "purchase_frequency_yearly":    3,
        "industry_type":                IndustryType.HEALTHCARE,
        "rolling_engagement_score":     71.0,
        "contract_tenure_months":       5,
        "contract_start_date":          "2026-05-01",
    },
    {
        "label":                        "QuickMart Retail (High Frequency)",
        "company_id":                   "lead-retail-009",
        "historical_spend_amount":      14000.0,
        "purchase_frequency_yearly":    16,
        "industry_type":                IndustryType.RETAIL,
        "rolling_engagement_score":     79.5,
        "contract_tenure_months":       18,
        "contract_start_date":          "2025-04-10",
    },
    {
        "label":                        "Bharat Logistics (Low Engagement)",
        "company_id":                   "client-logi-017",
        "historical_spend_amount":      26000.0,
        "purchase_frequency_yearly":    5,
        "industry_type":                IndustryType.LOGISTICS,
        "rolling_engagement_score":     31.0,
        "contract_tenure_months":       22,
        "contract_start_date":          "2024-12-01",
    },
    {
        "label":                        "EduTech Ventures (Education)",
        "company_id":                   "lead-edu-033",
        "historical_spend_amount":      9500.0,
        "purchase_frequency_yearly":    2,
        "industry_type":                IndustryType.EDUCATION,
        "rolling_engagement_score":     67.5,
        "contract_tenure_months":       14,
        "contract_start_date":          "2025-08-01",
    },
]


def _load_predictor() -> CLVPredictor:
    """Load predictor with a progress spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Loading CLV model artefacts...[/bold cyan]"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("load", total=None)
        try:
            predictor = CLVPredictor()
        except FileNotFoundError:
            console.print(
                "\n[bold red]ERROR:[/bold red] Model artefacts not found.\n"
                "Run [bold yellow]python train_clv_model.py[/bold yellow] first.\n"
            )
            sys.exit(1)
    return predictor


def _build_input(client: dict) -> CLVInputPayload:
    """Build CLVInputPayload from a demo client dict."""
    return CLVInputPayload(
        company_id=client["company_id"],
        historical_spend_amount=client["historical_spend_amount"],
        purchase_frequency_yearly=client["purchase_frequency_yearly"],
        industry_type=client["industry_type"],
        rolling_engagement_score=client["rolling_engagement_score"],
        contract_tenure_months=client["contract_tenure_months"],
        contract_start_date=client.get("contract_start_date"),
    )


def _render_prediction(label: str, payload: CLVInputPayload, result, idx: int) -> None:
    """Render a single prediction result as a rich panel."""

    # ── Header ────────────────────────────────────────────────────────────────
    upsell   = result.expansion_analytics.upsell_opportunity_detected
    csell    = result.cross_sell_opportunity_window.cross_sell_detected
    upsell_tag  = "[bold green]✓ UPSELL DETECTED[/bold green]" if upsell else "[dim]No upsell[/dim]"
    csell_tag   = "[bold cyan]✓ CROSS-SELL DETECTED[/bold cyan]" if csell else "[dim]No cross-sell[/dim]"

    # ── Main Metrics ─────────────────────────────────────────────────────────
    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Field", style="dim", no_wrap=True)
    metrics_table.add_column("Value", style="bold white", no_wrap=True)

    metrics_table.add_row("Predicted CLV",
        f"[bold green]₹{result.predicted_clv_amount:>12,.0f}[/bold green]"
    )
    metrics_table.add_row("Confidence",
        f"{result.confidence_interval_percentage:.1f}%"
    )
    metrics_table.add_row("Historical Spend",
        f"₹{payload.historical_spend_amount:>10,.0f}"
    )
    metrics_table.add_row("CLV / Spend Ratio",
        f"{(result.predicted_clv_amount / max(payload.historical_spend_amount, 1)):.2f}x"
    )
    metrics_table.add_row("Engagement Score",
        f"{payload.rolling_engagement_score:.1f}/100"
    )
    metrics_table.add_row("Contract Tenure",
        f"{payload.contract_tenure_months} months"
    )
    metrics_table.add_row("Industry",
        f"[bold yellow]{payload.industry_type.value}[/bold yellow]"
    )

    # ── SHAP Drivers ─────────────────────────────────────────────────────────
    shap_table = Table(title="Top SHAP Drivers", box=box.SIMPLE_HEAVY, padding=(0, 1))
    shap_table.add_column("#", style="dim", width=3)
    shap_table.add_column("Feature", style="bold")
    shap_table.add_column("Value", justify="right")
    shap_table.add_column("Impact", justify="right")
    shap_table.add_column("Share", justify="right")

    for i, driver in enumerate(result.expansion_analytics.shap_drivers, 1):
        # SHAPDriverEntry can be a dict (from model_dump) or a Pydantic model object
        d = driver if isinstance(driver, dict) else driver.model_dump()
        impact_color = "green" if d["direction"] == "positive" else "red"
        arrow = "\u25b2" if d["direction"] == "positive" else "\u25bc"
        val_str = (
            f"\u20b9{d['value']:,.0f}" if isinstance(d["value"], (int, float))
            else str(d["value"])
        )
        shap_table.add_row(
            str(i),
            d["feature"],
            val_str,
            f"[{impact_color}]{arrow} {abs(d['shap_contribution']):,.0f}[/{impact_color}]",
            f"{d['contribution_pct']:.0f}%",
        )

    # ── Upsell Panel ─────────────────────────────────────────────────────────
    ea = result.expansion_analytics
    upsell_text = (
        f"ML Score: {ea.upsell_confidence_score:.2%}\n"
        f"Product : {ea.recommended_target_product}\n"
        f"Driver  : {ea.top_prediction_driver_reason[:120]}..."
        if len(ea.top_prediction_driver_reason) > 120
        else f"Driver  : {ea.top_prediction_driver_reason}"
    )
    upsell_color = "green" if upsell else "dim white"

    # ── Cross-sell Panel ─────────────────────────────────────────────────────
    csw = result.cross_sell_opportunity_window
    cross_text = (
        f"Product : {csw.secondary_product_name}\n"
        f"Pitch By: {csw.optimal_pitch_target_date} ({csw.days_until_optimal_pitch} days)\n"
        f"Timing  : {csw.timing_confidence_score:.2%} confidence\n"
        f"Why     : {csw.pitch_rationale[:110]}..."
        if len(csw.pitch_rationale) > 110
        else (
            f"Product : {csw.secondary_product_name}\n"
            f"Pitch By: {csw.optimal_pitch_target_date} ({csw.days_until_optimal_pitch} days)\n"
            f"Timing  : {csw.timing_confidence_score:.2%} confidence\n"
            f"Why     : {csw.pitch_rationale}"
        )
    )
    cross_color = "cyan" if csell else "dim white"

    # ── Render ────────────────────────────────────────────────────────────────
    console.print(Rule(f"[bold magenta]Client {idx}[/bold magenta]: {label}"))
    console.print()
    console.print(metrics_table)
    console.print()
    console.print(shap_table)
    console.print()

    side_by_side = Columns([
        Panel(
            Text(upsell_text, no_wrap=False),
            title=upsell_tag,
            border_style=upsell_color,
            width=60,
        ),
        Panel(
            Text(cross_text, no_wrap=False),
            title=csell_tag,
            border_style=cross_color,
            width=60,
        ),
    ])
    console.print(side_by_side)
    console.print()


def demo_single(predictor: CLVPredictor) -> None:
    """Run a single client demo (first client in the list)."""
    client  = DEMO_CLIENTS[0]
    payload = _build_input(client)

    console.print(Rule("[bold]Single-Account Prediction Demo[/bold]"))
    with Progress(SpinnerColumn(), TextColumn("[cyan]Running inference...[/cyan]"),
                  transient=True, console=console) as p:
        p.add_task("", total=None)
        result = predictor.predict(payload)

    _render_prediction(client["label"], payload, result, 1)


def demo_batch(predictor: CLVPredictor) -> None:
    """Run all demo clients and show a comparison summary table."""
    console.print(Rule("[bold]Batch Prediction Demo — 6 Enterprise Client Accounts[/bold]"))
    console.print()

    results = []
    with Progress(SpinnerColumn(), TextColumn("[cyan]Running batch predictions...[/cyan]"),
                  transient=True, console=console) as p:
        task = p.add_task("", total=len(DEMO_CLIENTS))
        for client in DEMO_CLIENTS:
            payload = _build_input(client)
            result  = predictor.predict(payload)
            results.append((client, payload, result))
            p.advance(task)

    # ── Individual results ────────────────────────────────────────────────────
    for i, (client, payload, result) in enumerate(results, 1):
        _render_prediction(client["label"], payload, result, i)
        time.sleep(0.1)  # pacing for readability

    # ── Comparison summary table ───────────────────────────────────────────────
    console.print(Rule("[bold yellow]Batch Summary — Executive View[/bold yellow]"))
    console.print()

    summary = Table(
        title="CLV Batch Results",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        header_style="bold magenta",
    )
    summary.add_column("Client", style="bold")
    summary.add_column("Industry", style="yellow")
    summary.add_column("CLV (₹)", justify="right", style="green")
    summary.add_column("vs Spend", justify="right")
    summary.add_column("Confidence", justify="right")
    summary.add_column("Upsell", justify="center")
    summary.add_column("Cross-Sell", justify="center")
    summary.add_column("Pitch Date")

    for client, payload, result in results:
        ratio = result.predicted_clv_amount / max(payload.historical_spend_amount, 1)
        upsell_sym = "✓" if result.expansion_analytics.upsell_opportunity_detected else "–"
        csell_sym  = "✓" if result.cross_sell_opportunity_window.cross_sell_detected else "–"
        upsell_style = "bold green" if result.expansion_analytics.upsell_opportunity_detected else "dim"
        csell_style  = "bold cyan"  if result.cross_sell_opportunity_window.cross_sell_detected else "dim"

        summary.add_row(
            client["label"][:30],
            payload.industry_type.value[:20],
            f"₹{result.predicted_clv_amount:,.0f}",
            f"{ratio:.2f}x",
            f"{result.confidence_interval_percentage:.1f}%",
            f"[{upsell_style}]{upsell_sym}[/{upsell_style}]",
            f"[{csell_style}]{csell_sym}[/{csell_style}]",
            result.cross_sell_opportunity_window.optimal_pitch_target_date,
        )

    console.print(summary)
    console.print()


def demo_db() -> None:
    """Show what's in the predictions database."""
    from database.db import get_summary_stats, init_db, query_recent_predictions

    # Ensure the table exists even if the API has never been started
    init_db()

    console.print(Rule("[bold]Database — Stored Predictions View[/bold]"))
    console.print()

    stats = get_summary_stats()
    if not stats or stats.get("total_predictions", 0) == 0:
        console.print("[dim]No predictions in the database yet. Run the batch demo first.[/dim]")
        return

    stats_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", style="bold white")
    stats_table.add_row("Total Predictions", str(stats.get("total_predictions", 0)))
    stats_table.add_row("Avg Predicted CLV", f"₹{stats.get('avg_predicted_clv', 0):,.0f}")
    stats_table.add_row("Upsell Rate", f"{stats.get('upsell_rate_pct', 0):.1f}%")
    stats_table.add_row("Cross-Sell Rate", f"{stats.get('cross_sell_rate_pct', 0):.1f}%")
    stats_table.add_row("Avg Confidence", f"{stats.get('avg_confidence_pct', 0):.1f}%")

    console.print(Panel(stats_table, title="[bold]Executive Summary Stats[/bold]", border_style="yellow"))
    console.print()

    rows = query_recent_predictions(limit=8)
    if rows:
        db_table = Table(title="Recent Predictions (DB)", box=box.ROUNDED, show_lines=True)
        db_table.add_column("Timestamp", style="dim")
        db_table.add_column("Company ID")
        db_table.add_column("CLV (₹)", justify="right", style="green")
        db_table.add_column("Confidence", justify="right")
        db_table.add_column("Upsell", justify="center")
        db_table.add_column("Pitch Date")

        for r in rows:
            db_table.add_row(
                r["prediction_timestamp"][:19],
                r["company_id"],
                f"₹{r['predicted_clv_amount']:,.0f}",
                f"{r['confidence_interval_pct']:.1f}%",
                "✓" if r["upsell_opportunity_detected"] else "–",
                r["optimal_pitch_target_date"],
            )
        console.print(db_table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="CLV Engine Terminal Demo")
    parser.add_argument(
        "--scenario", choices=["all", "single", "batch", "db"],
        default="all",
        help="Which demo scenario to run (default: all)"
    )
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]AI/ML Platform[/bold white]\n"
            "[bold cyan]Client Lifetime Value (CLV) Prediction Engine[/bold cyan]\n"
            "[dim]Module 8 — v2.0 | XGBoost · SHAP · FastAPI · SQLite[/dim]\n"
            f"[dim]Demo run: {date.today().strftime('%B %d, %Y')}[/dim]"
        ),
        border_style="bright_magenta",
        padding=(1, 4),
    ))
    console.print()

    # ── Load predictor ────────────────────────────────────────────────────────
    predictor = _load_predictor()
    console.print(f"[bold green]✓ Model loaded.[/bold green] Training R² = {predictor._train_r2:.4f}\n")

    # ── Run scenarios ─────────────────────────────────────────────────────────
    if args.scenario in ("all", "single"):
        demo_single(predictor)

    if args.scenario in ("all", "batch"):
        demo_batch(predictor)

    if args.scenario in ("all", "db"):
        demo_db()

    # ── Footer ────────────────────────────────────────────────────────────────
    console.print(Rule("[bold green]Demo Complete[/bold green]"))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. [cyan]uvicorn main:app --reload[/cyan]   → Start the live API\n"
        "  2. [cyan]http://127.0.0.1:8000/docs[/cyan]  → Swagger UI\n"
        "  3. [cyan]http://127.0.0.1:8000/api/v1/clv/summary[/cyan]  → DB stats\n"
    )


if __name__ == "__main__":
    main()
