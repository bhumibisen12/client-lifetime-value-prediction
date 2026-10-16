"""
clv_cli.py — CLV Prediction Engine | Production Terminal Interface

A production-grade CLI that calls the live CLV API and presents results
as actionable business intelligence — not ML scores.

Modes:
    python clv_cli.py predict          → Interactive: enter client details, get prediction
    python clv_cli.py predict --json   → Pipe raw JSON to it
    python clv_cli.py batch            → Predict for multiple clients from a JSON file
    python clv_cli.py history <id>     → Show prediction history for a client
    python clv_cli.py dashboard        → Show overall account pipeline summary
    python clv_cli.py status           → Check if the API is running

Usage examples:
    python clv_cli.py predict
    python clv_cli.py predict --file client.json
    python clv_cli.py batch --file clients.json
    python clv_cli.py history client-tech-001
    python clv_cli.py dashboard
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

API_BASE = "http://127.0.0.1:8000/api/v1/clv"

INDUSTRY_CHOICES = [
    "Technology",
    "Finance",
    "Healthcare",
    "Retail",
    "Manufacturing",
    "Logistics & Supply Chain",
    "Education",
    "Real Estate",
    "Hospitality",
    "E-commerce",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_api() -> bool:
    """Return True if the API is reachable."""
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _require_api() -> None:
    if not _check_api():
        console.print(
            "\n[bold red]API is not running.[/bold red]\n"
            "Start it first in a separate terminal:\n"
            "  [bold yellow]uvicorn main:app --reload[/bold yellow]\n"
        )
        sys.exit(1)


def _confidence_label(pct: float) -> str:
    """Convert confidence % to a human label."""
    if pct >= 85:
        return "[bold green]HIGH[/bold green]"
    elif pct >= 70:
        return "[green]GOOD[/green]"
    elif pct >= 55:
        return "[yellow]MODERATE[/yellow]"
    else:
        return "[red]LOW — treat as estimate[/red]"


def _upsell_label(score: float, detected: bool) -> str:
    if detected:
        if score >= 0.80:
            return "[bold green]STRONG SIGNAL[/bold green]"
        elif score >= 0.65:
            return "[green]POSITIVE SIGNAL[/green]"
        else:
            return "[yellow]WEAK SIGNAL[/yellow]"
    return "[dim]Not flagged[/dim]"


def _timing_label(score: float) -> str:
    if score >= 0.75:
        return "[bold green]Excellent window[/bold green]"
    elif score >= 0.55:
        return "[green]Good window[/green]"
    elif score >= 0.35:
        return "[yellow]Acceptable window[/yellow]"
    else:
        return "[red]Poor timing — consider waiting[/red]"


def _days_label(days: int) -> str:
    if days <= 14:
        return f"[bold cyan]{days} days (act soon)[/bold cyan]"
    elif days <= 30:
        return f"[cyan]{days} days[/cyan]"
    else:
        return f"[dim]{days} days[/dim]"


def _clv_growth_label(clv: float, spend: float) -> str:
    ratio = clv / max(spend, 1)
    if ratio >= 10:
        return f"[bold green]{ratio:.1f}x growth projected[/bold green]"
    elif ratio >= 5:
        return f"[green]{ratio:.1f}x growth projected[/green]"
    elif ratio >= 3:
        return f"[yellow]{ratio:.1f}x growth projected[/yellow]"
    else:
        return f"[dim]{ratio:.1f}x growth projected[/dim]"


def _format_inr(amount: float) -> str:
    """Format a number as Indian rupee with lakh/crore notation."""
    if amount >= 1_00_00_000:  # 1 crore
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:   # 1 lakh
        return f"₹{amount / 1_00_000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


# ── Render ────────────────────────────────────────────────────────────────────

def _render_prediction(result: dict, input_data: dict) -> None:
    """
    Render a single CLV prediction as a production-grade account intelligence report.
    All output is in business language — no raw ML scores shown.
    """
    ea  = result["expansion_analytics"]
    csw = result["cross_sell_opportunity_window"]

    clv   = result["predicted_clv_amount"]
    spend = input_data.get("historical_spend_amount", 0)
    cid   = result["company_id"]
    conf  = result["confidence_interval_percentage"]

    console.print()
    console.print(Rule(f"[bold magenta]Account Intelligence Report[/bold magenta]"))

    # ── Revenue Forecast ─────────────────────────────────────────────────────
    forecast_table = Table(show_header=False, box=None, padding=(0, 3))
    forecast_table.add_column("Label", style="dim", width=30)
    forecast_table.add_column("Value", style="bold white")

    forecast_table.add_row("Client ID",        f"[bold]{cid}[/bold]")
    forecast_table.add_row("Industry",          input_data.get("industry_type", "—"))
    forecast_table.add_row(
        "Lifetime Revenue Forecast",
        f"[bold green]{_format_inr(clv)}[/bold green]"
    )
    forecast_table.add_row(
        "vs. Historical Spend",
        _clv_growth_label(clv, spend)
    )
    forecast_table.add_row(
        "Forecast Confidence",
        _confidence_label(conf)
    )
    forecast_table.add_row(
        "Contract Tenure",
        f"{input_data.get('contract_tenure_months', '—')} months active"
    )
    forecast_table.add_row(
        "Engagement Level",
        _engagement_bar(input_data.get("rolling_engagement_score", 0))
    )
    forecast_table.add_row(
        "Report Generated",
        result.get("prediction_timestamp", "—")[:19].replace("T", " ") + " UTC"
    )

    console.print(Panel(
        forecast_table,
        title="[bold]Revenue Forecast[/bold]",
        border_style="bright_blue",
    ))

    # ── What's driving this forecast ─────────────────────────────────────────
    drivers = ea.get("shap_drivers", [])
    if drivers:
        console.print()
        console.print("[bold dim]What's driving this forecast:[/bold dim]")
        for i, d in enumerate(drivers, 1):
            d = d if isinstance(d, dict) else d
            direction_icon = "▲" if d["direction"] == "positive" else "▼"
            color = "green" if d["direction"] == "positive" else "red"
            # Translate feature name into plain business language
            feat_plain = _feature_to_business_language(d["feature"], d["value"], d["direction"])
            console.print(
                f"  [{color}]{direction_icon}[/{color}] {feat_plain}"
            )

    # ── Upsell Recommendation ─────────────────────────────────────────────────
    console.print()
    upsell_detected = ea["upsell_opportunity_detected"]

    if upsell_detected:
        upsell_panel_content = (
            f"[bold]Recommended Action:[/bold] Pitch the following upgrade to this account\n\n"
            f"[bold cyan]{ea['recommended_target_product']}[/bold cyan]\n\n"
            f"[dim]Why:[/dim] {ea['top_prediction_driver_reason']}\n\n"
            f"[bold]Signal Strength:[/bold] {_upsell_label(ea['upsell_confidence_score'], True)}"
        )
        console.print(Panel(
            Text.from_markup(upsell_panel_content),
            title="[bold green]⬆  UPSELL OPPORTUNITY IDENTIFIED[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[dim]This account does not meet the upsell criteria at this time.\n"
            f"Continue standard engagement and re-evaluate at next billing cycle.[/dim]\n\n"
            f"Signal: {_upsell_label(ea['upsell_confidence_score'], False)} "
            f"(score: {ea['upsell_confidence_score']:.0%})",
            title="[dim]Upsell — No Action Required[/dim]",
            border_style="dim",
        ))

    # ── Cross-Sell Window ─────────────────────────────────────────────────────
    console.print()
    cross_detected = csw["cross_sell_detected"]

    pitch_date_str = csw["optimal_pitch_target_date"]
    try:
        pitch_date_obj = date.fromisoformat(pitch_date_str)
        pitch_date_display = pitch_date_obj.strftime("%B %d, %Y")
    except Exception:
        pitch_date_display = pitch_date_str

    if cross_detected:
        cross_content = (
            f"[bold]Module to pitch:[/bold] [cyan]{csw['secondary_product_name']}[/cyan]\n\n"
            f"[bold]Recommended Pitch Date:[/bold] [bold cyan]{pitch_date_display}[/bold cyan]  "
            f"({_days_label(csw['days_until_optimal_pitch'])})\n\n"
            f"[bold]Timing Assessment:[/bold] {_timing_label(csw['timing_confidence_score'])}\n\n"
            f"[dim]{csw['pitch_rationale']}[/dim]"
        )
        console.print(Panel(
            Text.from_markup(cross_content),
            title="[bold cyan]↔  CROSS-SELL WINDOW OPEN[/bold cyan]",
            border_style="cyan",
        ))
    else:
        console.print(Panel(
            f"[dim]No cross-sell window identified for this account right now.\n"
            f"{csw['pitch_rationale']}[/dim]",
            title="[dim]Cross-Sell — No Action Required[/dim]",
            border_style="dim",
        ))

    # ── Action Summary ────────────────────────────────────────────────────────
    console.print()
    actions = []
    if upsell_detected:
        actions.append(f"[green]→[/green] Schedule upgrade pitch: [bold]{ea['recommended_target_product'][:60]}[/bold]")
    if cross_detected:
        actions.append(f"[cyan]→[/cyan] Book cross-sell call by [bold]{pitch_date_display}[/bold] for [bold]{csw['secondary_product_name']}[/bold]")
    if not upsell_detected and not cross_detected:
        actions.append("[dim]→ No immediate revenue actions. Focus on engagement and retention.[/dim]")

    action_box = "\n".join(actions)
    console.print(Panel(
        Text.from_markup(action_box),
        title="[bold yellow]  Action Items for Account Manager[/bold yellow]",
        border_style="yellow",
    ))
    console.print()


def _engagement_bar(score: float) -> str:
    filled = int(score / 10)
    bar = "█" * filled + "░" * (10 - filled)
    if score >= 75:
        color = "green"
    elif score >= 50:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{bar}[/{color}] {score:.0f}/100"


def _feature_to_business_language(feature: str, value, direction: str) -> str:
    """Convert SHAP feature name into plain business English."""
    val_str = f"₹{value:,.0f}" if isinstance(value, (int, float)) else str(value)
    positive = direction == "positive"

    if "Engagement" in feature and "Tenure" in feature:
        return (
            f"Long-term, highly engaged account — this combination "
            f"{'strongly boosts' if positive else 'limits'} the revenue forecast."
        )
    elif "Engagement" in feature:
        return (
            f"Client engagement ({value:.0f}/100) is "
            f"{'actively boosting' if positive else 'dragging down'} the CLV score."
        )
    elif "Historical Spend" in feature:
        return (
            f"Historical spend of {val_str} is "
            f"{'a strong positive indicator' if positive else 'a limiting factor'} for future value."
        )
    elif "Tenure" in feature:
        return (
            f"Contract tenure ({value:.0f} months) "
            f"{'increases confidence' if positive else 'is short — limits long-term projection'}."
        )
    elif "Velocity" in feature:
        return (
            f"Spending velocity of {val_str}/month is "
            f"{'above average — good sign' if positive else 'below baseline — monitor closely'}."
        )
    elif "Frequency" in feature:
        return (
            f"Purchase frequency of {value:.0f}/year is "
            f"{'consistent — positive signal' if positive else 'low — consider re-engagement'}."
        )
    else:
        return f"{feature}: {val_str} ({'positive' if positive else 'negative'} impact on CLV)"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_status() -> None:
    """Check if the API is up."""
    console.print()
    if _check_api():
        try:
            r = httpx.get("http://127.0.0.1:8000/health", timeout=3)
            info = r.json()
            console.print(Panel(
                f"[bold green][ONLINE] API is running[/bold green]\n\n"
                f"Service : {info.get('service', 'CLV Engine')}\n"
                f"Version : {info.get('version', '—')}\n"
                f"Endpoint: [cyan]http://127.0.0.1:8000[/cyan]\n"
                f"Swagger : [cyan]http://127.0.0.1:8000/docs[/cyan]",
                title="API Status",
                border_style="green",
            ))
        except Exception:
            console.print("[bold green][ONLINE] API is running[/bold green]")
    else:
        console.print(Panel(
            "[bold red][OFFLINE] API is not running[/bold red]\n\n"
            "Start with:\n  [yellow]uvicorn main:app --reload[/yellow]",
            title="API Status",
            border_style="red",
        ))
    console.print()


def cmd_predict(file: Optional[str] = None, as_json: bool = False) -> None:
    """Interactive single-account prediction."""
    _require_api()

    if file:
        # Load from JSON file
        try:
            with open(file) as f:
                payload = json.load(f)
            # If the file contains a list, take the first element and warn
            if isinstance(payload, list):
                if len(payload) > 1:
                    console.print(
                        f"[yellow]Note:[/yellow] File contains {len(payload)} clients. "
                        "Predicting for the first one only. "
                        "Use [bold]batch --file[/bold] to run all.\n"
                    )
                payload = payload[0]
        except Exception as e:
            console.print(f"[red]Could not read file: {e}[/red]")
            sys.exit(1)
    else:
        # Interactive prompt
        console.print()
        console.print(Rule("[bold]CLV Prediction — Account Details[/bold]"))
        console.print("[dim]Enter the client account details below.[/dim]\n")

        company_id = Prompt.ask("[bold]Client ID[/bold]", default="client-001")

        console.print("\n[bold]Industry[/bold] (choose number):")
        for i, ind in enumerate(INDUSTRY_CHOICES, 1):
            console.print(f"  {i:2}. {ind}")
        while True:
            choice = IntPrompt.ask("Enter number", default=1)
            if 1 <= choice <= len(INDUSTRY_CHOICES):
                industry = INDUSTRY_CHOICES[choice - 1]
                break
            console.print("[red]Invalid choice.[/red]")

        console.print()
        spend         = FloatPrompt.ask("[bold]Historical Spend (INR)[/bold]", default=20000.0)
        frequency     = IntPrompt.ask("[bold]Purchase Frequency (per year)[/bold]", default=4)
        engagement    = FloatPrompt.ask("[bold]Engagement Score (0–100)[/bold]", default=75.0)
        tenure        = IntPrompt.ask("[bold]Contract Tenure (months)[/bold]", default=18)

        while True:
            raw_start_date = Prompt.ask(
                "[bold]Contract Start Date[/bold] (YYYY-MM-DD, optional — press Enter to skip)",
                default=""
            ).strip()

            if not raw_start_date:
                start_date = None
                break

            try:
                parsed_date = date.fromisoformat(raw_start_date)
                if parsed_date > date.today():
                    console.print(Panel(
                        f"[bold red]❌ Invalid Date Error[/bold red]\n\n"
                        f"Contract start date cannot be in the future: [yellow]{raw_start_date}[/yellow] (Today is {date.today()}).\n"
                        f"The contract must have already started in the past.",
                        title="[bold red]Date Validation Failed[/bold red]",
                        border_style="red",
                    ))
                    continue
                start_date = raw_start_date
                break
            except ValueError:
                console.print(Panel(
                    f"[bold red]❌ Invalid Date Format[/bold red]\n\n"
                    f"You entered: [yellow]{raw_start_date}[/yellow]\n"
                    f"Expected format: [bold cyan]YYYY-MM-DD[/bold cyan] (e.g. 2024-01-15).\n"
                    f"Please enter a valid calendar date or press [bold]Enter[/bold] to skip.",
                    title="[bold red]Date Validation Failed[/bold red]",
                    border_style="red",
                ))

        payload = {
            "company_id":                company_id,
            "historical_spend_amount":   spend,
            "purchase_frequency_yearly": frequency,
            "industry_type":             industry,
            "rolling_engagement_score":  engagement,
            "contract_tenure_months":    tenure,
        }
        if start_date:
            payload["contract_start_date"] = start_date

    # Call API
    console.print("\n[dim]Running prediction...[/dim]")
    try:
        r = httpx.post(f"{API_BASE}/predict", json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()
    except httpx.HTTPStatusError as e:
        error_msg = e.response.text
        try:
            err_json = e.response.json()
            if "detail" in err_json:
                detail = err_json["detail"]
                if isinstance(detail, list):
                    lines = []
                    for item in detail:
                        loc = " -> ".join(str(x) for x in item.get("loc", []))
                        msg = item.get("msg", "")
                        lines.append(f"- [bold]{loc}[/bold]: {msg}")
                    error_msg = "\n".join(lines)
                else:
                    error_msg = str(detail)
        except Exception:
            pass
        console.print(Panel(
            f"[bold red]Validation / API Error ({e.response.status_code})[/bold red]\n\n{error_msg}",
            title="[bold red]Prediction Request Rejected[/bold red]",
            border_style="red",
        ))
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Request failed:[/red] {e}")
        sys.exit(1)

    if as_json:
        console.print_json(data=result)
    else:
        _render_prediction(result, payload)


def cmd_batch(file: str) -> None:
    """Batch predict from a JSON file containing a list of client payloads."""
    _require_api()

    try:
        with open(file) as f:
            payloads = json.load(f)
    except Exception as e:
        console.print(f"[red]Could not read file: {e}[/red]")
        sys.exit(1)

    if not isinstance(payloads, list):
        console.print("[red]Batch file must contain a JSON array of client objects.[/red]")
        sys.exit(1)

    console.print(f"\n[dim]Running batch prediction for {len(payloads)} accounts...[/dim]\n")
    try:
        r = httpx.post(f"{API_BASE}/batch-predict", json=payloads, timeout=60)
        r.raise_for_status()
        results = r.json()
    except Exception as e:
        console.print(f"[red]Batch request failed:[/red] {e}")
        sys.exit(1)

    # Priority-ranked action table
    console.print(Rule("[bold]Batch Results — Ranked Action List[/bold]"))
    console.print()

    # Sort: upsell first, then cross-sell, then by CLV desc
    def sort_key(item):
        res, inp = item
        return (
            -int(res["expansion_analytics"]["upsell_opportunity_detected"]),
            -int(res["cross_sell_opportunity_window"]["cross_sell_detected"]),
            -res["predicted_clv_amount"],
        )

    paired = list(zip(results, payloads[:len(results)]))
    paired.sort(key=sort_key)

    t = Table(box=box.ROUNDED, show_lines=True, header_style="bold magenta")
    t.add_column("Priority", width=6, justify="center")
    t.add_column("Client ID")
    t.add_column("Industry")
    t.add_column("CLV Forecast", justify="right")
    t.add_column("Growth", justify="right")
    t.add_column("Upsell", justify="center")
    t.add_column("Cross-Sell", justify="center")
    t.add_column("Action Deadline")

    for rank, (res, inp) in enumerate(paired, 1):
        ea  = res["expansion_analytics"]
        csw = res["cross_sell_opportunity_window"]
        clv = res["predicted_clv_amount"]
        spend = inp.get("historical_spend_amount", 1)
        ratio = clv / max(spend, 1)

        upsell_sym = "[bold green]✓ Pitch now[/bold green]" if ea["upsell_opportunity_detected"] else "[dim]–[/dim]"
        cross_sym  = "[bold cyan]✓ Open[/bold cyan]" if csw["cross_sell_detected"] else "[dim]–[/dim]"

        if ea["upsell_opportunity_detected"] or csw["cross_sell_detected"]:
            deadline = csw["optimal_pitch_target_date"]
            try:
                d = date.fromisoformat(deadline)
                deadline_display = d.strftime("%b %d, %Y")
            except Exception:
                deadline_display = deadline
        else:
            deadline_display = "[dim]No action[/dim]"

        rank_color = "bold green" if rank <= 2 else ("yellow" if rank <= 4 else "dim")
        t.add_row(
            f"[{rank_color}]#{rank}[/{rank_color}]",
            res["company_id"],
            inp.get("industry_type", "—")[:20],
            f"[green]{_format_inr(clv)}[/green]",
            f"{ratio:.1f}x",
            upsell_sym,
            cross_sym,
            deadline_display,
        )

    console.print(t)

    # Quick stats
    total    = len(results)
    upsells  = sum(1 for r in results if r["expansion_analytics"]["upsell_opportunity_detected"])
    crosses  = sum(1 for r in results if r["cross_sell_opportunity_window"]["cross_sell_detected"])
    avg_clv  = sum(r["predicted_clv_amount"] for r in results) / max(total, 1)

    console.print(
        f"\n[dim]Batch complete:[/dim] {total} accounts processed  │  "
        f"[green]{upsells} upsell opportunities[/green]  │  "
        f"[cyan]{crosses} cross-sell windows[/cyan]  │  "
        f"Avg CLV: [bold]{_format_inr(avg_clv)}[/bold]\n"
    )

    # Ask if user wants individual reports
    if total <= 10 and Confirm.ask("Show full individual reports?", default=False):
        for res, inp in paired:
            _render_prediction(res, inp)


def cmd_history(company_id: str) -> None:
    """Show prediction history for a client."""
    _require_api()

    try:
        r = httpx.get(f"{API_BASE}/history/{company_id}", timeout=10)
        if r.status_code == 404:
            console.print(f"\n[yellow]No prediction history found for:[/yellow] {company_id}\n")
            return
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        console.print(f"[red]Request failed:[/red] {e}")
        sys.exit(1)

    console.print()
    console.print(Rule(f"[bold]Prediction History — {company_id}[/bold]"))

    t = Table(box=box.SIMPLE_HEAVY, show_lines=False, header_style="bold")
    t.add_column("Date", style="dim")
    t.add_column("CLV Forecast", justify="right")
    t.add_column("Confidence")
    t.add_column("Upsell", justify="center")
    t.add_column("Pitch Deadline")
    t.add_column("Top Driver", max_width=50)

    for row in rows:
        ts = row.get("prediction_timestamp", "")[:10]
        clv = row.get("predicted_clv_amount", 0)
        conf_pct = row.get("confidence_interval_pct", 0)
        upsell = "[green]Yes[/green]" if row.get("upsell_opportunity_detected") else "[dim]No[/dim]"
        pitch = row.get("optimal_pitch_target_date", "—")
        driver = row.get("top_prediction_driver_reason", "")[:60] + "..."

        t.add_row(ts, _format_inr(clv), _confidence_label(conf_pct), upsell, pitch, driver)

    console.print(t)
    console.print(f"[dim]{len(rows)} records found.[/dim]\n")


def cmd_dashboard() -> None:
    """Show overall pipeline summary from the database."""
    _require_api()

    try:
        summary_r = httpx.get(f"{API_BASE}/summary", timeout=10)
        recent_r  = httpx.get(f"{API_BASE}/recent?limit=8", timeout=10)
        summary_r.raise_for_status()
        recent_r.raise_for_status()
        stats  = summary_r.json()
        recent = recent_r.json()
    except Exception as e:
        console.print(f"[red]Request failed:[/red] {e}")
        sys.exit(1)

    console.print()
    console.print(Rule("[bold]Account Pipeline Dashboard[/bold]"))

    if "message" in stats:
        console.print(f"\n[dim]{stats['message']}[/dim]\n")
        return

    # KPI row
    kpi = Table(show_header=False, box=None, padding=(0, 4))
    kpi.add_column("", justify="center")
    kpi.add_column("", justify="center")
    kpi.add_column("", justify="center")
    kpi.add_column("", justify="center")

    kpi.add_row(
        f"[bold green]{_format_inr(stats.get('avg_predicted_clv', 0))}[/bold green]",
        f"[bold cyan]{stats.get('upsell_rate_pct', 0):.1f}%[/bold cyan]",
        f"[bold yellow]{stats.get('cross_sell_rate_pct', 0):.1f}%[/bold yellow]",
        f"[bold]{int(stats.get('total_predictions', 0))}[/bold]",
    )
    kpi.add_row(
        "[dim]Avg Predicted CLV[/dim]",
        "[dim]Upsell Rate[/dim]",
        "[dim]Cross-Sell Rate[/dim]",
        "[dim]Total Predictions[/dim]",
    )

    console.print(Panel(kpi, title="[bold]Pipeline KPIs[/bold]", border_style="bright_blue"))

    # Recent predictions
    if recent:
        console.print()
        console.print("[bold dim]Recent Accounts:[/bold dim]")
        t = Table(box=box.SIMPLE, show_lines=False)
        t.add_column("Client ID")
        t.add_column("Industry")
        t.add_column("CLV Forecast", justify="right")
        t.add_column("Upsell", justify="center")
        t.add_column("Pitch By")
        t.add_column("Predicted At", style="dim")

        for row in recent:
            upsell = "[green]✓[/green]" if row.get("upsell_opportunity_detected") else "[dim]–[/dim]"
            t.add_row(
                row.get("company_id", "—"),
                row.get("industry_type", "—")[:20],
                _format_inr(row.get("predicted_clv_amount", 0)),
                upsell,
                row.get("optimal_pitch_target_date", "—"),
                row.get("prediction_timestamp", "")[:19].replace("T", " "),
            )
        console.print(t)
    console.print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clv_cli",
        description="CLV Prediction Engine — Production Terminal Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status               Check if the API is running
  predict              Predict CLV for one account (interactive or from file)
  batch                Predict for multiple accounts from a JSON file
  history <client_id>  Show prediction history for a specific client
  dashboard            Show pipeline KPI dashboard

Examples:
  python clv_cli.py status
  python clv_cli.py predict
  python clv_cli.py predict --file client.json
  python clv_cli.py batch --file clients.json
  python clv_cli.py history client-tech-001
  python clv_cli.py dashboard
""",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Check if the API is running")

    p_predict = sub.add_parser("predict", help="Predict CLV for one account")
    p_predict.add_argument("--file", help="JSON file with a single client payload")
    p_predict.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON response with all technical schema keys (predicted_clv_amount, shap_drivers, etc.)",
    )

    p_batch = sub.add_parser("batch", help="Predict CLV for multiple accounts")
    p_batch.add_argument("--file", required=True, help="JSON file with a list of client payloads")

    p_history = sub.add_parser("history", help="Show prediction history for a client")
    p_history.add_argument("client_id", help="Client ID to look up")

    sub.add_parser("dashboard", help="Show account pipeline KPI dashboard")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "predict":
        cmd_predict(file=getattr(args, "file", None), as_json=getattr(args, "json", False))
    elif args.command == "batch":
        cmd_batch(file=args.file)
    elif args.command == "history":
        cmd_history(args.client_id)
    elif args.command == "dashboard":
        cmd_dashboard()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
