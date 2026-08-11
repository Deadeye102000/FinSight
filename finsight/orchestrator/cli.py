"""Typer CLI entrypoint for FinSight LangGraph Orchestration Layer."""

from __future__ import annotations

from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from finsight.orchestrator.pipeline import run_research_pipeline

app = typer.Typer(
    name="FinSight Research CLI",
    help="Autonomous stock research pipeline orchestrated by LangGraph StateGraph.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    ticker: str = typer.Argument(..., help="Stock ticker symbol (e.g. AAPL, MSFT, TCS.NS)"),
    peers: Optional[List[str]] = typer.Option(
        None,
        "--peer",
        "-p",
        help="Peer tickers to compare against (e.g., -p MSFT -p GOOGL)",
    ),
) -> None:
    """Run full FinSight research pipeline end-to-end and print final state."""
    console.print(f"[bold cyan]Launching FinSight LangGraph Pipeline for {ticker.upper()}...[/bold cyan]")
    if peers:
        console.print(f"[dim]Peers: {', '.join(peers)}[/dim]")

    try:
        final_state = run_research_pipeline(ticker=ticker, peer_tickers=peers)
        console.print("\n[bold green]=== Research Pipeline State Output ===[/bold green]")
        console.print(Pretty(final_state.model_dump()))

        if final_state.report:
            console.print("\n[bold yellow]=== Generated Research Report ===[/bold yellow]")
            console.print(Panel(final_state.report, title=f"Report: {ticker.upper()}"))

    except Exception as exc:
        console.print(f"[bold red]Pipeline failed: {exc}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
