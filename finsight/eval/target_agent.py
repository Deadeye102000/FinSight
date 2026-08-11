"""FinSight TargetAgent implementation bridging to agent-eval-orchestrator."""

from __future__ import annotations

import logging
import time
from typing import Any

from finsight.orchestrator.pipeline import run_research_pipeline

logger = logging.getLogger(__name__)


class FinSightTargetAgent:
    """TargetAgent protocol implementation for FinSight's LangGraph StateGraph pipeline."""

    def __init__(self, version: str = "finsight-langgraph-v1.0") -> None:
        self.version = version

    async def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        """Invoke FinSight's StateGraph research pipeline asynchronously."""
        ticker = str(input.get("ticker") or input.get("symbol") or "AAPL").strip().upper()
        raw_peers = input.get("peer_tickers") or input.get("peers") or ["MSFT", "GOOGL"]
        if isinstance(raw_peers, str):
            raw_peers = [p.strip().upper() for p in raw_peers.split(",") if p.strip()]

        peer_tickers = [str(p).upper() for p in raw_peers if str(p).upper() != ticker]
        if len(peer_tickers) < 2:
            peer_tickers = ["MSFT", "GOOGL"] if ticker != "MSFT" else ["AAPL", "GOOGL"]

        t0 = time.perf_counter()

        try:
            # Execute full StateGraph research pipeline
            state = run_research_pipeline(ticker=ticker, peer_tickers=peer_tickers)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            steps = [
                {
                    "step": 1,
                    "node": "fetch_data",
                    "output": f"Fetched price, fundamentals, news, and peers for {ticker}",
                },
                {
                    "step": 2,
                    "node": "analyze",
                    "output": (
                        f"Analyzed valuation ({state.fundamentals_analysis.get('valuation_summary') if state.fundamentals_analysis else 'Unknown'}), "
                        f"sentiment ({state.sentiment_analysis.get('overall_sentiment') if state.sentiment_analysis else 'Neutral'}), "
                        f"and peer metrics"
                    ),
                },
                {
                    "step": 3,
                    "node": "synthesize",
                    "output": f"Synthesized thesis: {state.thesis[:120] if state.thesis else 'N/A'}...",
                },
                {
                    "step": 4,
                    "node": "generate_report",
                    "output": f"Generated markdown research report ({len(state.report or '')} chars)",
                },
            ]

            final_output = state.report or state.thesis or f"Research report for {ticker} completed."
            tokens_estimate = len(final_output.split())

            return {
                "agent_version": self.version,
                "steps": steps,
                "final_output": final_output,
                "latency_ms": round(elapsed_ms, 2),
                "cost_usd": 0.0,
                "tokens_used": tokens_estimate,
                "metadata": {
                    "ticker": ticker,
                    "peer_tickers": peer_tickers,
                    "overall_sentiment": state.sentiment_analysis.get("overall_sentiment") if state.sentiment_analysis else "Neutral",
                    "valuation_summary": state.fundamentals_analysis.get("valuation_summary") if state.fundamentals_analysis else "Unknown",
                    "winner_overall": state.peer_analysis.get("winner_overall") if state.peer_analysis else ticker,
                },
            }
        except Exception as exc:
            logger.error("Error executing FinSightTargetAgent for %s: %s", ticker, exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "agent_version": self.version,
                "steps": [{"step": 1, "node": "error", "output": str(exc)}],
                "final_output": f"Error executing research pipeline for {ticker}: {exc}",
                "latency_ms": round(elapsed_ms, 2),
                "cost_usd": 0.0,
                "tokens_used": 0,
                "metadata": {"ticker": ticker, "error": str(exc)},
            }
