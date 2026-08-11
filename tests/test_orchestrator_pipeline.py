"""Unit and integration tests for FinSight LangGraph Orchestration Layer and Typer CLI."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from finsight.orchestrator.cli import app
from finsight.orchestrator.pipeline import build_research_graph, run_research_pipeline
from finsight.orchestrator.state import ResearchState

runner = CliRunner()


def test_research_state_schema() -> None:
    """Test ResearchState Pydantic v2 model creation and default fields."""
    state = ResearchState(ticker="AAPL", peer_tickers=["MSFT", "GOOGL"])
    assert state.ticker == "AAPL"
    assert state.peer_tickers == ["MSFT", "GOOGL"]
    assert state.raw_price is None
    assert state.thesis is None
    assert state.errors == []


def test_build_research_graph_compiles() -> None:
    """Test building and compiling the LangGraph StateGraph."""
    graph = build_research_graph()
    assert graph is not None


def test_run_research_pipeline_end_to_end() -> None:
    """Test executing the research pipeline end-to-end."""
    state = run_research_pipeline(ticker="AAPL", peer_tickers=["MSFT"])

    assert state.ticker == "AAPL"
    assert state.peer_tickers == ["MSFT"]
    assert state.raw_price is not None
    assert state.raw_fundamentals is not None
    assert state.raw_news is not None
    assert state.raw_peers is not None

    assert state.fundamentals_analysis is not None
    assert state.sentiment_analysis is not None
    assert state.peer_analysis is not None

    assert state.thesis is not None
    assert len(state.thesis) > 0
    assert state.report is not None
    assert "FinSight Stock Research Report: AAPL" in state.report


def test_cli_execution() -> None:
    """Test running the Typer CLI entrypoint."""
    result = runner.invoke(app, ["AAPL", "-p", "MSFT"])
    assert result.exit_code == 0
    assert "Launching FinSight LangGraph Pipeline for AAPL" in result.output
    assert "Research Pipeline State Output" in result.output


def test_sentiment_analysis_node_finbert_structure() -> None:
    """Test sentiment_analysis_node populates per-headline details and scores."""
    from finsight.orchestrator.pipeline import sentiment_analysis_node

    state = ResearchState(
        ticker="AAPL",
        raw_news={
            "headlines": [
                {"title": "Apple reports record profit and revenue surge", "publisher": "Bloomberg", "published_at": "2026-08-10"},
                {"title": "Tech stocks face pressure over rising costs", "publisher": "Reuters", "published_at": "2026-08-10"},
            ]
        },
    )
    res = sentiment_analysis_node(state)
    sent = res["sentiment_analysis"]

    assert sent["headline_count"] == 2
    assert sent["mode"] in ["finbert", "lexicon_fallback"]
    assert len(sent["per_headline_details"]) == 2
    assert "sentiment" in sent["per_headline_details"][0]
    assert "confidence" in sent["per_headline_details"][0]
    assert "score" in sent["per_headline_details"][0]


def test_sentiment_analysis_node_lexicon_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test sentiment_analysis_node falls back to lexicon mode if FinBERT fails."""
    from finsight.orchestrator import pipeline

    def mock_broken_loader():
        raise RuntimeError("GPU OOM error loading FinBERT")

    monkeypatch.setattr(pipeline, "load_finbert_model", mock_broken_loader)

    state = ResearchState(
        ticker="AAPL",
        raw_news={
            "headlines": [
                {"title": "Company reports record profit beat", "publisher": "News"},
                {"title": "Shares drop after weak guidance warning", "publisher": "News"},
            ]
        },
    )
    res = pipeline.sentiment_analysis_node(state)
    sent = res["sentiment_analysis"]

    assert sent["mode"] == "lexicon_fallback"
    assert sent["headline_count"] == 2
    assert len(sent["per_headline_details"]) == 2

