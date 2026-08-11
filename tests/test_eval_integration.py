"""Unit and integration tests for FinSight evaluation integration with agent-eval-orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finsight.eval.target_agent import FinSightTargetAgent


def test_eval_scenarios_json_valid() -> None:
    """Test finsight/eval/eval_scenarios.json exists and contains 7 scenarios."""
    scenarios_path = Path(__file__).resolve().parents[1] / "finsight" / "eval" / "eval_scenarios.json"
    assert scenarios_path.exists()

    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 7
    for item in data:
        assert "id" in item
        assert "input" in item
        assert "expected_behavior_description" in item


def test_target_agent_invoke_mocked() -> None:
    """Test FinSightTargetAgent invoke method returns valid protocol response payload."""
    mock_state = MagicMock()
    mock_state.ticker = "AAPL"
    mock_state.peer_tickers = ["MSFT", "GOOGL"]
    mock_state.fundamentals_analysis = {"valuation_summary": "Fairly Valued"}
    mock_state.sentiment_analysis = {"overall_sentiment": "Positive"}
    mock_state.peer_analysis = {"winner_overall": "AAPL"}
    mock_state.thesis = "Investment Thesis for AAPL."
    mock_state.report = "# FinSight Stock Research Report: AAPL\n\nTest report content."

    with patch("finsight.eval.target_agent.run_research_pipeline", return_value=mock_state):
        agent = FinSightTargetAgent(version="finsight-test-v1.0")
        res = pytest.Extension.asyncio_loop.run_until_complete(
            agent.invoke({"ticker": "AAPL", "peer_tickers": ["MSFT", "GOOGL"]})
        ) if hasattr(pytest, "Extension") else None

        # Direct async call
        import asyncio
        res = asyncio.run(agent.invoke({"ticker": "AAPL", "peer_tickers": ["MSFT", "GOOGL"]}))

        assert res["agent_version"] == "finsight-test-v1.0"
        assert len(res["steps"]) == 4
        assert "# FinSight Stock Research Report: AAPL" in res["final_output"]
        assert res["metadata"]["ticker"] == "AAPL"
        assert res["metadata"]["overall_sentiment"] == "Positive"
        assert res["metadata"]["valuation_summary"] == "Fairly Valued"


def test_baseline_storage_files_exist() -> None:
    """Test baseline storage output files exist after evaluation run."""
    baseline_path = Path(__file__).resolve().parents[1] / "finsight" / "eval" / "eval_baseline.json"
    verdict_path = Path(__file__).resolve().parents[1] / "finsight" / "eval" / "verdict_report.txt"

    assert baseline_path.exists()
    assert verdict_path.exists()
    assert verdict_path.stat().st_size > 0
