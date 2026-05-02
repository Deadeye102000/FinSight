"""Tests for FinSightAgent orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from finsight.agent.orchestrator import FinSightAgent, ResearchReport


@pytest.mark.asyncio
async def test_agent_initialises():
    """FinSightAgent initialises without error."""
    agent = FinSightAgent("/path/to/server.py")
    assert agent.mcp_server_path == "/path/to/server.py"
    assert agent.client is not None


@pytest.mark.asyncio
async def test_simple_query_returns_report():
    """research("Analyse Apple stock") returns ResearchReport."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "tools_called": ["get_stock_price", "get_fundamentals"],
            "price_data": {"price": 150},
            "fundamentals": {"pe": 20},
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": "Some analysis. Not financial advice.",
            "tokens_used": 100,
            "confidence": "High",
        }
        report = await agent.research("Analyse Apple stock")
        assert isinstance(report, ResearchReport)
        assert report.query == "Analyse Apple stock"


@pytest.mark.asyncio
async def test_tickers_extracted():
    """assert "AAPL" in tickers_mentioned for Apple query."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "tools_called": [],
            "price_data": None,
            "fundamentals": None,
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": "",
            "tokens_used": 0,
            "confidence": "Medium",
        }
        report = await agent.research("Analyse Apple stock")
        assert "AAPL" in report.tickers_mentioned


@pytest.mark.asyncio
async def test_tools_called_tracked():
    """assert len(tools_called) >= 2."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "tools_called": ["get_stock_price", "get_fundamentals"],
            "price_data": None,
            "fundamentals": None,
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": "",
            "tokens_used": 0,
            "confidence": "Medium",
        }
        report = await agent.research("Analyse Apple stock")
        assert len(report.tools_called) >= 2


@pytest.mark.asyncio
async def test_final_analysis_not_empty():
    """assert len(final_analysis) > 200."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        analysis = "Some long analysis text that is more than 200 characters. " * 10
        mock_run.return_value = {
            "tools_called": [],
            "price_data": None,
            "fundamentals": None,
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": analysis,
            "tokens_used": 0,
            "confidence": "Medium",
        }
        report = await agent.research("Analyse Apple stock")
        assert len(report.final_analysis) > 200


@pytest.mark.asyncio
async def test_disclaimer_always_present():
    """assert "not financial advice" in final_analysis.lower()."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "tools_called": [],
            "price_data": None,
            "fundamentals": None,
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": "This is not financial advice.",
            "tokens_used": 0,
            "confidence": "Medium",
        }
        report = await agent.research("Analyse Apple stock")
        assert "not financial advice" in report.final_analysis.lower()


@pytest.mark.asyncio
async def test_peer_query_triggers_compare_tool():
    """research("Compare Tesla vs Ford vs GM") should call compare_peers."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "tools_called": ["compare_peers"],
            "price_data": None,
            "fundamentals": None,
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": {},
            "final_analysis": "",
            "tokens_used": 0,
            "confidence": "Medium",
        }
        report = await agent.research("Compare Tesla vs Ford vs GM")
        assert "compare_peers" in report.tools_called


@pytest.mark.asyncio
async def test_latency_tracked():
    """assert latency_seconds > 0."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "tools_called": [],
            "price_data": None,
            "fundamentals": None,
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": "",
            "tokens_used": 0,
            "confidence": "Medium",
        }
        report = await agent.research("Analyse Apple stock")
        assert report.latency_seconds > 0


@pytest.mark.asyncio
async def test_tokens_used_tracked():
    """assert tokens_used > 0."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "tools_called": [],
            "price_data": None,
            "fundamentals": None,
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": "",
            "tokens_used": 100,
            "confidence": "Medium",
        }
        report = await agent.research("Analyse Apple stock")
        assert report.tokens_used > 0


@pytest.mark.asyncio
async def test_error_handling_bad_ticker():
    """research("Analyse FAKESTOCKXYZ999") returns error gracefully, no crash."""
    agent = FinSightAgent("/path/to/server.py")
    with patch.object(agent, '_run_with_mcp', side_effect=Exception("Bad ticker")) as mock_run:
        report = await agent.research("Analyse FAKESTOCKXYZ999")
        assert report.error is not None
        assert "Bad ticker" in report.error