"""Tests for FinSightAgent orchestrator."""

import pytest
from types import SimpleNamespace
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


def test_openai_provider_initialises(monkeypatch):
    """FinSightAgent can be configured to use OpenAI instead of Anthropic."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = FinSightAgent("/path/to/server.py", llm_provider="openai", openai_model="gpt-4.1")

    assert agent.llm_provider == "openai"
    assert agent.openai_model == "gpt-4.1"
    assert agent.client is agent.openai_client


def test_invalid_llm_provider_rejected():
    """Unsupported provider names fail fast."""
    with pytest.raises(ValueError, match="FINSIGHT_LLM_PROVIDER"):
        FinSightAgent("/path/to/server.py", llm_provider="gemini")


def test_mcp_tool_to_openai_tool(monkeypatch):
    """MCP tool schemas are converted into OpenAI function tools."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = FinSightAgent("/path/to/server.py", llm_provider="openai")
    tool = SimpleNamespace(
        name="get_stock_price",
        description="Fetch price",
        inputSchema={"type": "object", "properties": {"ticker": {"type": "string"}}},
    )

    openai_tool = agent._mcp_tool_to_openai_tool(tool)

    assert openai_tool == {
        "type": "function",
        "name": "get_stock_price",
        "description": "Fetch price",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
    }


def test_openai_helpers_parse_usage_and_text(monkeypatch):
    """OpenAI response helpers tolerate SDK response shapes."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = FinSightAgent("/path/to/server.py", llm_provider="openai")
    response = SimpleNamespace(
        output_text="Report text",
        usage=SimpleNamespace(input_tokens=10, output_tokens=15, total_tokens=None),
        output=[],
    )

    assert agent._openai_response_text(response) == "Report text"
    assert agent._openai_usage_tokens(response) == 25


def test_store_tool_result_maps_known_tools(monkeypatch):
    """Tool outputs are stored under ResearchReport field names."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = FinSightAgent("/path/to/server.py", llm_provider="openai")
    results = {}

    agent._store_tool_result(results, "get_stock_price", {"price": 1})
    agent._store_tool_result(results, "get_fundamentals", {"pe": 2})
    agent._store_tool_result(results, "get_news_sentiment", {"sentiment": "Neutral"})
    agent._store_tool_result(results, "get_corporate_announcements", {"summary": "ok"})
    agent._store_tool_result(results, "compare_peers", {"winner": "AAPL"})

    assert results == {
        "price_data": {"price": 1},
        "fundamentals": {"pe": 2},
        "sentiment": {"sentiment": "Neutral"},
        "filing_summary": {"summary": "ok"},
        "peer_comparison": {"winner": "AAPL"},
    }
