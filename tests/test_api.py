"""Tests for FinSight API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from httpx import ASGITransport
import pytest

from finsight.api.main import app, global_exception_handler, rate_limit_store


@pytest.fixture(autouse=True)
def clear_rate_limit_store():
    rate_limit_store.clear()


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health returns 200 with {"status": "ok"}."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_research_endpoint_valid():
    """POST /research with valid query returns 200."""
    with patch("finsight.api.main.agent.research", new_callable=AsyncMock) as mock_research:
        mock_research.return_value = {
            "query": "Analyse Apple stock",
            "tickers_mentioned": ["AAPL"],
            "tools_called": ["get_stock_price"],
            "price_data": {},
            "fundamentals": {},
            "sentiment": None,
            "filing_summary": None,
            "peer_comparison": None,
            "final_analysis": "Analysis here. Not financial advice.",
            "confidence": "High",
            "disclaimer": "Not financial advice",
            "tokens_used": 100,
            "latency_seconds": 1.0,
            "error": None,
        }
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/research", json={"query": "Analyse Apple stock"})
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["query"] == "Analyse Apple stock"
            assert "request_id" in data
            assert "timestamp" in data


@pytest.mark.asyncio
async def test_research_rejects_short_query():
    """POST /research with "AAPL" (< 5 chars) returns 400."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/research", json={"query": "AAPL"})
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_research_rejects_long_query():
    """POST /research with 501-char string returns 422."""
    long_query = "A" * 501
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/research", json={"query": long_query})
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_price_tool_endpoint():
    """POST /tools/price with {"ticker":"AAPL"} returns 200."""
    with patch("finsight.api.main.get_stock_price", return_value={"price": 150}):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/tools/price", json={"ticker": "AAPL"})
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["price"] == 150


@pytest.mark.asyncio
async def test_fundamentals_tool_endpoint_success():
    """POST /tools/fundamentals returns wrapped tool data."""
    with patch("finsight.api.main.get_fundamentals", return_value={"pe_ratio": 22}):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/tools/fundamentals", json={"ticker": "AAPL"})

    assert response.status_code == 200
    assert response.json()["data"]["pe_ratio"] == 22


@pytest.mark.asyncio
async def test_sentiment_tool_endpoint_success():
    """POST /tools/sentiment returns wrapped tool data."""
    with patch("finsight.api.main.get_news_sentiment", return_value={"overall_sentiment": "Neutral"}):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/tools/sentiment", json={"ticker": "AAPL", "company_name": "Apple"})

    assert response.status_code == 200
    assert response.json()["data"]["overall_sentiment"] == "Neutral"


@pytest.mark.asyncio
async def test_filings_tool_endpoint_success():
    """POST /tools/filings returns wrapped tool data."""
    with patch("finsight.api.main.get_corporate_announcements", return_value={"source": "BSE"}):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/tools/filings", json={"ticker": "TCS.NS"})

    assert response.status_code == 200
    assert response.json()["data"]["source"] == "BSE"


@pytest.mark.asyncio
async def test_peers_tool_endpoint_success():
    """POST /tools/peers returns wrapped tool data."""
    with patch("finsight.api.main.compare_peers", return_value={"winner_overall": "AAPL"}):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/tools/peers", json={"ticker": "AAPL", "peers": ["MSFT", "GOOGL"]})

    assert response.status_code == 200
    assert response.json()["data"]["winner_overall"] == "AAPL"


@pytest.mark.asyncio
async def test_tool_endpoint_failures_are_wrapped():
    """Tool exceptions return success=False payloads without stack traces."""
    endpoint_to_patch = {
        "/tools/price": "get_stock_price",
        "/tools/fundamentals": "get_fundamentals",
        "/tools/sentiment": "get_news_sentiment",
        "/tools/filings": "get_corporate_announcements",
        "/tools/peers": "compare_peers",
    }
    transport = ASGITransport(app=app)

    for endpoint, target in endpoint_to_patch.items():
        rate_limit_store.clear()
        with patch(f"finsight.api.main.{target}", side_effect=Exception("provider failed")):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(endpoint, json={"ticker": "AAPL", "peers": ["MSFT", "GOOGL"]})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "provider failed"
        assert "traceback" not in str(data).lower()


@pytest.mark.asyncio
async def test_request_id_header_present():
    """Every response has X-Request-ID header."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_rate_limit_enforced():
    """Send 11 requests rapidly, 11th returns 429."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        for i in range(10):
            response = await ac.get("/health")
            assert response.status_code == 200
        # 11th should be 429
        response = await ac.get("/health")
        assert response.status_code == 429


@pytest.mark.asyncio
async def test_cors_headers_present():
    """OPTIONS request returns Access-Control-Allow-Origin header."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.options(
            "/health",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" in response.headers


@pytest.mark.asyncio
async def test_invalid_tool_ticker():
    """POST /tools/fundamentals with {"ticker":""} returns 400."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/tools/fundamentals", json={"ticker": ""})
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_invalid_tool_ticker_all_tool_endpoints():
    """All direct tool endpoints reject blank tickers."""
    transport = ASGITransport(app=app)
    endpoints = ["/tools/price", "/tools/sentiment", "/tools/filings", "/tools/peers"]

    for endpoint in endpoints:
        rate_limit_store.clear()
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(endpoint, json={"ticker": "", "peers": ["MSFT", "GOOGL"]})

        assert response.status_code == 400
        assert response.json()["success"] is False


@pytest.mark.asyncio
async def test_error_no_stack_trace():
    """Cause an error, assert "traceback" not in response body."""
    with patch("finsight.api.main.agent.research", side_effect=Exception("Test error")):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/research", json={"query": "Valid query here"})
            assert response.status_code == 200  # Since we catch and return success=False
            data = response.json()
            assert data["success"] is False
            assert "traceback" not in data["error"].lower()


@pytest.mark.asyncio
async def test_global_exception_handler_returns_safe_payload():
    """Unhandled exceptions are converted to generic JSON errors."""
    request = SimpleNamespace(state=SimpleNamespace())

    response = await global_exception_handler(request, RuntimeError("sensitive failure"))

    assert response.status_code == 500
    assert b"Internal server error" in response.body
    assert b"sensitive failure" not in response.body
