"""Tests for FinSight API."""

from unittest.mock import AsyncMock, patch

import httpx
from httpx import ASGITransport
import pytest

from finsight.api.main import app, rate_limit_store


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
