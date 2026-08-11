"""Unit and integration tests for the 3 new MCP tools."""

from __future__ import annotations

import pytest

from finsight.mcp_server.tools.filings import get_filing_text
from finsight.mcp_server.tools.peers import get_peer_comparison
from finsight.mcp_server.tools.sentiment import get_news_sentiment


def test_get_news_sentiment_raw_headlines() -> None:
    """Test get_news_sentiment returns raw headlines + timestamps without scoring sentiment."""
    result = get_news_sentiment(ticker="AAPL", days_back=7)

    assert result["ticker"] == "AAPL"
    assert result["days_back"] == 7
    assert result["error"] is None
    assert isinstance(result["headlines"], list)
    assert result["headline_count"] == len(result["headlines"])

    if result["headlines"]:
        headline = result["headlines"][0]
        assert "title" in headline
        assert "publisher" in headline
        assert "published_at" in headline
        assert "url" in headline


def test_get_news_sentiment_invalid_ticker() -> None:
    """Test get_news_sentiment handles invalid tickers gracefully."""
    result = get_news_sentiment(ticker="", days_back=7)
    assert result["error"] is not None


def test_get_peer_comparison_returns_table() -> None:
    """Test get_peer_comparison returns key ratios and comparable table."""
    result = get_peer_comparison(ticker="AAPL", peer_tickers=["MSFT", "GOOGL"])

    assert result["main_ticker"] == "AAPL"
    assert result["peer_tickers"] == ["MSFT", "GOOGL"]
    assert result["error"] is None
    assert isinstance(result["comparison_table"], list)
    assert len(result["comparison_table"]) == 3

    for row in result["comparison_table"]:
        assert "ticker" in row
        assert "company_name" in row
        assert "reporting_currency" in row
        assert row["normalized_currency"] == "USD"
        assert "pe_ratio" in row
        assert "market_cap" in row
        assert "revenue_growth" in row


def test_get_peer_comparison_invalid_peers() -> None:
    """Test get_peer_comparison rejects invalid peer lists."""
    result = get_peer_comparison(ticker="AAPL", peer_tickers=["AAPL"])
    assert result["error"] is not None


def test_get_filing_text_indian_ticker() -> None:
    """Test get_filing_text for Indian equity symbol (TCS.NS)."""
    result = get_filing_text(ticker="TCS.NS", filing_type="annual_report")

    assert result["ticker"] == "TCS.NS"
    assert result["filing_type"] == "annual_report"
    assert result["reporting_currency"] == "INR"
    assert result["normalized_currency"] == "USD"
    assert result["error"] is None
    assert "Filing Title:" in result["filing_text"] or result["company_name"] != ""


def test_get_filing_text_us_ticker() -> None:
    """Test get_filing_text fallback for US equity symbol (AAPL)."""
    result = get_filing_text(ticker="AAPL", filing_type="annual_report")

    assert result["ticker"] == "AAPL"
    assert result["filing_type"] == "annual_report"
    assert result["reporting_currency"] == "USD"
    assert result["normalized_currency"] == "USD"
    assert result["error"] is None
    assert len(result["filing_text"]) > 0


def test_get_filing_text_invalid_ticker() -> None:
    """Test get_filing_text handles invalid tickers gracefully."""
    result = get_filing_text(ticker="", filing_type="annual_report")
    assert result["error"] is not None
