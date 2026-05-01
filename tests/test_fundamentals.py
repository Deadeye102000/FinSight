"""Tests for the FinSight fundamentals tool."""

from __future__ import annotations

from finsight.mcp_server.tools.fundamentals import get_fundamentals


def test_valid_ticker_returns_company_name() -> None:
    result = get_fundamentals("MSFT")
    assert result["error"] is None
    assert result["company_name"] != ""


def test_market_cap_positive() -> None:
    result = get_fundamentals("AAPL")
    assert result["error"] is None
    assert result["market_cap"] > 0


def test_none_fields_dont_crash() -> None:
    result = get_fundamentals("AAME")
    assert isinstance(result, dict)


def test_valuation_summary_valid_values() -> None:
    result = get_fundamentals("MSFT")
    assert result["valuation_summary"] in [
        "Undervalued",
        "Fairly Valued",
        "Overvalued",
        "Insufficient data",
    ]


def test_margins_are_percentages() -> None:
    result = get_fundamentals("AAPL")
    if result["gross_margin"] is not None:
        assert -100 < result["gross_margin"] < 100


def test_invalid_ticker_returns_error() -> None:
    result = get_fundamentals("FAKEXYZ123")
    assert result["error"] is not None


def test_debt_to_equity_non_negative() -> None:
    result = get_fundamentals("AAPL")
    if result["debt_to_equity"] is not None:
        assert result["debt_to_equity"] >= 0


def test_sector_and_industry_present() -> None:
    result = get_fundamentals("GOOGL")
    assert isinstance(result["sector"], str)
    assert isinstance(result["industry"], str)


def test_indian_stock_fundamentals() -> None:
    result = get_fundamentals("TCS.NS")
    assert isinstance(result, dict)


def test_analyst_rating_valid_or_none() -> None:
    result = get_fundamentals("MSFT")
    if result["analyst_rating"] is not None:
        assert result["analyst_rating"] in ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
