"""Tests for FinSight MCP server wrapper functions."""

from unittest.mock import patch

from finsight.mcp_server import server


def test_mcp_price_wrapper_delegates() -> None:
    with patch.object(server, "get_stock_price_tool", return_value={"ticker": "AAPL"}) as mock_tool:
        result = server.get_stock_price("AAPL", period="1mo", interval="1d")

    assert result == {"ticker": "AAPL"}
    mock_tool.assert_called_once_with(ticker="AAPL", period="1mo", interval="1d")


def test_mcp_fundamentals_wrapper_delegates() -> None:
    with patch.object(server, "get_fundamentals_tool", return_value={"pe_ratio": 20}) as mock_tool:
        result = server.get_fundamentals("AAPL")

    assert result == {"pe_ratio": 20}
    mock_tool.assert_called_once_with(ticker="AAPL")


def test_mcp_sentiment_wrapper_delegates() -> None:
    with patch.object(server, "get_news_sentiment_tool", return_value={"overall_sentiment": "Positive"}) as mock_tool:
        result = server.get_news_sentiment("AAPL", "Apple", n=3)

    assert result == {"overall_sentiment": "Positive"}
    mock_tool.assert_called_once_with(ticker="AAPL", company_name="Apple", n=3)


def test_mcp_announcements_wrapper_delegates() -> None:
    with patch.object(server, "get_corporate_announcements_tool", return_value={"source": "BSE"}) as mock_tool:
        result = server.get_corporate_announcements("TCS.NS", announcement_type="results", n=4)

    assert result == {"source": "BSE"}
    mock_tool.assert_called_once_with(ticker="TCS.NS", announcement_type="results", n=4)


def test_mcp_peers_wrapper_delegates() -> None:
    with patch.object(server, "compare_peers_tool", return_value={"winner_overall": "AAPL"}) as mock_tool:
        result = server.compare_peers("AAPL", ["MSFT", "GOOGL"], include_sentiment=True)

    assert result == {"winner_overall": "AAPL"}
    mock_tool.assert_called_once_with(ticker="AAPL", peers=["MSFT", "GOOGL"], include_sentiment=True)


def test_mcp_main_runs_stdio_transport() -> None:
    with patch.object(server.server, "run") as mock_run:
        server.main()

    mock_run.assert_called_once_with(transport="stdio")
