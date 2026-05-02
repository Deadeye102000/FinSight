"""Main MCP server entrypoint for FinSight."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from finsight.mcp_server.tools.announcements import (
    get_corporate_announcements as get_corporate_announcements_tool,
)
from finsight.mcp_server.tools.fundamentals import get_fundamentals as get_fundamentals_tool
from finsight.mcp_server.tools.peers import compare_peers as compare_peers_tool
from finsight.mcp_server.tools.price import get_stock_price as get_stock_price_tool
from finsight.mcp_server.tools.sentiment import get_news_sentiment as get_news_sentiment_tool


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)
server = FastMCP("FinSight")


@server.tool()
def get_stock_price(ticker: str, period: str = "3mo", interval: str = "1d") -> dict[str, Any]:
    """Return recent market data and technical indicators for a stock ticker."""
    logger.info("MCP tool invocation received for %s", ticker)
    return get_stock_price_tool(ticker=ticker, period=period, interval=interval)


@server.tool()
def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Return valuation and financial-ratio fundamentals for a stock ticker."""
    logger.info("MCP fundamentals invocation received for %s", ticker)
    return get_fundamentals_tool(ticker=ticker)


@server.tool()
def get_news_sentiment(ticker: str, company_name: str, n: int = 10) -> dict[str, Any]:
    """Return recent headline sentiment for a stock ticker using local FinBERT."""
    logger.info("MCP news sentiment invocation received for %s", ticker)
    return get_news_sentiment_tool(ticker=ticker, company_name=company_name, n=n)


@server.tool()
def get_corporate_announcements(ticker: str, announcement_type: str = "all", n: int = 10) -> dict[str, Any]:
    """Return recent BSE/NSE corporate announcements summarized for investors."""
    logger.info("MCP corporate announcements invocation received for %s", ticker)
    return get_corporate_announcements_tool(
        ticker=ticker,
        announcement_type=announcement_type,
        n=n,
    )


@server.tool()
def compare_peers(ticker: str, peers: list[str], include_sentiment: bool = False) -> dict[str, Any]:
    """Return a structured peer comparison across valuation and quality metrics."""
    logger.info("MCP peer comparison invocation received for %s", ticker)
    return compare_peers_tool(ticker=ticker, peers=peers, include_sentiment=include_sentiment)


def main() -> None:
    """Run the FinSight MCP server over stdio transport."""
    logger.info("Starting FinSight MCP server on stdio transport")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
