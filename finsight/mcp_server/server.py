"""Main MCP server entrypoint for FinSight."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from finsight.mcp_server.tools.price import get_stock_price as get_stock_price_tool


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


def main() -> None:
    """Run the FinSight MCP server over stdio transport."""
    logger.info("Starting FinSight MCP server on stdio transport")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
