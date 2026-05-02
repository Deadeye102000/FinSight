"""Claude orchestration for FinSight."""

import asyncio
import json
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel


class ResearchReport(BaseModel):
    query: str
    tickers_mentioned: List[str]
    tools_called: List[str]
    price_data: Optional[Dict[str, Any]] = None
    fundamentals: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    filing_summary: Optional[Dict[str, Any]] = None
    peer_comparison: Optional[Dict[str, Any]] = None
    final_analysis: str
    confidence: str  # "High" | "Medium" | "Low"
    disclaimer: str  # always include "Not financial advice"
    tokens_used: int
    latency_seconds: float
    error: Optional[str] = None


class FinSightAgent:
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.client = Anthropic()  # Assuming API key is set in env
        self.system_prompt = """You are FinSight, an AI financial research analyst. You have access to these tools:
- get_stock_price: technical data, price action, RSI, MACD
- get_fundamentals: valuation ratios, margins, analyst ratings  
- get_news_sentiment: news sentiment from recent headlines
- get_sec_filing_summary: latest 10-K/10-Q filing analysis
- compare_peers: peer group comparison and relative valuation

When a user asks about a stock:
1. Always call get_stock_price and get_fundamentals first
2. Call get_news_sentiment if the user asks about recent events or news
3. Call get_sec_filing_summary if the user asks about company strategy, risks, or annual report
4. Call compare_peers if the user asks how the stock compares to competitors
5. After gathering data, write a structured research report in markdown with:
   - Executive Summary (2-3 sentences)
   - Technical Picture (price action, RSI, MACD interpretation)
   - Fundamental Analysis (valuation verdict with specific numbers)
   - Key Risks (bullet list)
   - Conclusion with a clear stance (Bullish / Neutral / Bearish) and reasoning

Always end with: "⚠️ This is not financial advice. Do your own research."
Be specific — cite actual numbers from the tools, never make up data."""

    async def research(self, query: str) -> ResearchReport:
        start_time = time.time()
        tickers = self._extract_tickers(query)
        tools_called = []
        price_data = None
        fundamentals = None
        sentiment = None
        filing_summary = None
        peer_comparison = None
        error = None
        final_analysis = ""
        confidence = "Medium"
        tokens_used = 0

        try:
            result = await self._run_with_mcp(query)
            # Parse the result
            tools_called = result.get("tools_called", [])
            price_data = result.get("price_data")
            fundamentals = result.get("fundamentals")
            sentiment = result.get("sentiment")
            filing_summary = result.get("filing_summary")
            peer_comparison = result.get("peer_comparison")
            final_analysis = result.get("final_analysis", "")
            tokens_used = result.get("tokens_used", 0)
            confidence = result.get("confidence", "Medium")
        except Exception as e:
            error = str(e)
            final_analysis = f"Error occurred: {error}"

        latency = time.time() - start_time
        disclaimer = "Not financial advice"

        return ResearchReport(
            query=query,
            tickers_mentioned=tickers,
            tools_called=tools_called,
            price_data=price_data,
            fundamentals=fundamentals,
            sentiment=sentiment,
            filing_summary=filing_summary,
            peer_comparison=peer_comparison,
            final_analysis=final_analysis,
            confidence=confidence,
            disclaimer=disclaimer,
            tokens_used=tokens_used,
            latency_seconds=latency,
            error=error,
        )

    def _extract_tickers(self, query: str) -> List[str]:
        # Simple extraction: look for uppercase words that look like tickers
        # For Apple, assume AAPL, etc. For now, hardcode common ones
        ticker_map = {
            "apple": "AAPL",
            "tesla": "TSLA",
            "ford": "F",
            "gm": "GM",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "amazon": "AMZN",
        }
        tickers = []
        words = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        for word in words:
            if len(word) >= 2:  # Assume tickers are 2-5 letters
                tickers.append(word)
        # Also check for company names
        for company, ticker in ticker_map.items():
            if company.lower() in query.lower():
                tickers.append(ticker)
        return list(set(tickers))  # unique

    async def _run_with_mcp(self, query: str) -> Dict[str, Any]:
        # Start MCP server process
        server_params = StdioServerParameters(
            command="python",
            args=[self.mcp_server_path],
            env=None,
        )

        tools_called = []
        tool_results = {}

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Get available tools
                tools_response = await session.list_tools()
                tools = tools_response.tools

                # Convert MCP tools to Anthropic format
                anthropic_tools = []
                for tool in tools:
                    anthropic_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    })

                messages = [{"role": "user", "content": query}]

                # Tool calling loop
                total_tokens = 0
                final_analysis = ""
                max_iterations = 5
                for _ in range(max_iterations):
                    response = self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4096,
                        system=self.system_prompt,
                        messages=messages,
                        tools=anthropic_tools,
                    )

                    total_tokens += response.usage.input_tokens + response.usage.output_tokens

                    if response.stop_reason == "tool_use":
                        tool_calls = []
                        content = []
                        for block in response.content:
                            if block.type == "text":
                                content.append(block.text)
                            elif block.type == "tool_use":
                                tool_calls.append(block)

                        messages.append({"role": "assistant", "content": response.content})

                        for tool_call in tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.input
                            tool_call_id = tool_call.id

                            tools_called.append(tool_name)

                            # Call the tool via MCP
                            result = await session.call_tool(tool_name, tool_args)

                            # Store result
                            if result.content:
                                result_text = result.content[0].text
                                try:
                                    parsed = json.loads(result_text)
                                except json.JSONDecodeError:
                                    parsed = {"error": "Invalid JSON", "raw": result_text}
                            else:
                                parsed = None

                            if tool_name == "get_stock_price":
                                tool_results["price_data"] = parsed
                            elif tool_name == "get_fundamentals":
                                tool_results["fundamentals"] = parsed
                            elif tool_name == "get_news_sentiment":
                                tool_results["sentiment"] = parsed
                            elif tool_name == "get_corporate_announcements":
                                tool_results["filing_summary"] = parsed
                            elif tool_name == "compare_peers":
                                tool_results["peer_comparison"] = parsed

                            messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_call_id": tool_call_id,
                                        "content": result_text if result.content else "",
                                    }
                                ],
                            })
                    else:
                        # Final response
                        for block in response.content:
                            if block.type == "text":
                                final_analysis += block.text
                        break

        return {
            "tools_called": tools_called,
            "price_data": tool_results.get("price_data"),
            "fundamentals": tool_results.get("fundamentals"),
            "sentiment": tool_results.get("sentiment"),
            "filing_summary": tool_results.get("filing_summary"),
            "peer_comparison": tool_results.get("peer_comparison"),
            "final_analysis": final_analysis,
            "tokens_used": tokens_used,
            "confidence": "High" if len(tools_called) >= 2 else "Medium",
        }
