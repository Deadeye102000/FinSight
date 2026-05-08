"""LLM orchestration for FinSight."""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
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
    def __init__(
        self,
        mcp_server_path: str,
        llm_provider: str | None = None,
        anthropic_model: str | None = None,
        openai_model: str | None = None,
    ):
        load_dotenv()
        self.mcp_server_path = mcp_server_path
        self.llm_provider = (llm_provider or os.getenv("FINSIGHT_LLM_PROVIDER") or "anthropic").strip().lower()
        self.anthropic_model = anthropic_model or os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL") or "gpt-5"

        if self.llm_provider not in {"anthropic", "openai"}:
            raise ValueError("FINSIGHT_LLM_PROVIDER must be either 'anthropic' or 'openai'.")

        self.anthropic_client = Anthropic() if self.llm_provider == "anthropic" else None
        self.openai_client = OpenAI() if self.llm_provider == "openai" else None
        self.client = self.anthropic_client or self.openai_client
        self.system_prompt = """You are FinSight, an AI financial research analyst. You have access to these tools:
- get_stock_price: technical data, price action, RSI, MACD
- get_fundamentals: valuation ratios, margins, analyst ratings  
- get_news_sentiment: news sentiment from recent headlines
- get_corporate_announcements: latest exchange announcements and filing-style company updates
- compare_peers: peer group comparison and relative valuation

When a user asks about a stock:
1. Always call get_stock_price and get_fundamentals first
2. Call get_news_sentiment if the user asks about recent events or news
3. Call get_corporate_announcements if the user asks about company strategy, risks, filings, or recent corporate updates
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
        if self.llm_provider == "openai":
            return await self._run_with_mcp_openai(query)
        return await self._run_with_mcp_anthropic(query)

    async def _run_with_mcp_anthropic(self, query: str) -> Dict[str, Any]:
        # Start MCP server process
        server_params = StdioServerParameters(
            command=sys.executable,
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
                    response = self.anthropic_client.messages.create(
                        model=self.anthropic_model,
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
            "tokens_used": total_tokens,
            "confidence": "High" if len(tools_called) >= 2 else "Medium",
        }

    async def _run_with_mcp_openai(self, query: str) -> Dict[str, Any]:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.mcp_server_path],
            env=None,
        )

        tools_called: list[str] = []
        tool_results: dict[str, Any] = {}

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_response = await session.list_tools()
                tools = tools_response.tools
                openai_tools = [self._mcp_tool_to_openai_tool(tool) for tool in tools]

                messages: list[Any] = [{"role": "user", "content": query}]
                total_tokens = 0
                final_analysis = ""
                max_iterations = 5

                for _ in range(max_iterations):
                    response = self.openai_client.responses.create(
                        model=self.openai_model,
                        instructions=self.system_prompt,
                        input=messages,
                        tools=openai_tools,
                    )
                    total_tokens += self._openai_usage_tokens(response)

                    function_calls = [
                        item for item in getattr(response, "output", []) if getattr(item, "type", None) == "function_call"
                    ]
                    if not function_calls:
                        final_analysis = self._openai_response_text(response)
                        break

                    messages.extend(getattr(response, "output", []))
                    for tool_call in function_calls:
                        tool_name = tool_call.name
                        tool_args = json.loads(tool_call.arguments or "{}")

                        tools_called.append(tool_name)
                        result = await session.call_tool(tool_name, tool_args)
                        result_text = result.content[0].text if result.content else ""
                        parsed = self._parse_tool_result(result_text)
                        self._store_tool_result(tool_results, tool_name, parsed)

                        messages.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.call_id,
                                "output": result_text,
                            }
                        )

        return {
            "tools_called": tools_called,
            "price_data": tool_results.get("price_data"),
            "fundamentals": tool_results.get("fundamentals"),
            "sentiment": tool_results.get("sentiment"),
            "filing_summary": tool_results.get("filing_summary"),
            "peer_comparison": tool_results.get("peer_comparison"),
            "final_analysis": final_analysis,
            "tokens_used": total_tokens,
            "confidence": "High" if len(tools_called) >= 2 else "Medium",
        }

    def _mcp_tool_to_openai_tool(self, tool: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        }

    def _parse_tool_result(self, result_text: str) -> Any:
        if not result_text:
            return None
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw": result_text}

    def _store_tool_result(self, tool_results: dict[str, Any], tool_name: str, parsed: Any) -> None:
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

    def _openai_usage_tokens(self, response: Any) -> int:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            return int(total)
        return int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "output_tokens", 0) or 0)

    def _openai_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        chunks: list[str] = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return "".join(chunks)
