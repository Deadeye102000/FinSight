"""Peer comparison tool for FinSight."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from finsight.mcp_server.tools.fundamentals import get_fundamentals
from finsight.mcp_server.tools.price import get_stock_price
from finsight.mcp_server.tools.sentiment import get_news_sentiment
from finsight.mcp_server.utils.validators import validate_peers_list, validate_ticker


logger = logging.getLogger(__name__)

LOWER_IS_BETTER = {"pe_ratio", "pb_ratio", "debt_to_equity"}
HIGHER_IS_BETTER = {"roe", "gross_margin", "net_margin", "rsi"}
RANKING_METRICS = ["pe_ratio", "pb_ratio", "debt_to_equity", "roe", "gross_margin", "net_margin", "rsi"]


class PeerComparisonRequest(BaseModel):
    """Validated request payload for peer comparisons."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., description="Main ticker to compare")
    peers: list[str] = Field(..., description="Two to five peer tickers")
    include_sentiment: bool = Field(default=False, description="Include slower sentiment lookup")

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, value: str) -> str:
        normalized = value.upper()
        if not validate_ticker(normalized):
            raise ValueError("Ticker must be non-empty, 20 chars or fewer, and contain no spaces.")
        return normalized

    @field_validator("peers")
    @classmethod
    def validate_peers_field(cls, value: list[str]) -> list[str]:
        normalized = [peer.upper() for peer in value]
        if not validate_peers_list(normalized):
            raise ValueError("Peers must contain 2 to 5 unique non-empty ticker strings.")
        return normalized


def _empty_result(ticker: str, error: str | None) -> dict[str, Any]:
    """Return a stable peer-comparison payload shape."""
    return {
        "main_ticker": ticker,
        "comparison_date": date.today().isoformat(),
        "metrics_compared": [],
        "rankings": {},
        "winner_overall": "",
        "summary_table": [],
        "relative_valuation": "",
        "error": error,
    }


def _safe_float(value: Any) -> float | None:
    """Coerce a metric to float when present."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_ticker_metrics_sync(ticker: str, include_sentiment: bool) -> dict[str, Any]:
    """Fetch price, fundamentals, and optional sentiment for one ticker."""
    price = get_stock_price(ticker=ticker)
    fundamentals = get_fundamentals(ticker=ticker)

    row = {
        "ticker": ticker,
        "company_name": fundamentals.get("company_name") or "",
        "current_price": price.get("current_price"),
        "currency": price.get("currency") or fundamentals.get("currency") or "",
        "market_cap": fundamentals.get("market_cap"),
        "pe_ratio": fundamentals.get("pe_ratio"),
        "pb_ratio": fundamentals.get("pb_ratio"),
        "debt_to_equity": fundamentals.get("debt_to_equity"),
        "roe": fundamentals.get("roe"),
        "gross_margin": fundamentals.get("gross_margin"),
        "net_margin": fundamentals.get("net_margin"),
        "rsi": price.get("rsi_14"),
        "valuation_summary": fundamentals.get("valuation_summary"),
        "error": fundamentals.get("error") or price.get("error"),
    }

    if include_sentiment:
        company_name = row["company_name"] or ticker
        sentiment = get_news_sentiment(ticker=ticker, company_name=company_name, n=5)
        row["sentiment"] = sentiment.get("overall_sentiment")
        row["sentiment_score"] = sentiment.get("sentiment_score")
        row["sentiment_error"] = sentiment.get("error")

    return row


async def _fetch_ticker_metrics(ticker: str, include_sentiment: bool) -> dict[str, Any]:
    """Fetch one ticker's metrics in a worker thread."""
    return await asyncio.to_thread(_fetch_ticker_metrics_sync, ticker, include_sentiment)


async def _fetch_all_concurrent(tickers: list[str], include_sentiment: bool) -> list[dict[str, Any]]:
    """Fetch all ticker metrics concurrently."""
    return await asyncio.gather(*[_fetch_ticker_metrics(ticker, include_sentiment) for ticker in tickers])


async def _fetch_all_sequential(tickers: list[str], include_sentiment: bool) -> list[dict[str, Any]]:
    """Fetch all ticker metrics sequentially; kept to verify concurrency benefit."""
    results = []
    for ticker in tickers:
        results.append(await _fetch_ticker_metrics(ticker, include_sentiment))
    return results


def _run_async(coro: Any) -> Any:
    """Run an async helper from the synchronous MCP tool boundary."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        result["value"] = asyncio.run(coro)

    import threading

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    return result["value"]


def _ranking_sort_key(metric: str, row: dict[str, Any]) -> tuple[float, float]:
    """Return a sortable key where lower tuple values are better."""
    value = _safe_float(row.get(metric))
    if value is None:
        return (1, float("inf"))

    if metric in LOWER_IS_BETTER:
        return (0, value)
    if metric == "rsi":
        if value < 70:
            return (0, -value)
        return (1, value)
    return (0, -value)


def _build_rankings(summary_table: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Build best-to-worst rankings and composite rank scores."""
    metrics_compared: list[str] = []
    rankings: dict[str, list[str]] = {}
    composite_scores = {row["ticker"]: 0 for row in summary_table}

    for metric in RANKING_METRICS:
        valid_rows = [row for row in summary_table if _safe_float(row.get(metric)) is not None]
        if len(valid_rows) < 2:
            continue

        ranked_rows = sorted(summary_table, key=lambda row: _ranking_sort_key(metric, row))
        rankings[metric] = [row["ticker"] for row in ranked_rows]
        metrics_compared.append(metric)

        for rank, row in enumerate(ranked_rows, start=1):
            composite_scores[row["ticker"]] += rank

    return metrics_compared, rankings, composite_scores


def _winner_from_rankings(rankings: dict[str, list[str]], composite_scores: dict[str, int]) -> str:
    """Pick the overall winner by top-3 count, then composite score."""
    top_three_counts = {ticker: 0 for ticker in composite_scores}
    for ranked_tickers in rankings.values():
        for ticker in ranked_tickers[:3]:
            top_three_counts[ticker] += 1

    return min(
        composite_scores,
        key=lambda ticker: (-top_three_counts[ticker], composite_scores[ticker], ticker),
    )


def _relative_valuation(main_ticker: str, peers: list[str], summary_table: list[dict[str, Any]]) -> str:
    """Describe whether the main ticker is cheap or expensive versus peers."""
    rows_by_ticker = {row["ticker"]: row for row in summary_table}
    main_pe = _safe_float(rows_by_ticker.get(main_ticker, {}).get("pe_ratio"))
    peer_pe_values = [_safe_float(rows_by_ticker.get(peer, {}).get("pe_ratio")) for peer in peers]
    peer_pe_values = [value for value in peer_pe_values if value is not None]

    if main_pe is None or not peer_pe_values:
        return f"{main_ticker} cannot be valued against peers because comparable P/E data is incomplete."

    peer_average = sum(peer_pe_values) / len(peer_pe_values)
    if peer_average == 0:
        return f"{main_ticker} cannot be valued against peers because the peer average P/E is zero."

    gap_pct = ((main_pe - peer_average) / peer_average) * 100
    if gap_pct <= -10:
        stance = "cheap"
    elif gap_pct >= 10:
        stance = "expensive"
    else:
        stance = "roughly in line"

    return (
        f"{main_ticker} is {stance} versus peers on P/E: "
        f"{main_pe:.2f} compared with a peer average of {peer_average:.2f}."
    )


def compare_peers(ticker: str, peers: list[str], include_sentiment: bool = False) -> dict[str, Any]:
    """Compare a main ticker against peers using price and fundamentals tools."""
    logger.info("Processing peer comparison request for ticker=%s peers=%s", ticker, peers)

    try:
        request = PeerComparisonRequest(ticker=ticker, peers=peers, include_sentiment=include_sentiment)
    except ValidationError as exc:
        return _empty_result(ticker, exc.errors()[0]["msg"])

    tickers = [request.ticker, *request.peers]
    if len(set(tickers)) != len(tickers):
        return _empty_result(request.ticker, "Main ticker must not also appear in peers.")

    try:
        summary_table = _run_async(_fetch_all_concurrent(tickers, request.include_sentiment))
        metrics_compared, rankings, composite_scores = _build_rankings(summary_table)
        winner = _winner_from_rankings(rankings, composite_scores) if rankings else request.ticker

        return {
            "main_ticker": request.ticker,
            "comparison_date": date.today().isoformat(),
            "metrics_compared": metrics_compared,
            "rankings": rankings,
            "winner_overall": winner,
            "summary_table": summary_table,
            "relative_valuation": _relative_valuation(request.ticker, request.peers, summary_table),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive around provider/tool internals
        logger.exception("Failed to compare peers for %s", request.ticker)
        return _empty_result(request.ticker, str(exc))
