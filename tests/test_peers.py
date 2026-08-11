"""Tests for the FinSight peer comparison tool."""

from __future__ import annotations

import time

from finsight.mcp_server.tools import peers as peers_module
from finsight.mcp_server.tools.peers import compare_peers


def test_basic_comparison() -> None:
    result = compare_peers("AAPL", ["MSFT", "GOOGL"])
    assert result["error"] is None


def test_summary_table_has_all_tickers() -> None:
    result = compare_peers("AAPL", ["MSFT", "GOOGL"])
    assert len(result["summary_table"]) == 3


def test_winner_is_valid_ticker() -> None:
    ticker = "AAPL"
    peers = ["MSFT", "GOOGL"]
    result = compare_peers(ticker, peers)
    assert result["winner_overall"] in [ticker, *peers]


def test_rankings_keys_present() -> None:
    result = compare_peers("AAPL", ["MSFT", "GOOGL"])
    assert "pe_ratio" in result["rankings"]
    assert "roe" in result["rankings"]


def test_too_few_peers() -> None:
    result = compare_peers("AAPL", ["MSFT"])
    assert result["error"] is not None


def test_too_many_peers() -> None:
    result = compare_peers("AAPL", ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"])
    assert result["error"] is not None


def test_duplicate_peers_rejected() -> None:
    result = compare_peers("AAPL", ["MSFT", "MSFT"])
    assert result["error"] is not None


def test_concurrent_faster_than_sequential(monkeypatch) -> None:
    def fake_fetch(ticker: str, include_sentiment: bool) -> dict:
        time.sleep(0.15)
        return {
            "ticker": ticker,
            "pe_ratio": 10.0,
            "pb_ratio": 2.0,
            "debt_to_equity": 1.0,
            "roe": 20.0,
            "gross_margin": 50.0,
            "net_margin": 15.0,
            "rsi": 55.0,
            "error": None,
        }

    tickers = ["AAPL", "MSFT", "GOOGL"]
    monkeypatch.setattr(peers_module, "_fetch_ticker_metrics_sync", fake_fetch)

    sequential_start = time.perf_counter()
    peers_module._run_async(peers_module._fetch_all_sequential(tickers, False))
    sequential_elapsed = time.perf_counter() - sequential_start

    concurrent_start = time.perf_counter()
    peers_module._run_async(peers_module._fetch_all_concurrent(tickers, False))
    concurrent_elapsed = time.perf_counter() - concurrent_start

    assert sequential_elapsed / concurrent_elapsed >= 1.5


def test_relative_valuation_not_empty() -> None:
    result = compare_peers("AAPL", ["MSFT", "GOOGL"])
    assert len(result["relative_valuation"]) > 20


def test_include_sentiment_flag(monkeypatch) -> None:
    def fake_fetch(ticker: str, include_sentiment: bool) -> dict:
        row = {
            "ticker": ticker,
            "pe_ratio": 10.0,
            "pb_ratio": 2.0,
            "debt_to_equity": 1.0,
            "roe": 20.0,
            "gross_margin": 50.0,
            "net_margin": 15.0,
            "rsi": 55.0,
            "error": None,
        }
        if include_sentiment:
            row["sentiment"] = "Neutral"
        return row

    monkeypatch.setattr(peers_module, "_fetch_ticker_metrics_sync", fake_fetch)
    result = compare_peers("AAPL", ["MSFT", "GOOGL"], include_sentiment=True)

    assert result["error"] is None
    assert all("sentiment" in row for row in result["summary_table"])
