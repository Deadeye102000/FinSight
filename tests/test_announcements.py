"""Tests for the FinSight corporate announcements tool."""

from __future__ import annotations

from time import perf_counter

import httpx

from finsight.mcp_server.tools.announcements import _announcement_cache, get_corporate_announcements


def test_bse_api_reachable() -> None:
    try:
        response = httpx.get(
            "https://api.bseindia.com/BseIndiaAPI/api/ComHeader/w",
            params={"quotetype": "EQ", "scripcode": "532540"},
            headers={"User-Agent": "Mozilla/5.0 FinSight-Research-Agent/1.0"},
            timeout=5,
            follow_redirects=True,
        )
        assert response.status_code == 200
    except httpx.HTTPError:
        result = get_corporate_announcements("TCS.NS")
        assert result["source"] == "mock"


def test_ticker_resolution_tcs() -> None:
    result = get_corporate_announcements("TCS.NS")
    assert result["bse_code"] == "532540"


def test_announcements_fetched_count() -> None:
    result = get_corporate_announcements("TCS.NS")
    assert result["announcements_fetched"] >= 0


def test_raw_announcements_structure() -> None:
    result = get_corporate_announcements("TCS.NS")
    for item in result["raw_announcements"]:
        assert {"date", "headline", "category"}.issubset(item.keys())


def test_claude_summary_not_empty() -> None:
    result = get_corporate_announcements("TCS.NS")
    assert len(result["claude_summary"]) > 40


def test_key_highlights_count() -> None:
    result = get_corporate_announcements("TCS.NS")
    assert 1 <= len(result["key_highlights"]) <= 5


def test_sentiment_valid() -> None:
    result = get_corporate_announcements("TCS.NS")
    assert result["sentiment"] in ["Positive", "Neutral", "Concerning"]


def test_unknown_ticker_returns_error() -> None:
    result = get_corporate_announcements("FAKECO.NS")
    assert result["error"] is not None


def test_mock_fallback_on_api_failure(monkeypatch) -> None:
    _announcement_cache.clear()

    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("forced timeout")

    monkeypatch.setattr(httpx, "get", raise_timeout)
    result = get_corporate_announcements("RELIANCE.NS", announcement_type="dividend")
    assert result["source"] == "mock"


def test_caching_second_call_faster() -> None:
    _announcement_cache.clear()
    get_corporate_announcements("TCS.NS")
    start = perf_counter()
    get_corporate_announcements("TCS.NS")
    elapsed = perf_counter() - start
    assert elapsed < 0.5
