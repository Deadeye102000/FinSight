"""Unit tests for filing text RAG layer and retrieval function."""

from __future__ import annotations

import pytest

from finsight.rag.retriever import retrieve_relevant_filing_context
from scripts.ingest_filings import WATCHLIST, ingest_watchlist


def test_retrieve_relevant_filing_context_missing_ticker() -> None:
    """Test retrieve_relevant_filing_context returns graceful message for non-existent ticker."""
    res = retrieve_relevant_filing_context("UNKNOWN_TICKER_999", "revenue growth")
    assert "No filing context collection found" in res or "No relevant filing excerpts" in res


def test_retrieve_relevant_filing_context_signature() -> None:
    """Test retrieve_relevant_filing_context handles valid queries cleanly."""
    res = retrieve_relevant_filing_context("TCS.NS", "revenue growth risk outlook", k=2)
    assert isinstance(res, str)
    assert len(res) > 0


def test_watchlist_configuration() -> None:
    """Test watchlist contains valid non-empty ticker strings."""
    assert isinstance(WATCHLIST, list)
    assert len(WATCHLIST) >= 20
    assert "TCS.NS" in WATCHLIST
    assert "RELIANCE.NS" in WATCHLIST
