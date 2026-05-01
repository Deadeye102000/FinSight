"""Tests for the FinSight news sentiment tool."""

from __future__ import annotations

import pytest

from finsight.mcp_server.tools.sentiment import get_news_sentiment, load_finbert_model
from finsight.mcp_server.utils.validators import validate_n


@pytest.fixture(autouse=True)
def without_news_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use deterministic mock NewsAPI data unless a test says otherwise."""
    monkeypatch.delenv("NEWS_API_KEY", raising=False)


def test_finbert_loads() -> None:
    classifier = load_finbert_model()
    assert classifier is not None


def test_sentiment_score_range() -> None:
    result = get_news_sentiment("MSFT", "Microsoft", n=5)
    assert -1.0 <= result["sentiment_score"] <= 1.0


def test_overall_sentiment_valid() -> None:
    result = get_news_sentiment("MSFT", "Microsoft", n=5)
    assert result["overall_sentiment"] in ["Positive", "Negative", "Neutral"]


def test_confidence_range() -> None:
    result = get_news_sentiment("MSFT", "Microsoft", n=5)
    assert 0.0 <= result["confidence"] <= 1.0


def test_counts_sum_to_total() -> None:
    result = get_news_sentiment("MSFT", "Microsoft", n=5)
    total = result["positive_count"] + result["negative_count"] + result["neutral_count"]
    assert total == result["headlines_analysed"]


def test_no_api_key_returns_graceful_error() -> None:
    result = get_news_sentiment("MSFT", "Microsoft", n=5)
    assert result["error"] is not None
    assert result["headlines_analysed"] <= 5


def test_headlines_list_structure() -> None:
    result = get_news_sentiment("MSFT", "Microsoft", n=5)
    for headline in result["headlines"]:
        assert {"headline", "sentiment", "score", "source"}.issubset(headline.keys())


def test_n_parameter_respected() -> None:
    result = get_news_sentiment("MSFT", "Microsoft", n=5)
    assert result["headlines_analysed"] <= 5


def test_validate_n_bounds() -> None:
    assert validate_n(0) is False
    assert validate_n(51) is False
    assert validate_n(10) is True


def test_finbert_classifies_positive() -> None:
    classifier = load_finbert_model()
    result = classifier("Company reports record profits and revenue growth")
    assert result[0]["label"].lower() == "positive"


def test_finbert_classifies_negative() -> None:
    classifier = load_finbert_model()
    result = classifier("Company faces bankruptcy and massive losses")
    assert result[0]["label"].lower() == "negative"
