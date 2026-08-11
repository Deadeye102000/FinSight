"""News sentiment tool for FinSight."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests
import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from transformers import pipeline
from transformers.pipelines import Pipeline

from finsight.mcp_server.utils.validators import validate_n, validate_ticker


logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

NEWS_API_URL = "https://newsapi.org/v2/everything"
FINBERT_MODEL = "ProsusAI/finbert"
SENTIMENT_LABELS = {"Positive", "Negative", "Neutral"}

_finbert_pipeline: Pipeline | None = None


class NewsSentimentRequest(BaseModel):
    """Validated request payload for news sentiment lookups."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., description="Ticker symbol such as MSFT or TCS.NS")
    days_back: int = Field(default=7, description="Days back to fetch headlines")
    company_name: str | None = Field(default=None, description="Optional company name")
    n: int = Field(default=10, description="Number of headlines to fetch")

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, value: str) -> str:
        normalized = value.upper()
        if not validate_ticker(normalized):
            raise ValueError("Ticker must be non-empty, 20 chars or fewer, and contain no spaces.")
        return normalized

    @field_validator("days_back")
    @classmethod
    def validate_days_back_field(cls, value: int) -> int:
        if value < 1 or value > 90:
            raise ValueError("days_back must be between 1 and 90.")
        return value

    @field_validator("n")
    @classmethod
    def validate_n_field(cls, value: int) -> int:
        if not validate_n(value):
            raise ValueError("n must be between 1 and 50.")
        return value


def load_finbert_model() -> Pipeline:
    """Load and cache the local FinBERT sentiment pipeline."""
    global _finbert_pipeline

    if _finbert_pipeline is None:
        logger.info("Loading FinBERT model %s", FINBERT_MODEL)
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            model_kwargs={"use_safetensors": False},
        )
    return _finbert_pipeline


def _empty_result(ticker: str, error: str | None) -> dict[str, Any]:
    """Return a stable news sentiment payload shape."""
    return {
        "ticker": ticker,
        "headlines_analysed": 0,
        "overall_sentiment": "Neutral",
        "sentiment_score": 0.0,
        "confidence": 0.0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "top_positive_headline": None,
        "top_negative_headline": None,
        "headlines": [],
        "error": error,
    }


def _mock_articles(ticker: str, company_name: str, n: int) -> list[dict[str, str]]:
    """Return deterministic mock articles when NewsAPI is not configured."""
    templates = [
        f"{company_name} reports record profits and revenue growth",
        f"{company_name} shares steady as investors await guidance",
        f"{company_name} faces pressure after weaker demand outlook",
        f"Analysts review {ticker} after quarterly update",
        f"{company_name} announces new product expansion plans",
    ]
    repeated = (templates * ((n // len(templates)) + 1))[:n]
    return [
        {
            "title": headline,
            "source": {"name": "Mock News"},
            "publishedAt": None,
        }
        for headline in repeated
    ]


def _fetch_articles(ticker: str, company_name: str, n: int, api_key: str) -> list[dict[str, Any]]:
    """Fetch latest English headlines from NewsAPI."""
    query = f'("{company_name}" OR {ticker})'
    response = requests.get(
        NEWS_API_URL,
        params={
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": n,
            "apiKey": api_key,
        },
        timeout=2,
    )
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("articles") or [])


def _classify_headlines(articles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, float]:
    """Run FinBERT over article titles and return normalized headline payloads."""
    article_headlines = [
        (article, headline)
        for article in articles
        if (headline := str(article.get("title") or "").strip())
    ]
    headlines = [headline for _, headline in article_headlines]
    if not headlines:
        return [], 0.0, 0.0

    classifier = load_finbert_model()
    predictions = classifier(headlines, truncation=True)
    if isinstance(predictions, dict):
        predictions = [predictions]

    analysed_headlines: list[dict[str, Any]] = []
    scores: list[float] = []
    confidences: list[float] = []

    for (article, headline), prediction in zip(article_headlines, predictions, strict=False):
        label = str(prediction["label"]).lower()
        confidence = float(prediction["score"])
        signed_score = confidence if label == "positive" else -confidence if label == "negative" else 0.0
        sentiment = label.capitalize()

        source = article.get("source") or {}
        source_name = source.get("name") if isinstance(source, dict) else None

        analysed_headlines.append(
            {
                "headline": headline,
                "sentiment": sentiment if sentiment in SENTIMENT_LABELS else "Neutral",
                "score": round(signed_score, 4),
                "source": source_name or "",
                "published_at": article.get("publishedAt"),
            }
        )
        scores.append(signed_score)
        confidences.append(confidence)

    sentiment_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    return analysed_headlines, sentiment_score, confidence


def _summarize_result(
    ticker: str,
    headlines: list[dict[str, Any]],
    confidence: float,
    error: str | None,
) -> dict[str, Any]:
    """Summarize classified headlines into the public tool response."""
    if not headlines:
        return _empty_result(ticker, error)

    sentiment_score = round(sum(float(item["score"]) for item in headlines) / len(headlines), 4)
    positive_headlines = [item for item in headlines if item["sentiment"] == "Positive"]
    negative_headlines = [item for item in headlines if item["sentiment"] == "Negative"]
    neutral_headlines = [item for item in headlines if item["sentiment"] == "Neutral"]

    if sentiment_score > 0.1:
        overall_sentiment = "Positive"
    elif sentiment_score < -0.1:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    top_positive = max(positive_headlines, key=lambda item: item["score"], default=None)
    top_negative = min(negative_headlines, key=lambda item: item["score"], default=None)

    return {
        "ticker": ticker,
        "headlines_analysed": len(headlines),
        "overall_sentiment": overall_sentiment,
        "sentiment_score": sentiment_score,
        "confidence": confidence,
        "positive_count": len(positive_headlines),
        "negative_count": len(negative_headlines),
        "neutral_count": len(neutral_headlines),
        "top_positive_headline": top_positive["headline"] if top_positive else None,
        "top_negative_headline": top_negative["headline"] if top_negative else None,
        "headlines": headlines,
        "error": error,
    }


def _fetch_raw_headlines(ticker: str, days_back: int = 7) -> list[dict[str, Any]]:
    """Fetch raw headlines and timestamps using yfinance news with fallback."""
    raw_items: list[dict[str, Any]] = []
    try:
        yf_news = yf.Ticker(ticker).news or []
        for item in yf_news:
            content = item.get("content", item)
            title = str(content.get("title") or item.get("title") or "").strip()
            if not title:
                continue
            pub_date = (
                content.get("pubDate")
                or content.get("displayTime")
                or item.get("providerPublishTime")
            )
            provider_obj = content.get("provider") or {}
            publisher = (
                provider_obj.get("displayName")
                if isinstance(provider_obj, dict)
                else item.get("publisher") or "Yahoo Finance"
            )
            canonical = content.get("canonicalUrl") or {}
            url = (
                canonical.get("url")
                if isinstance(canonical, dict)
                else item.get("link") or ""
            )

            raw_items.append(
                {
                    "title": title,
                    "publisher": str(publisher or "Yahoo Finance"),
                    "published_at": str(pub_date) if pub_date else None,
                    "url": str(url or ""),
                }
            )
    except Exception as exc:
        logger.warning("yfinance news lookup failed for %s: %s", ticker, exc)

    if not raw_items:
        mock_articles = _mock_articles(ticker, ticker, 5)
        for article in mock_articles:
            raw_items.append(
                {
                    "title": article["title"],
                    "publisher": article["source"]["name"],
                    "published_at": None,
                    "url": "",
                }
            )

    return raw_items


def get_news_sentiment(
    ticker: str,
    days_back: int = 7,
    company_name: str | None = None,
    n: int = 10,
) -> dict[str, Any]:
    """Fetch recent headlines for a stock ticker.

    When company_name is provided, legacy FinBERT sentiment scoring is performed.
    Otherwise, returns raw headlines + timestamps without sentiment scoring.
    """
    logger.info("Processing news sentiment request for ticker=%s (days_back=%d)", ticker, days_back)

    try:
        request = NewsSentimentRequest(ticker=ticker, days_back=days_back, company_name=company_name, n=n)
    except ValidationError as exc:
        logger.warning("Validation failed for news sentiment ticker=%s: %s", ticker, exc)
        return _empty_result(ticker, exc.errors()[0]["msg"])

    if company_name is not None:
        api_key = os.getenv("NEWS_API_KEY")
        error = None

        if not api_key:
            error = (
                "NEWS_API_KEY is not set. Returning mock sentiment data. "
                "Get a free key at https://newsapi.org and set NEWS_API_KEY in your environment."
            )
            articles = _mock_articles(request.ticker, request.company_name or request.ticker, request.n)
        else:
            try:
                articles = _fetch_articles(request.ticker, request.company_name or request.ticker, request.n, api_key)
            except Exception as exc:  # pragma: no cover - network/provider issues
                logger.exception("Failed to fetch NewsAPI articles for %s", request.ticker)
                return _empty_result(request.ticker, str(exc))

        headlines, _, confidence = _classify_headlines(articles[: request.n])
        return _summarize_result(request.ticker, headlines, confidence, error)

    raw_headlines = _fetch_raw_headlines(request.ticker, request.days_back)
    return {
        "ticker": request.ticker,
        "days_back": request.days_back,
        "headline_count": len(raw_headlines),
        "headlines": raw_headlines,
        "source": "yfinance",
        "error": None,
    }
