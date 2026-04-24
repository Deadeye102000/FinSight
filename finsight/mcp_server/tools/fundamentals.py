"""Fundamentals tool for FinSight."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import yfinance as yf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from finsight.mcp_server.utils.validators import validate_ticker


logger = logging.getLogger(__name__)

SECTOR_AVERAGE_PE = {
    "tech": 28.0,
    "finance": 14.0,
    "healthcare": 22.0,
    "energy": 12.0,
    "consumer": 20.0,
    "default": 18.0,
}

ANALYST_RATING_MAP = {
    "strong_buy": "Strong Buy",
    "buy": "Buy",
    "hold": "Hold",
    "sell": "Sell",
    "strong_sell": "Strong Sell",
}


class FundamentalsRequest(BaseModel):
    """Validated request payload for fundamentals lookups."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., description="Ticker symbol such as MSFT or TCS.NS")

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, value: str) -> str:
        normalized = value.upper()
        if not validate_ticker(normalized):
            raise ValueError("Ticker must be non-empty, 20 chars or fewer, and contain no spaces.")
        return normalized


def _empty_result(ticker: str, error: str | None) -> dict[str, Any]:
    """Return a stable fundamentals payload shape."""
    return {
        "ticker": ticker,
        "company_name": "",
        "sector": "",
        "industry": "",
        "currency": "",
        "reporting_currency": "",
        "normalized_currency": "USD",
        "market_cap": 0.0,
        "pe_ratio": None,
        "pb_ratio": None,
        "ps_ratio": None,
        "peg_ratio": None,
        "ev_ebitda": None,
        "eps_ttm": None,
        "revenue_ttm": None,
        "gross_margin": None,
        "operating_margin": None,
        "net_margin": None,
        "roe": None,
        "roa": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "dividend_yield": None,
        "analyst_rating": None,
        "target_price_mean": None,
        "valuation_summary": "Insufficient data",
        "error": error,
    }


def _safe_float(value: Any) -> float | None:
    """Coerce a Yahoo field to float when present."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_billions(value: Any) -> float | None:
    """Convert a raw currency amount into billions."""
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return round(numeric / 1_000_000_000, 2)


def _to_percentage(value: Any) -> float | None:
    """Convert ratio-style values into percentages when needed."""
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if -1.0 <= numeric <= 1.0:
        numeric *= 100
    return round(numeric, 2)


def _normalize_analyst_rating(value: Any) -> str | None:
    """Normalize Yahoo recommendation keys to user-facing labels."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    return ANALYST_RATING_MAP.get(normalized)


def _sector_pe_baseline(sector: str) -> float:
    """Map a Yahoo sector string to a coarse PE benchmark."""
    normalized = sector.strip().lower()
    if "tech" in normalized or "communication" in normalized:
        return SECTOR_AVERAGE_PE["tech"]
    if "financial" in normalized or "bank" in normalized or "insurance" in normalized:
        return SECTOR_AVERAGE_PE["finance"]
    if "health" in normalized or "biotech" in normalized or "pharma" in normalized:
        return SECTOR_AVERAGE_PE["healthcare"]
    if "energy" in normalized or "oil" in normalized or "gas" in normalized:
        return SECTOR_AVERAGE_PE["energy"]
    if "consumer" in normalized or "retail" in normalized:
        return SECTOR_AVERAGE_PE["consumer"]
    return SECTOR_AVERAGE_PE["default"]


def _valuation_summary(pe_ratio: float | None, sector: str) -> str:
    """Summarize valuation relative to a hardcoded sector PE benchmark."""
    if pe_ratio is None:
        return "Insufficient data"

    sector_avg_pe = _sector_pe_baseline(sector)
    if pe_ratio < sector_avg_pe * 0.8:
        return "Undervalued"
    if pe_ratio > sector_avg_pe * 1.2:
        return "Overvalued"
    return "Fairly Valued"


@lru_cache(maxsize=64)
def _fetch_info(ticker: str) -> dict[str, Any]:
    """Fetch fundamentals data from Yahoo Finance."""
    logger.info("Fetching fundamentals info for %s", ticker)
    return yf.Ticker(ticker).info


@lru_cache(maxsize=64)
def _fetch_quote_currency(ticker: str) -> str | None:
    """Fetch the quote currency for a ticker from Yahoo fast info."""
    logger.info("Fetching quote currency for %s", ticker)
    try:
        currency = yf.Ticker(ticker).fast_info.get("currency")
        return str(currency).upper() if currency else None
    except Exception as exc:  # pragma: no cover - defensive around yfinance internals
        logger.warning("Unable to fetch fast_info currency for %s: %s", ticker, exc)
        return None


@lru_cache(maxsize=64)
def _fetch_fx_rate_to_usd(currency: str) -> float | None:
    """Fetch a conversion rate from the quote currency into USD."""
    normalized = currency.strip().upper()
    if not normalized or normalized == "USD":
        return 1.0

    direct_pair = f"{normalized}USD=X"
    inverse_pair = f"USD{normalized}=X"

    try:
        history = yf.Ticker(direct_pair).history(period="5d", interval="1d", auto_adjust=False, timeout=15)
        closes = history.get("Close")
        if closes is not None and not closes.dropna().empty:
            return float(closes.dropna().iloc[-1])
    except Exception as exc:  # pragma: no cover - network/provider issues
        logger.warning("Direct FX lookup failed for %s: %s", direct_pair, exc)

    try:
        history = yf.Ticker(inverse_pair).history(period="5d", interval="1d", auto_adjust=False, timeout=15)
        closes = history.get("Close")
        if closes is not None and not closes.dropna().empty:
            inverse_rate = float(closes.dropna().iloc[-1])
            if inverse_rate != 0:
                return 1 / inverse_rate
    except Exception as exc:  # pragma: no cover - network/provider issues
        logger.warning("Inverse FX lookup failed for %s: %s", inverse_pair, exc)

    logger.warning("Unable to resolve FX conversion to USD for currency %s", normalized)
    return None


def _resolve_quote_currency(ticker: str, info: dict[str, Any]) -> str:
    """Resolve the most reliable quote currency available for a ticker."""
    info_currency = info.get("currency")
    if info_currency:
        return str(info_currency).upper()

    fast_currency = _fetch_quote_currency(ticker)
    if fast_currency:
        return fast_currency

    return "USD"


def _to_billions_usd(value: Any, currency: str) -> float | None:
    """Convert a raw currency amount into USD billions when possible."""
    numeric = _safe_float(value)
    if numeric is None:
        return None

    fx_rate = _fetch_fx_rate_to_usd(currency)
    if fx_rate is None:
        logger.warning("Falling back to raw %s amount because USD FX conversion was unavailable", currency)
        return _to_billions(numeric)

    return round((numeric * fx_rate) / 1_000_000_000, 2)


def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Fetch company fundamentals and valuation metrics for a ticker."""
    logger.info("Processing fundamentals request for ticker=%s", ticker)

    try:
        request = FundamentalsRequest(ticker=ticker)
    except ValidationError as exc:
        logger.warning("Validation failed for fundamentals ticker=%s: %s", ticker, exc)
        return _empty_result(ticker, exc.errors()[0]["msg"])

    try:
        info = _fetch_info(request.ticker)
        company_name = str(info.get("longName") or info.get("shortName") or "")
        sector = str(info.get("sector") or "")
        industry = str(info.get("industry") or "")
        currency = _resolve_quote_currency(request.ticker, info)

        if not any([company_name, sector, industry, info.get("marketCap")]):
            logger.warning("No fundamentals data returned for %s", request.ticker)
            return _empty_result(request.ticker, f"No fundamentals data found for ticker '{request.ticker}'.")

        pe_ratio = _safe_float(info.get("trailingPE"))
        result = {
            **_empty_result(request.ticker, None),
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "currency": currency,
            "reporting_currency": currency,
            "market_cap": _to_billions_usd(info.get("marketCap"), currency) or 0.0,
            "pe_ratio": pe_ratio,
            "pb_ratio": _safe_float(info.get("priceToBook")),
            "ps_ratio": _safe_float(info.get("priceToSalesTrailing12Months")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "eps_ttm": _safe_float(info.get("trailingEps")),
            "revenue_ttm": _to_billions_usd(info.get("totalRevenue"), currency),
            "gross_margin": _to_percentage(info.get("grossMargins")),
            "operating_margin": _to_percentage(info.get("operatingMargins")),
            "net_margin": _to_percentage(info.get("profitMargins")),
            "roe": _to_percentage(info.get("returnOnEquity")),
            "roa": _to_percentage(info.get("returnOnAssets")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "dividend_yield": _to_percentage(info.get("dividendYield")),
            "analyst_rating": _normalize_analyst_rating(info.get("recommendationKey")),
            "target_price_mean": _safe_float(info.get("targetMeanPrice")),
            "valuation_summary": _valuation_summary(pe_ratio, sector),
        }
        logger.info("Completed fundamentals request for %s", request.ticker)
        return result
    except Exception as exc:  # pragma: no cover - network/provider issues
        logger.exception("Failed to fetch fundamentals for %s", request.ticker)
        return _empty_result(request.ticker, str(exc))
