"""Input validation helpers."""

from __future__ import annotations


ALLOWED_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}


def validate_ticker(ticker: str) -> bool:
    """Return ``True`` when the ticker matches FinSight's safety rules."""
    if not isinstance(ticker, str):
        return False

    normalized = ticker.strip()
    if not normalized:
        return False
    if len(normalized) > 20:
        return False
    if " " in normalized:
        return False
    return True


def validate_period(period: str) -> bool:
    """Return ``True`` when the period is one of the supported Yahoo ranges."""
    return period in ALLOWED_PERIODS


def validate_n(n: int) -> bool:
    """Return ``True`` when a result count is within FinSight's supported bounds."""
    return isinstance(n, int) and 1 <= n <= 50


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a stock ticker symbol."""
    normalized = symbol.strip().upper()
    if not validate_ticker(normalized):
        raise ValueError("Stock symbol is required.")
    return normalized
