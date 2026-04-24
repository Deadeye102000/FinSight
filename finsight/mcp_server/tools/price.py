"""Stock price tool for FinSight."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from finsight.mcp_server.utils.validators import validate_period, validate_ticker


logger = logging.getLogger(__name__)
PERIOD_ORDER = {"1mo": 1, "3mo": 2, "6mo": 3, "1y": 4, "2y": 5, "5y": 6}
TECHNICAL_MINIMUM_PERIOD = "2y"


class StockPriceRequest(BaseModel):
    """Validated input payload for the stock price tool."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., description="Ticker symbol such as AAPL or RELIANCE.NS")
    period: str = Field(default="3mo", description="Supported periods: 1mo, 3mo, 6mo, 1y, 2y, 5y")
    interval: str = Field(default="1d", min_length=2, max_length=10)

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, value: str) -> str:
        normalized = value.upper()
        if not validate_ticker(normalized):
            raise ValueError("Ticker must be non-empty, 20 chars or fewer, and contain no spaces.")
        return normalized

    @field_validator("period")
    @classmethod
    def validate_period_field(cls, value: str) -> str:
        if not validate_period(value):
            raise ValueError("Period must be one of: 1mo, 3mo, 6mo, 1y, 2y, 5y.")
        return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float while tolerating pandas / numpy scalars."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int while tolerating pandas / numpy scalars."""
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute Wilder's RSI manually from a closing-price series."""
    closes = pd.Series(series, copy=False).dropna().astype(float)
    if len(closes) <= period:
        return 0.0

    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.iloc[1 : period + 1].mean()
    avg_loss = losses.iloc[1 : period + 1].mean()

    for index in range(period + 1, len(closes)):
        avg_gain = ((avg_gain * (period - 1)) + gains.iloc[index]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses.iloc[index]) / period

    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi), 2)


def _compute_macd_signal(series: pd.Series) -> str:
    """Classify MACD as bullish, bearish, or neutral."""
    closes = pd.Series(series, copy=False).dropna().astype(float)
    if closes.empty:
        return "neutral"

    ema_12 = closes.ewm(span=12, adjust=False).mean()
    ema_26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    latest_macd = float(macd_line.iloc[-1])
    latest_signal = float(signal_line.iloc[-1])
    baseline = max(abs(latest_macd), abs(latest_signal), 1e-9)
    if abs(latest_macd - latest_signal) / baseline <= 0.005:
        return "neutral"
    return "bullish" if latest_macd > latest_signal else "bearish"


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output and keep only the columns we need."""
    if history.empty:
        return history

    normalized = history.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    desired_columns = ["Open", "High", "Low", "Close", "Volume"]
    available = [column for column in desired_columns if column in normalized.columns]
    normalized = normalized[available].dropna(subset=["Close"])
    return normalized


@lru_cache(maxsize=64)
def _fetch_history(ticker: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    """Fetch market data for the resolved analysis window."""
    logger.info("Fetching market data for %s with period=%s interval=%s", ticker, period, interval)
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, interval=interval, auto_adjust=False, timeout=15)
    history = _normalize_history(history)
    currency = "USD"

    try:
        fast_info = stock.fast_info
        if fast_info:
            currency = str(fast_info.get("currency") or currency)
    except Exception as exc:  # pragma: no cover - defensive against yfinance internals
        logger.warning("Unable to read fast_info currency for %s: %s", ticker, exc)

    if history.empty:
        logger.warning("No market data returned for %s", ticker)

    return history, currency


def _resolve_fetch_period(user_period: str) -> str:
    """Choose the smallest Yahoo period that still supports all requested indicators."""
    if PERIOD_ORDER[user_period] >= PERIOD_ORDER[TECHNICAL_MINIMUM_PERIOD]:
        return user_period
    return TECHNICAL_MINIMUM_PERIOD


def _slice_requested_history(history: pd.DataFrame, user_period: str) -> pd.DataFrame:
    """Return the portion of data aligned with the caller's requested analysis window."""
    period_to_business_days = {
        "1mo": 22,
        "3mo": 66,
        "6mo": 132,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
    }
    return history.tail(period_to_business_days[user_period])


def get_stock_price(ticker: str, period: str = "3mo", interval: str = "1d") -> dict[str, Any]:
    """Fetch OHLCV data and technical indicators for a stock ticker."""
    logger.info("Processing stock price request for ticker=%s period=%s interval=%s", ticker, period, interval)

    try:
        request = StockPriceRequest(ticker=ticker, period=period, interval=interval)
    except ValidationError as exc:
        logger.warning("Validation failed for ticker=%s: %s", ticker, exc)
        return {
            "ticker": ticker,
            "current_price": 0.0,
            "currency": "",
            "price_change_pct_30d": 0.0,
            "high_52w": 0.0,
            "low_52w": 0.0,
            "avg_volume_10d": 0,
            "rsi_14": 0.0,
            "macd_signal": "neutral",
            "ma_50": 0.0,
            "ma_200": 0.0,
            "golden_cross": False,
            "ohlcv_last_5": [],
            "error": exc.errors()[0]["msg"],
        }

    try:
        fetch_period = _resolve_fetch_period(request.period)
        history, currency = _fetch_history(request.ticker, fetch_period, request.interval)
        if history.empty:
            return {
                "ticker": request.ticker,
                "current_price": 0.0,
                "currency": currency,
                "price_change_pct_30d": 0.0,
                "high_52w": 0.0,
                "low_52w": 0.0,
                "avg_volume_10d": 0,
                "rsi_14": 0.0,
                "macd_signal": "neutral",
                "ma_50": 0.0,
                "ma_200": 0.0,
                "golden_cross": False,
                "ohlcv_last_5": [],
                "error": f"No price data found for ticker '{request.ticker}'.",
            }

        requested_history = _slice_requested_history(history, request.period)
        technical_history = history
        close = technical_history["Close"]
        latest_close = _safe_float(close.iloc[-1])
        requested_close = requested_history["Close"]
        comparison_index = -31 if len(requested_close) > 30 else 0
        comparison_close = _safe_float(requested_close.iloc[comparison_index], latest_close)
        if comparison_close == 0:
            price_change_pct_30d = 0.0
        else:
            price_change_pct_30d = round(((latest_close - comparison_close) / comparison_close) * 100, 2)

        trailing_year = technical_history.tail(252) if len(technical_history) >= 252 else technical_history
        avg_volume_10d = _safe_int(requested_history["Volume"].tail(10).mean())
        ma_50 = round(_safe_float(close.rolling(window=50, min_periods=1).mean().iloc[-1]), 2)
        ma_200 = round(_safe_float(close.rolling(window=200, min_periods=1).mean().iloc[-1]), 2)

        ohlcv_last_5 = []
        for index, row in requested_history.tail(5).iterrows():
            ohlcv_last_5.append(
                {
                    "date": index.date().isoformat(),
                    "open": round(_safe_float(row["Open"]), 2),
                    "high": round(_safe_float(row["High"]), 2),
                    "low": round(_safe_float(row["Low"]), 2),
                    "close": round(_safe_float(row["Close"]), 2),
                    "volume": _safe_int(row["Volume"]),
                }
            )

        result = {
            "ticker": request.ticker,
            "current_price": round(latest_close, 2),
            "currency": currency,
            "price_change_pct_30d": price_change_pct_30d,
            "high_52w": round(_safe_float(trailing_year["High"].max()), 2),
            "low_52w": round(_safe_float(trailing_year["Low"].min()), 2),
            "avg_volume_10d": avg_volume_10d,
            "rsi_14": _compute_rsi(close, period=14),
            "macd_signal": _compute_macd_signal(close),
            "ma_50": ma_50,
            "ma_200": ma_200,
            "golden_cross": ma_50 > ma_200,
            "ohlcv_last_5": ohlcv_last_5,
            "error": None,
        }
        logger.info("Completed stock price request for %s", request.ticker)
        return result
    except Exception as exc:  # pragma: no cover - network/provider issues
        logger.exception("Failed to fetch stock price data for %s", request.ticker)
        return {
            "ticker": request.ticker,
            "current_price": 0.0,
            "currency": "",
            "price_change_pct_30d": 0.0,
            "high_52w": 0.0,
            "low_52w": 0.0,
            "avg_volume_10d": 0,
            "rsi_14": 0.0,
            "macd_signal": "neutral",
            "ma_50": 0.0,
            "ma_200": 0.0,
            "golden_cross": False,
            "ohlcv_last_5": [],
            "error": str(exc),
        }
