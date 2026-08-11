"""Filing text tool for FinSight (BSE/NSE filings and EDGAR/yfinance fallback)."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import yfinance as yf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from finsight.mcp_server.utils.validators import validate_ticker

logger = logging.getLogger(__name__)

BSE_ANNOUNCEMENTS_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
BSE_COMPANY_INFO_URL = "https://api.bseindia.com/BseIndiaAPI/api/ComHeader/w"
BSE_SCRIP_HEADER_URL = "https://api.bseindia.com/BseIndiaAPI/api/GetScripHeaderData/w"
BSE_ATTACHMENT_BASE_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive"
HTTP_TIMEOUT_SECONDS = 5
BSE_REQUEST_DELAY_SECONDS = 1.0

NSE_TO_BSE_CODE = {
    "TCS": "532540",
    "RELIANCE": "500325",
    "INFY": "500209",
    "HDFCBANK": "500180",
    "ICICIBANK": "532174",
    "WIPRO": "507685",
    "HINDUNILVR": "500696",
    "ITC": "500875",
    "SBIN": "500112",
    "BAJFINANCE": "500034",
    "MARUTI": "532500",
    "ASIANPAINT": "500820",
    "TITAN": "500114",
    "ULTRACEMCO": "532538",
    "NESTLEIND": "500790",
    "POWERGRID": "532898",
    "NTPC": "532555",
    "ONGC": "500312",
    "SUNPHARMA": "524715",
    "DRREDDY": "500124",
    "TECHM": "532755",
    "HCLTECH": "532281",
    "AXISBANK": "532215",
    "KOTAKBANK": "500247",
    "LT": "500510",
    "BHARTIARTL": "532454",
    "ADANIPORTS": "532921",
    "TATAMOTORS": "500570",
    "TATASTEEL": "500470",
    "COALINDIA": "533278",
}

_last_bse_request_at = 0.0


class FilingTextRequest(BaseModel):
    """Validated request payload for filing text retrieval."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., description="Ticker symbol such as TCS.NS, 532540, or AAPL")
    filing_type: str = Field(default="annual_report", description="annual_report, financial_results, or all")

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, value: str) -> str:
        normalized = value.upper()
        if not validate_ticker(normalized):
            raise ValueError("Ticker must be non-empty, 20 chars or fewer, and contain no spaces.")
        return normalized

    @field_validator("filing_type")
    @classmethod
    def validate_filing_type_field(cls, value: str) -> str:
        normalized = value.lower().strip()
        valid = {"annual_report", "financial_results", "10-k", "10-q", "all"}
        if normalized not in valid:
            raise ValueError("filing_type must be annual_report, financial_results, 10-k, 10-q, or all.")
        return normalized


def _bse_headers() -> dict[str, str]:
    """Return headers accepted by BSE public APIs."""
    return {
        "User-Agent": "Mozilla/5.0 FinSight-Research-Agent/1.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    }


def _sleep_before_bse_call() -> None:
    """Throttle BSE API calls to comply with terms of service."""
    global _last_bse_request_at
    elapsed = time.monotonic() - _last_bse_request_at
    if _last_bse_request_at and elapsed < BSE_REQUEST_DELAY_SECONDS:
        time.sleep(BSE_REQUEST_DELAY_SECONDS - elapsed)
    _last_bse_request_at = time.monotonic()


def _empty_result(ticker: str, error: str | None) -> dict[str, Any]:
    """Return a stable empty filing text payload."""
    is_indian = ticker.endswith((".NS", ".BO")) or ticker.isdigit()
    return {
        "ticker": ticker,
        "bse_code": None,
        "company_name": "",
        "filing_type": "annual_report",
        "filing_date": None,
        "title": "",
        "attachment_url": "",
        "reporting_currency": "INR" if is_indian else "USD",
        "normalized_currency": "USD",
        "filing_text": "",
        "source": "BSE" if is_indian else "yfinance",
        "error": error,
    }


def _resolve_bse_code(ticker: str) -> tuple[str | None, str]:
    """Resolve ticker symbol to BSE code and symbol."""
    normalized = ticker.strip().upper()
    if normalized.isdigit():
        return normalized, normalized

    symbol = normalized[:-3] if normalized.endswith((".NS", ".BO")) else normalized
    if symbol in NSE_TO_BSE_CODE:
        return NSE_TO_BSE_CODE[symbol], symbol

    try:
        _sleep_before_bse_call()
        res = httpx.get(
            BSE_SCRIP_HEADER_URL,
            params={"quotetype": "EQ", "scripcode": symbol},
            headers=_bse_headers(),
            timeout=HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        if res.status_code == 200:
            payload = res.json()
            equity = (payload.get("Cmpname") or {}).get("EquityScrips") if isinstance(payload, dict) else None
            if isinstance(equity, list) and equity:
                code = equity[0].get("SCRIP_CD") or equity[0].get("SecurityCode")
                if code:
                    return str(code), symbol
    except Exception as exc:
        logger.warning("BSE code lookup failed for %s: %s", symbol, exc)

    return None, symbol


def _fetch_bse_disclosures(bse_code: str) -> list[dict[str, Any]]:
    """Fetch raw BSE corporate disclosures."""
    _sleep_before_bse_call()
    response = httpx.get(
        BSE_ANNOUNCEMENTS_URL,
        params={
            "pageno": 1,
            "strCat": "-1",
            "strPrevDate": "",
            "strScrip": bse_code,
            "strSearch": "P",
            "strToDate": "",
            "strType": "C",
        },
        headers=_bse_headers(),
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "Table" in payload:
        return list(payload["Table"])
    return []


def _matches_filing_type(item: dict[str, Any], filing_type: str) -> bool:
    """Check if a BSE disclosure matches the requested filing type."""
    headline = str(item.get("NEWSSUB") or "").lower()
    category = str(item.get("CATEGORYNAME") or "").lower()
    text = f"{headline} {category}"

    if filing_type in {"annual_report", "10-k"}:
        return any(k in text for k in ("annual report", "reg. 34", "reg 34", "annual"))
    if filing_type in {"financial_results", "10-q"}:
        return any(k in text for k in ("result", "financial", "audited", "unaudited", "quarter"))
    return True


def get_filing_text(ticker: str, filing_type: str = "annual_report") -> dict[str, Any]:
    """Fetch recent filing text or annual report disclosure for a ticker."""
    logger.info("Processing get_filing_text request for ticker=%s filing_type=%s", ticker, filing_type)

    try:
        request = FilingTextRequest(ticker=ticker, filing_type=filing_type)
    except ValidationError as exc:
        logger.warning("Validation failed for get_filing_text ticker=%s: %s", ticker, exc)
        return _empty_result(ticker, exc.errors()[0]["msg"])

    bse_code, symbol = _resolve_bse_code(request.ticker)

    if bse_code:
        try:
            disclosures = _fetch_bse_disclosures(bse_code)
            matching = [item for item in disclosures if _matches_filing_type(item, request.filing_type)]
            selected = matching[0] if matching else (disclosures[0] if disclosures else None)

            if selected:
                headline = str(selected.get("NEWSSUB") or selected.get("CATEGORYNAME") or "Corporate Filing").strip()
                filing_date = str(selected.get("DT_TM") or "").split("T")[0] or None
                attach_name = str(selected.get("ATTACHMENTNAME") or "").strip()
                attach_url = f"{BSE_ATTACHMENT_BASE_URL}/{attach_name}" if attach_name else ""
                more_notes = str(selected.get("MORE") or selected.get("HEADLINE") or "").strip()
                company_name = str(selected.get("SLONGNAME") or symbol)

                filing_text = (
                    f"Filing Title: {headline}\n"
                    f"Filing Category: {selected.get('CATEGORYNAME')}\n"
                    f"Submission Date: {filing_date}\n"
                    f"Exchange/Source: BSE ({bse_code})\n"
                    f"Attachment: {attach_url}\n\n"
                    f"Summary Text:\n{more_notes if more_notes else headline}"
                )

                return {
                    "ticker": request.ticker,
                    "bse_code": bse_code,
                    "company_name": company_name,
                    "filing_type": request.filing_type,
                    "filing_date": filing_date,
                    "title": headline,
                    "attachment_url": attach_url,
                    "reporting_currency": "INR",
                    "normalized_currency": "USD",
                    "filing_text": filing_text,
                    "source": "BSE Disclosures",
                    "error": None,
                }
        except Exception as exc:
            logger.warning("Failed to fetch BSE disclosures for %s: %s", request.ticker, exc)

    try:
        info = yf.Ticker(request.ticker).info
        company_name = str(info.get("longName") or info.get("shortName") or request.ticker)
        business_summary = str(info.get("longBusinessSummary") or "")
        quote_currency = str(info.get("currency") or "USD").upper()

        filing_text = (
            f"Filing Type: {request.filing_type.upper()}\n"
            f"Company Name: {company_name}\n"
            f"Ticker: {request.ticker}\n"
            f"Reporting Currency: {quote_currency}\n\n"
            f"Business Overview & Summary:\n{business_summary[:2000] if business_summary else 'No business summary available.'}"
        )

        return {
            "ticker": request.ticker,
            "bse_code": None,
            "company_name": company_name,
            "filing_type": request.filing_type,
            "filing_date": None,
            "title": f"Recent Disclosure / Overview ({request.filing_type})",
            "attachment_url": f"https://finance.yahoo.com/quote/{request.ticker}",
            "reporting_currency": quote_currency,
            "normalized_currency": "USD",
            "filing_text": filing_text,
            "source": "yfinance",
            "error": None,
        }
    except Exception as exc:
        logger.exception("Failed filing lookup for ticker %s", request.ticker)
        return _empty_result(request.ticker, str(exc))
