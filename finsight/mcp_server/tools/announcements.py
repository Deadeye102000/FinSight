"""BSE/NSE corporate announcements tool for FinSight."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from finsight.mcp_server.utils.validators import validate_n, validate_ticker


logger = logging.getLogger(__name__)

BSE_ANNOUNCEMENTS_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
BSE_COMPANY_INFO_URL = "https://api.bseindia.com/BseIndiaAPI/api/ComHeader/w"
BSE_SCRIP_HEADER_URL = "https://api.bseindia.com/BseIndiaAPI/api/GetScripHeaderData/w"
NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corp-info"
BSE_ATTACHMENT_BASE_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive"
CLAUDE_MODEL = "claude-3-haiku-20240307"
HTTP_TIMEOUT_SECONDS = 5
BSE_REQUEST_DELAY_SECONDS = 1
CACHE_TTL_SECONDS = 30 * 60
VALID_ANNOUNCEMENT_TYPES = {"results", "dividend", "merger", "all"}
VALID_SENTIMENTS = {"Positive", "Neutral", "Concerning"}

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

_announcement_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_last_bse_request_at = 0.0


class CorporateAnnouncementsRequest(BaseModel):
    """Validated request payload for Indian corporate announcements."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., description="NSE ticker such as TCS.NS or BSE code such as 532540")
    announcement_type: str = Field(default="all", description="results, dividend, merger, or all")
    n: int = Field(default=10, description="Number of announcements to fetch")

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, value: str) -> str:
        normalized = value.upper()
        if not validate_ticker(normalized):
            raise ValueError("Ticker must be non-empty, 20 chars or fewer, and contain no spaces.")
        return normalized

    @field_validator("announcement_type")
    @classmethod
    def validate_announcement_type_field(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in VALID_ANNOUNCEMENT_TYPES:
            raise ValueError("announcement_type must be results, dividend, merger, or all.")
        return normalized

    @field_validator("n")
    @classmethod
    def validate_n_field(cls, value: int) -> int:
        if not validate_n(value):
            raise ValueError("n must be between 1 and 50.")
        return value


def _headers(referer: str = "https://www.bseindia.com/") -> dict[str, str]:
    """Return browser-like headers accepted by BSE/NSE public APIs."""
    return {
        "User-Agent": "Mozilla/5.0 FinSight-Research-Agent/1.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": referer,
        "Origin": referer.rstrip("/"),
    }


def _sleep_before_bse_call() -> None:
    """Add a small delay between BSE calls."""
    global _last_bse_request_at

    elapsed = time.monotonic() - _last_bse_request_at
    if _last_bse_request_at and elapsed < BSE_REQUEST_DELAY_SECONDS:
        time.sleep(BSE_REQUEST_DELAY_SECONDS - elapsed)
    _last_bse_request_at = time.monotonic()


def _bse_get(url: str, params: dict[str, str]) -> httpx.Response:
    """GET a BSE API resource with rate limiting."""
    _sleep_before_bse_call()
    response = httpx.get(
        url,
        params=params,
        headers=_headers(),
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response


def _nse_get(symbol: str) -> httpx.Response:
    """GET NSE announcements as a fallback."""
    response = httpx.get(
        NSE_ANNOUNCEMENTS_URL,
        params={"symbol": symbol, "corpType": "announcements"},
        headers=_headers("https://www.nseindia.com/"),
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response


def _empty_result(ticker: str, bse_code: str | None, company_name: str, error: str | None) -> dict[str, Any]:
    """Return a stable corporate announcements payload shape."""
    return {
        "ticker": ticker,
        "bse_code": bse_code,
        "company_name": company_name,
        "announcements_fetched": 0,
        "announcement_types_found": [],
        "latest_quarterly_result": None,
        "upcoming_events": [],
        "claude_summary": "",
        "key_highlights": [],
        "sentiment": "Neutral",
        "raw_announcements": [],
        "source": "BSE",
        "error": error,
    }


def _nse_symbol_from_ticker(ticker: str) -> str:
    """Normalize an NSE-style ticker."""
    normalized = ticker.strip().upper()
    return normalized[:-3] if normalized.endswith(".NS") else normalized


def _parse_json_response(response: httpx.Response) -> Any:
    """Parse JSON and reject HTML fallback pages."""
    content_type = response.headers.get("content-type", "")
    if "html" in content_type.lower() or response.text.lstrip().startswith("<"):
        raise ValueError("API returned HTML instead of JSON.")
    return response.json()


def _search_bse_code(symbol: str) -> str | None:
    """Best-effort BSE lookup for symbols outside the hardcoded top-30 map."""
    if symbol.isdigit():
        return symbol

    response = _bse_get(BSE_SCRIP_HEADER_URL, params={"quotetype": "EQ", "scripcode": symbol})
    payload = _parse_json_response(response)
    equity = (payload.get("Cmpname") or {}).get("EquityScrips") if isinstance(payload, dict) else None
    if isinstance(equity, list) and equity:
        code = equity[0].get("SCRIP_CD") or equity[0].get("SecurityCode")
        return str(code) if code else None
    return None


def _resolve_ticker(ticker: str) -> tuple[str | None, str | None]:
    """Resolve an input ticker to BSE code and NSE symbol when available."""
    normalized = ticker.strip().upper()
    if normalized.isdigit():
        return normalized, None

    symbol = _nse_symbol_from_ticker(normalized)
    if symbol in NSE_TO_BSE_CODE:
        return NSE_TO_BSE_CODE[symbol], symbol

    return _search_bse_code(symbol), symbol


def _fetch_company_name(bse_code: str, fallback_name: str = "") -> str:
    """Fetch company metadata from BSE."""
    try:
        response = _bse_get(BSE_COMPANY_INFO_URL, params={"quotetype": "EQ", "scripcode": bse_code})
        payload = _parse_json_response(response)
        for key in ("CompanyName", "SecurityId", "SecurityCode"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if value and key == "CompanyName":
                return str(value).strip()

        header_response = _bse_get(BSE_SCRIP_HEADER_URL, params={"quotetype": "EQ", "scripcode": bse_code})
        header_payload = _parse_json_response(header_response)
        company = (header_payload.get("Cmpname") or {}).get("FullN") if isinstance(header_payload, dict) else None
        return str(company).strip() if company else fallback_name
    except Exception as exc:
        logger.warning("Unable to fetch BSE company info for %s: %s", bse_code, exc)
        return fallback_name


def _classify_announcement(headline: str, category: str) -> str:
    """Classify a raw announcement into FinSight's coarse types."""
    text = f"{headline} {category}".lower()
    if any(keyword in text for keyword in ("result", "financial result", "quarter", "audited", "unaudited")):
        return "results"
    if any(keyword in text for keyword in ("dividend", "record date", "book closure")):
        return "dividend"
    if any(keyword in text for keyword in ("merger", "amalgamation", "acquisition", "scheme", "arrangement")):
        return "merger"
    if any(keyword in text for keyword in ("board meeting", "investor meet", "conference call")):
        return "events"
    return "general"


def _attachment_url(name: Any) -> str:
    """Build a BSE attachment URL when a PDF attachment name is present."""
    attachment = str(name or "").strip()
    if not attachment:
        return ""
    if attachment.startswith("http"):
        return attachment
    return f"{BSE_ATTACHMENT_BASE_URL}/{attachment}"


def _normalize_bse_announcements(payload: Any, n: int, announcement_type: str) -> tuple[list[dict[str, Any]], str]:
    """Normalize BSE announcement payloads into FinSight's public schema."""
    rows = payload.get("Table", []) if isinstance(payload, dict) else []
    normalized: list[dict[str, Any]] = []
    company_name = ""

    for row in rows:
        headline = str(row.get("NEWSSUB") or row.get("HEADLINE") or "").strip()
        category = str(row.get("CATEGORYNAME") or row.get("SUBCATNAME") or "").strip()
        detected_type = _classify_announcement(headline, category)
        if announcement_type != "all" and detected_type != announcement_type:
            continue
        if not company_name:
            company_name = str(row.get("SLONGNAME") or "").strip()
        normalized.append(
            {
                "date": str(row.get("DT_TM") or row.get("NEWS_DT") or row.get("News_submission_dt") or ""),
                "headline": headline,
                "category": category or detected_type,
                "attachment_url": _attachment_url(row.get("ATTACHMENTNAME")),
                "type": detected_type,
            }
        )
        if len(normalized) >= n:
            break

    return normalized, company_name


def _normalize_nse_announcements(payload: Any, n: int, announcement_type: str) -> list[dict[str, Any]]:
    """Normalize NSE fallback announcement payloads."""
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        headline = str(row.get("desc") or row.get("subject") or row.get("attchmntText") or "").strip()
        category = str(row.get("an_dt") or row.get("sm_name") or "").strip()
        detected_type = _classify_announcement(headline, category)
        if announcement_type != "all" and detected_type != announcement_type:
            continue
        normalized.append(
            {
                "date": str(row.get("an_dt") or row.get("date") or ""),
                "headline": headline,
                "category": category or detected_type,
                "attachment_url": str(row.get("attchmntFile") or ""),
                "type": detected_type,
            }
        )
        if len(normalized) >= n:
            break
    return normalized


def _fetch_bse_announcements(bse_code: str, n: int, announcement_type: str) -> tuple[list[dict[str, Any]], str]:
    """Fetch and normalize BSE corporate announcements."""
    today = date.today()
    thirty_days_ago = today - timedelta(days=30)
    response = _bse_get(
        BSE_ANNOUNCEMENTS_URL,
        params={
            "strCat": "-1",
            "strPrevDate": thirty_days_ago.strftime("%Y%m%d"),
            "strScrip": bse_code,
            "strSearch": "P",
            "strToDate": today.strftime("%Y%m%d"),
            "strType": "C",
            "subcategory": "-1",
        },
    )
    return _normalize_bse_announcements(_parse_json_response(response), n, announcement_type)


def _fetch_nse_announcements(symbol: str | None, n: int, announcement_type: str) -> list[dict[str, Any]]:
    """Fetch and normalize NSE fallback announcements."""
    if not symbol:
        return []
    response = _nse_get(symbol)
    return _normalize_nse_announcements(_parse_json_response(response), n, announcement_type)


def _mock_announcements(ticker: str, bse_code: str | None) -> list[dict[str, Any]]:
    """Return deterministic mock announcements when BSE/NSE are unreachable."""
    today = date.today()
    return [
        {
            "date": today.isoformat(),
            "headline": f"{ticker} board meeting scheduled to consider quarterly results",
            "category": "Board Meeting",
            "attachment_url": "",
            "type": "results",
        },
        {
            "date": (today - timedelta(days=3)).isoformat(),
            "headline": f"{ticker} announces investor presentation and business update",
            "category": "Company Update",
            "attachment_url": "",
            "type": "general",
        },
        {
            "date": (today - timedelta(days=7)).isoformat(),
            "headline": f"{ticker} fixes record date for dividend related corporate action",
            "category": "Corporate Action",
            "attachment_url": "",
            "type": "dividend",
        },
    ]


def _announcement_types(raw_announcements: list[dict[str, Any]]) -> list[str]:
    """Return unique announcement types in display order."""
    seen: set[str] = set()
    found: list[str] = []
    for item in raw_announcements:
        detected = str(item.get("type") or "general")
        if detected not in seen:
            seen.add(detected)
            found.append(detected)
    return found


def _announcements_text(raw_announcements: list[dict[str, Any]]) -> str:
    """Format announcements for Claude or fallback analysis."""
    lines = []
    for index, item in enumerate(raw_announcements, start=1):
        lines.append(
            f"{index}. Date: {item.get('date')}; Category: {item.get('category')}; "
            f"Headline: {item.get('headline')}"
        )
    return "\n".join(lines) or "No recent announcements found."


def _extract_latest_quarterly_result(raw_announcements: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best-effort extraction of quarterly result metadata from headlines."""
    for item in raw_announcements:
        if item.get("type") != "results":
            continue
        headline = str(item.get("headline") or "")
        period_match = re.search(r"\bQ[1-4]\b|quarter(?:ed| ended)?\s+[A-Za-z]+\s+\d{1,2},?\s+\d{4}", headline, re.I)
        revenue_match = re.search(r"revenue[^0-9]*(\d+(?:,\d+)*(?:\.\d+)?)\s*crore", headline, re.I)
        profit_match = re.search(r"(?:profit|pat)[^0-9]*(\d+(?:,\d+)*(?:\.\d+)?)\s*crore", headline, re.I)
        yoy_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:yoy|year-on-year)", headline, re.I)
        return {
            "period": period_match.group(0) if period_match else "",
            "revenue_crore": float(revenue_match.group(1).replace(",", "")) if revenue_match else None,
            "net_profit_crore": float(profit_match.group(1).replace(",", "")) if profit_match else None,
            "yoy_change_pct": float(yoy_match.group(1)) if yoy_match else None,
        }
    return None


def _fallback_analysis(
    company_name: str,
    bse_code: str | None,
    raw_announcements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce deterministic local analysis when Claude is unavailable."""
    text = _announcements_text(raw_announcements)
    highlights = [str(item.get("headline") or "").strip() for item in raw_announcements if item.get("headline")][:5]
    while len(highlights) < 5:
        highlights.append("Recent announcements were reviewed for investor relevance")

    upcoming_events = [
        str(item.get("headline"))
        for item in raw_announcements
        if any(keyword in str(item.get("headline", "")).lower() for keyword in ("board meeting", "record date", "scheduled"))
    ][:5]

    positive_terms = len(re.findall(r"(?i)dividend|result|growth|award|contract|approval|record", text))
    concerning_terms = len(re.findall(r"(?i)delay|resignation|penalty|default|loss|investigation|downgrade", text))
    sentiment = "Positive" if positive_terms > concerning_terms + 1 else "Concerning" if concerning_terms > positive_terms else "Neutral"
    company = company_name or "the company"
    summary = (
        f"{company} has recent corporate announcements covering {', '.join(_announcement_types(raw_announcements)) or 'general updates'}. "
        "The most relevant items appear to involve results, board or investor updates, and possible corporate actions where disclosed. "
        "Investors should review the attached exchange filings for exact dates, amounts, and management commentary before acting. "
        "Overall, the announcement flow looks balanced based on the latest available exchange data."
    )
    return {
        "key_highlights": highlights[:5],
        "upcoming_events": upcoming_events,
        "sentiment": sentiment,
        "claude_summary": summary,
        "latest_quarterly_result": _extract_latest_quarterly_result(raw_announcements),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from Claude output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_analysis(payload: dict[str, Any], raw_announcements: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize Claude analysis into the public schema."""
    highlights = [str(item).strip() for item in payload.get("key_highlights", []) if str(item).strip()][:5]
    upcoming = [str(item).strip() for item in payload.get("upcoming_events", []) if str(item).strip()]
    sentiment = str(payload.get("sentiment") or "Neutral").strip()
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "Neutral"
    summary = str(payload.get("claude_summary") or "").strip()
    latest_result = payload.get("latest_quarterly_result")
    if not isinstance(latest_result, dict):
        latest_result = _extract_latest_quarterly_result(raw_announcements)
    return {
        "key_highlights": highlights,
        "upcoming_events": upcoming,
        "sentiment": sentiment,
        "claude_summary": summary,
        "latest_quarterly_result": latest_result,
    }


def _analyse_with_claude(
    company_name: str,
    bse_code: str | None,
    raw_announcements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask Claude Haiku to summarize Indian corporate announcements."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_key_here":
        return _fallback_analysis(company_name, bse_code, raw_announcements)

    announcements_text = _announcements_text(raw_announcements)
    prompt = f"""You are a financial analyst covering Indian equity markets.

Given these corporate announcements for {company_name} (BSE: {bse_code}):

{announcements_text}

Return a JSON object with EXACTLY these keys:
- key_highlights: list of exactly 5 highlights as short strings (max 15 words each), 
  prioritising quarterly results, dividends, and major corporate actions
- upcoming_events: list of strings describing upcoming events with dates if available
- sentiment: exactly one of "Positive", "Neutral", or "Concerning" based on the 
  overall tone of announcements
- claude_summary: exactly 3-4 sentences summarising what is happening with this company 
  based on recent announcements, in plain English suitable for a retail investor
- latest_quarterly_result: object with keys period (str), revenue_crore (float or null), 
  net_profit_crore (float or null), yoy_change_pct (float or null) — extract from 
  results announcements if present, otherwise null

Return ONLY valid JSON, no other text.
"""

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=700,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    return _normalize_analysis(_extract_json_object(response_text), raw_announcements)


def _mock_result(ticker: str, bse_code: str | None, company_name: str, error: str) -> dict[str, Any]:
    """Build a graceful mock response when exchanges are unavailable."""
    raw_announcements = _mock_announcements(ticker, bse_code)
    analysis = _fallback_analysis(company_name or ticker, bse_code, raw_announcements)
    return {
        **_empty_result(ticker, bse_code, company_name or ticker, error),
        "announcements_fetched": len(raw_announcements),
        "announcement_types_found": _announcement_types(raw_announcements),
        "raw_announcements": raw_announcements,
        "source": "mock",
        **analysis,
    }


def get_corporate_announcements(
    ticker: str,
    announcement_type: str = "all",
    n: int = 10,
) -> dict[str, Any]:
    """Fetch BSE/NSE corporate announcements and summarize investor-relevant items."""
    logger.info("Processing corporate announcements request for ticker=%s", ticker)

    try:
        request = CorporateAnnouncementsRequest(ticker=ticker, announcement_type=announcement_type, n=n)
    except ValidationError as exc:
        return _empty_result(ticker, None, "", exc.errors()[0]["msg"])

    cache_key = (request.ticker, request.announcement_type)
    cached = _announcement_cache.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return dict(cached[1])

    try:
        bse_code, nse_symbol = _resolve_ticker(request.ticker)
    except (httpx.HTTPError, ValueError) as exc:
        return _mock_result(
            request.ticker,
            None,
            request.ticker,
            f"BSE API is unreachable. Returning mock announcements. Details: {exc}",
        )

    if not bse_code:
        return _empty_result(
            request.ticker,
            None,
            "",
            f"Unable to resolve '{request.ticker}' to a supported BSE code.",
        )

    try:
        raw_announcements, announced_company_name = _fetch_bse_announcements(
            bse_code,
            request.n,
            request.announcement_type,
        )
        company_name = _fetch_company_name(bse_code, announced_company_name or request.ticker)
        source = "BSE"
        if not raw_announcements:
            raw_announcements = _fetch_nse_announcements(nse_symbol, request.n, request.announcement_type)
            source = "NSE" if raw_announcements else "BSE"
    except (httpx.HTTPError, ValueError) as exc:
        result = _mock_result(
            request.ticker,
            bse_code,
            request.ticker,
            f"BSE API is unreachable. Returning mock announcements. Details: {exc}",
        )
        _announcement_cache[cache_key] = (now, dict(result))
        return result

    company_name = company_name or announced_company_name or request.ticker
    analysis = _analyse_with_claude(company_name, bse_code, raw_announcements)
    result = {
        **_empty_result(request.ticker, bse_code, company_name, None),
        "announcements_fetched": len(raw_announcements),
        "announcement_types_found": _announcement_types(raw_announcements),
        "raw_announcements": raw_announcements,
        "source": source,
        **analysis,
        "error": None,
    }
    _announcement_cache[cache_key] = (now, dict(result))
    return result
