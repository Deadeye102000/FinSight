"""Streamlit UI entrypoint for FinSight."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import streamlit as st


API_BASE_URL = os.getenv("FINSIGHT_API_URL", "http://localhost:8000").rstrip("/")
API_DEFAULT_TIMEOUT_SECONDS = int(os.getenv("FINSIGHT_API_TIMEOUT_SECONDS", "120"))
API_LONG_TIMEOUT_SECONDS = int(os.getenv("FINSIGHT_API_LONG_TIMEOUT_SECONDS", "300"))
API_RETRY_COUNT = int(os.getenv("FINSIGHT_API_RETRY_COUNT", "2"))
API_RETRY_BACKOFF_SECONDS = float(os.getenv("FINSIGHT_API_RETRY_BACKOFF_SECONDS", "1.0"))

retry_strategy = Retry(
    total=API_RETRY_COUNT,
    connect=API_RETRY_COUNT,
    read=API_RETRY_COUNT,
    status=API_RETRY_COUNT,
    status_forcelist=[429, 502, 503, 504],
    allowed_methods=frozenset(["HEAD", "GET", "OPTIONS", "POST"]),
    backoff_factor=API_RETRY_BACKOFF_SECONDS,
    raise_on_status=False,
)
_API_SESSION = requests.Session()
_API_SESSION.mount("http://", HTTPAdapter(max_retries=retry_strategy))
_API_SESSION.mount("https://", HTTPAdapter(max_retries=retry_strategy))

PRESET_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "TCS.NS", "RELIANCE.NS"]
INDIAN_DEMO_TICKERS = [
    "TCS.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "WIPRO.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "ONGC.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "TECHM.NS",
    "HCLTECH.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "ADANIPORTS.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "COALINDIA.NS",
]
US_DEMO_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "JPM",
    "V",
    "MA",
    "UNH",
    "JNJ",
    "WMT",
    "COST",
    "XOM",
    "CVX",
    "CRM",
    "INTC",
]
PEER_GUIDE = [
    {"Main ticker": "AAPL", "Peers": "MSFT, GOOGL, AMZN"},
    {"Main ticker": "MSFT", "Peers": "AAPL, GOOGL, AMZN"},
    {"Main ticker": "NVDA", "Peers": "AMD, INTC, MSFT"},
    {"Main ticker": "TSLA", "Peers": "F, GM, RIVN"},
    {"Main ticker": "TCS.NS", "Peers": "INFY.NS, WIPRO.NS, HCLTECH.NS, TECHM.NS"},
    {"Main ticker": "HDFCBANK.NS", "Peers": "ICICIBANK.NS, AXISBANK.NS, KOTAKBANK.NS, SBIN.NS"},
    {"Main ticker": "RELIANCE.NS", "Peers": "ONGC.NS, NTPC.NS, POWERGRID.NS, COALINDIA.NS"},
    {"Main ticker": "SUNPHARMA.NS", "Peers": "DRREDDY.NS, CIPLA.NS, DIVISLAB.NS"},
]
FULL_TOOL_NAMES = {
    "get_stock_price",
    "get_fundamentals",
    "get_news_sentiment",
    "get_corporate_announcements",
    "compare_peers",
}
DASHBOARD_TABS = [
    "📊 Research Report",
    "💹 Price & Technicals",
    "📋 Fundamentals",
    "📰 News Sentiment",
    "📁 Filing Analysis",
    "🤖 Doc Q&A",
]


st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide")


def _theme_option(name: str, default: str = "") -> str:
    """Read Streamlit theme options without assuming a specific Streamlit version."""
    theme = getattr(st, "theme", {}) or {}
    if isinstance(theme, dict) and theme.get(name):
        return str(theme[name])
    return str(st.get_option(f"theme.{name}") or default)


THEME_PRIMARY_COLOR = _theme_option("primaryColor", "primary")
DEFAULT_BADGE_COLOR = "primary" if THEME_PRIMARY_COLOR else "blue"


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _format_number(value: Any, suffix: str = "", decimals: int = 2) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:,.{decimals}f}{suffix}"


def _format_money(value: Any, currency: str = "") -> str:
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{_format_number(value)}"


def _request(
    method: str,
    endpoint: str | None = None,
    url: str | None = None,
    timeout: int | None = None,
    **kwargs: Any,
) -> requests.Response:
    if endpoint is not None:
        url = f"{API_BASE_URL}{endpoint}"
    if not url:
        raise ValueError("Either endpoint or url must be provided")

    effective_timeout = timeout or API_DEFAULT_TIMEOUT_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            response = _API_SESSION.request(method, url, timeout=effective_timeout, **kwargs)
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt > API_RETRY_COUNT:
                raise RuntimeError(f"Request timed out after {effective_timeout}s for {url}") from exc
            time.sleep(API_RETRY_BACKOFF_SECONDS * attempt)
            continue
        except requests.HTTPError as exc:
            if exc.response is not None:
                raise RuntimeError(
                    f"Request failed for {url}: {exc.response.status_code} {exc.response.text}"
                ) from exc
            raise RuntimeError(f"Request failed for {url}: {exc}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def _post(endpoint: str, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
    response = _request("POST", endpoint=endpoint, json=payload, timeout=timeout)
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected response body from {endpoint}")
    if not body.get("success", False):
        raise RuntimeError(body.get("error") or f"{endpoint} request failed")
    return body


@st.cache_data(ttl=60, show_spinner=False)
def fetch_rag_documents() -> list[str]:
    response = _request("GET", endpoint="/rag/documents", timeout=30)
    return response.json().get("documents") or []


def _tool_data(endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return _post(endpoint, payload).get("data")
    except Exception as exc:
        return {"error": str(exc)}


def _start_research_thread(ticker: str, research_query: str) -> None:
    thread = st.session_state.get("research_thread")
    current_ticker = st.session_state.get("research_ticker")
    if thread and getattr(thread, "is_alive", lambda: False)() and current_ticker == ticker:
        return

    def _research_task() -> None:
        try:
            research = _post("/research", {"query": research_query}, timeout=API_LONG_TIMEOUT_SECONDS)
            st.session_state.research_result = research.get("data") or {}
            st.session_state.research_complete = True
            st.session_state.research_in_progress = False
            st.session_state.research_error = None
            st.session_state.research_started_at = datetime.now().isoformat(timespec="seconds")
        except Exception as exc:
            st.session_state.research_result = {
                "final_analysis": f"Research agent unavailable: {exc}",
                "confidence": "Low",
                "error": str(exc),
                "tools_called": [],
                "price_data": None,
                "fundamentals": None,
                "sentiment": None,
                "filing_summary": None,
                "peer_comparison": None,
            }
            st.session_state.research_complete = True
            st.session_state.research_in_progress = False
            st.session_state.research_error = str(exc)

    thread = threading.Thread(target=_research_task, daemon=True)
    thread.start()
    st.session_state.research_thread = thread
    st.session_state.research_ticker = ticker
    st.session_state.research_query = research_query
    st.session_state.research_in_progress = True
    st.session_state.research_complete = False


def _merge_analysis_results(partial: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
    merged = {**partial}
    merged["tools_called"] = sorted(
        set((partial.get("tools_called") or []) + (research.get("tools_called") or []))
    )
    merged["price_data"] = research.get("price_data") or partial.get("price_data")
    merged["fundamentals"] = research.get("fundamentals") or partial.get("fundamentals")
    merged["sentiment"] = research.get("sentiment") or partial.get("sentiment")
    merged["filing_summary"] = research.get("filing_summary") or partial.get("filing_summary")
    merged["peer_comparison"] = research.get("peer_comparison") or partial.get("peer_comparison")
    merged["final_analysis"] = research.get("final_analysis") or partial.get("final_analysis")
    merged["confidence"] = research.get("confidence") or partial.get("confidence")
    merged["latency_seconds"] = research.get("latency_seconds") or partial.get("latency_seconds")
    merged["tokens_used"] = research.get("tokens_used") or partial.get("tokens_used")
    if research.get("error"):
        merged["error"] = research["error"]
    merged["research_status"] = "complete"
    return merged


def _fetch_core_tool_data(ticker: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        price_future = executor.submit(_tool_data, "/tools/price", {"ticker": ticker})
        fundamentals_future = executor.submit(_tool_data, "/tools/fundamentals", {"ticker": ticker})
        return price_future.result(), fundamentals_future.result()


def _fetch_auxiliary_tool_data(ticker: str, company_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        sentiment_future = executor.submit(
            _tool_data,
            "/tools/sentiment",
            {"ticker": ticker, "company_name": company_name},
        )
        filing_future = executor.submit(_tool_data, "/tools/filings", {"ticker": ticker})
        return sentiment_future.result(), filing_future.result()


def _quick_report(ticker: str, price: dict[str, Any] | None, fundamentals: dict[str, Any] | None) -> str:
    price = price or {}
    fundamentals = fundamentals or {}
    company = fundamentals.get("company_name") or ticker
    valuation = fundamentals.get("valuation_summary") or "Insufficient data"
    current_price = _format_money(price.get("current_price"), price.get("currency"))
    change = _format_number(price.get("price_change_pct_30d"), "%")
    pe_ratio = _format_number(fundamentals.get("pe_ratio"))
    roe = _format_number(fundamentals.get("roe"), "%")
    rsi = _format_number(price.get("rsi_14"))
    macd = str(price.get("macd_signal") or "neutral").title()

    return f"""## {company} ({ticker}) Quick Research

**Executive Summary:** {ticker} is currently trading at **{current_price}**, with a 30-day move of **{change}**. The available valuation view is **{valuation}**.

**Technical Picture:** RSI is **{rsi}** and MACD is **{macd}**, giving a compact read on recent momentum.

**Fundamental Snapshot:** P/E is **{pe_ratio}** and ROE is **{roe}**, based on the latest available fundamentals.

**Conclusion:** Use this as a fast first pass before deeper research. Not financial advice.
"""


def run_analysis(ticker: str, mode: str) -> dict[str, Any]:
    start = time.perf_counter()
    ticker = _normalize_ticker(ticker)

    if mode == "Quick":
        price = _tool_data("/tools/price", {"ticker": ticker})
        fundamentals = _tool_data("/tools/fundamentals", {"ticker": ticker})
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data": {
                "query": f"Quick research for {ticker}",
                "tickers_mentioned": [ticker],
                "tools_called": ["get_stock_price", "get_fundamentals"],
                "price_data": price,
                "fundamentals": fundamentals,
                "sentiment": None,
                "filing_summary": None,
                "peer_comparison": None,
                "final_analysis": _quick_report(ticker, price, fundamentals),
                "confidence": "High" if not (price or {}).get("error") and not (fundamentals or {}).get("error") else "Medium",
                "tokens_used": 0,
                "latency_seconds": round(time.perf_counter() - start, 2),
                "research_status": "complete",
            },
        }

    price, fundamentals = _fetch_core_tool_data(ticker)
    company_name = (fundamentals or {}).get("company_name") or ticker
    sentiment, filing_summary = _fetch_auxiliary_tool_data(ticker, company_name)

    research_query = (
        f"Analyse {ticker} stock with a full research report. Include price and technicals, "
        "fundamentals, recent news sentiment, latest filings or corporate announcements, "
        "and peer comparison where available."
    )
    _start_research_thread(ticker, research_query)

    partial_data = {
        "query": research_query,
        "tickers_mentioned": [ticker],
        "tools_called": ["get_stock_price", "get_fundamentals", "get_news_sentiment", "get_corporate_announcements"],
        "price_data": price,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "filing_summary": filing_summary,
        "peer_comparison": None,
        "final_analysis": (
            "Partial research is ready. Full agent synthesis is still running in the background. "
            "Refresh the report when the full synthesis completes."
        ),
        "confidence": "Medium",
        "tokens_used": 0,
        "latency_seconds": round(time.perf_counter() - start, 2),
        "research_status": "pending",
    }

    if (
        st.session_state.get("research_ticker") == ticker
        and st.session_state.get("research_complete")
        and st.session_state.get("research_result")
    ):
        merged = _merge_analysis_results(partial_data, st.session_state.research_result)
        merged["research_status"] = "complete"
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data": merged,
        }

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": partial_data,
    }


@st.cache_data(ttl=300, show_spinner=False)
def run_peer_comparison(ticker: str, peers: tuple[str, ...]) -> dict[str, Any]:
    if not peers:
        return {}
    return _post("/tools/peers", {"ticker": ticker, "peers": list(peers)}).get("data") or {}


def render_badge(label: str, color: str = DEFAULT_BADGE_COLOR) -> None:
    if hasattr(st, "badge"):
        st.badge(label, color=color)
    else:
        st.markdown(f"`{label}`")


def render_status_badge(label: str) -> None:
    normalized = (label or "Unknown").strip().lower()
    if normalized in {"high", "positive", "undervalued", "bullish"}:
        render_badge(label, "green")
    elif normalized in {"medium", "neutral", "fairly valued"}:
        render_badge(label, "yellow")
    elif normalized in {"low", "negative", "overvalued", "bearish", "concerning"}:
        render_badge(label, "red")
    else:
        render_badge(label, "gray")


def render_ticker_peer_guide() -> None:
    with st.sidebar.expander("Ticker & Peer Guide"):
        st.caption("Best-supported demo tickers")
        st.markdown("**India**")
        st.code(", ".join(INDIAN_DEMO_TICKERS[:10]))
        st.markdown("**US**")
        st.code(", ".join(US_DEMO_TICKERS[:10]))

        st.caption("Peer combinations")
        st.dataframe(pd.DataFrame(PEER_GUIDE), width="stretch", hide_index=True)

        st.caption("NSE expansion plan")
        st.markdown(
            "- Ingest NSE's official equity symbol CSV.\n"
            "- Append `.NS` for Yahoo Finance price/fundamentals.\n"
            "- Group companies by sector or industry for peer suggestions.\n"
            "- Cache the ticker universe and refresh it daily or weekly."
        )


def render_error_note(data: dict[str, Any] | None) -> None:
    error = (data or {}).get("error")
    if error:
        st.warning(str(error), icon="⚠️")


def _safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Streamlit rerun is not available in this version. Please refresh the page manually.")


def render_research_report(data: dict[str, Any]) -> None:
    is_pending = data.get("research_status") == "pending"
    background_done = st.session_state.get("research_complete") is True
    if is_pending:
        status_text = (
            "Partial evidence is available while the full research synthesis is still running. "
            "Press refresh to load the final report once it completes."
        )
        if background_done:
            status_text = (
                "Background research is complete. "
                "Please refresh the report to view the final synthesis."
            )
        st.warning(status_text, icon="⏳")
        if st.button("Refresh full report", key="refresh_full_report"):
            _safe_rerun()
    elif data.get("research_status") == "complete":
        st.success("Full research synthesis is complete.", icon="✅")

    st.markdown(data.get("final_analysis") or "No research report returned yet.")

    st.divider()
    st.caption("Tools called")
    tool_cols = st.columns(min(max(len(data.get("tools_called") or []), 1), 5))
    for index, tool in enumerate(data.get("tools_called") or ["No tools reported"]):
        with tool_cols[index % len(tool_cols)]:
            render_badge(str(tool).replace("_", " "), "primary")

    confidence = data.get("confidence") or "Unknown"
    st.caption("Confidence")
    render_status_badge(confidence)
    st.caption(
        f"Latency: {_format_number(data.get('latency_seconds'), 's')} | "
        f"Tokens used: {_format_number(data.get('tokens_used'), decimals=0)}"
    )


def render_price_tab(price: dict[str, Any] | None) -> None:
    price = price or {}
    render_error_note(price)
    currency = price.get("currency") or ""

    metric_cols = st.columns(3)
    metric_cols[0].metric("Current Price", _format_money(price.get("current_price"), currency))
    metric_cols[1].metric("30-Day Change", _format_number(price.get("price_change_pct_30d"), "%"), delta=price.get("price_change_pct_30d"))
    metric_cols[2].metric("Avg Volume 10D", _format_number(price.get("avg_volume_10d"), decimals=0))

    rsi = float(price.get("rsi_14") or 0)
    macd = str(price.get("macd_signal") or "neutral").title()
    low_52w = float(price.get("low_52w") or 0)
    high_52w = float(price.get("high_52w") or 0)
    current = float(price.get("current_price") or 0)

    rsi_col, macd_col = st.columns(2)
    with rsi_col:
        st.subheader("RSI")
        st.progress(min(max(rsi / 100, 0), 1), text=f"{rsi:.2f}")
        if rsi > 70:
            st.error("Overbought", icon="📈")
        elif rsi < 30:
            st.success("Oversold", icon="📉")
        else:
            st.warning("Neutral range", icon="➖")

    with macd_col:
        st.subheader("MACD Signal")
        render_status_badge(macd)
        st.caption(f"50 DMA: {_format_number(price.get('ma_50'))} | 200 DMA: {_format_number(price.get('ma_200'))}")

    st.subheader("52-Week Range")
    range_position = 0.0 if high_52w == low_52w else (current - low_52w) / (high_52w - low_52w)
    st.progress(min(max(range_position, 0), 1), text=f"{_format_money(current, currency)}")
    low_col, high_col = st.columns(2)
    low_col.caption(f"Low: {_format_money(low_52w, currency)}")
    high_col.caption(f"High: {_format_money(high_52w, currency)}")

    st.subheader("OHLCV - Last 5 Days")
    st.dataframe(pd.DataFrame(price.get("ohlcv_last_5") or []), width="stretch", hide_index=True)


def render_fundamentals_tab(fundamentals: dict[str, Any] | None) -> None:
    fundamentals = fundamentals or {}
    render_error_note(fundamentals)

    metrics = [
        ("P/E", fundamentals.get("pe_ratio"), ""),
        ("P/B", fundamentals.get("pb_ratio"), ""),
        ("ROE", fundamentals.get("roe"), "%"),
        ("Debt/Equity", fundamentals.get("debt_to_equity"), ""),
        ("Gross Margin", fundamentals.get("gross_margin"), "%"),
        ("Net Margin", fundamentals.get("net_margin"), "%"),
    ]
    for row in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (label, value, suffix) in zip(cols, metrics[row : row + 3], strict=False):
            col.metric(label, _format_number(value, suffix))

    valuation = fundamentals.get("valuation_summary") or "Insufficient data"
    st.caption("Valuation Summary")
    render_status_badge(valuation)

    analyst_cols = st.columns(2)
    analyst_cols[0].metric("Analyst Rating", fundamentals.get("analyst_rating") or "N/A")
    analyst_cols[1].metric("Mean Target Price", _format_money(fundamentals.get("target_price_mean"), fundamentals.get("currency") or ""))


def render_sentiment_tab(sentiment: dict[str, Any] | None) -> None:
    sentiment = sentiment or {}
    render_error_note(sentiment)
    overall = sentiment.get("overall_sentiment") or "Neutral"
    score = float(sentiment.get("sentiment_score") or 0)

    st.subheader("Overall Sentiment")
    if overall == "Positive":
        st.success(f"Positive ({score:+.2f})", icon="✅")
    elif overall == "Negative":
        st.error(f"Negative ({score:+.2f})", icon="⚠️")
    else:
        st.info(f"Neutral ({score:+.2f})", icon="ℹ️")

    st.progress(min(max((score + 1) / 2, 0), 1), text=f"Sentiment score: {score:+.2f}")

    top_cols = st.columns(2)
    top_positive = sentiment.get("top_positive_headline")
    top_negative = sentiment.get("top_negative_headline")
    with top_cols[0]:
        st.caption("Top Positive")
        st.success(top_positive or "No positive headline identified.")
    with top_cols[1]:
        st.caption("Top Negative")
        st.error(top_negative or "No negative headline identified.")

    headlines = pd.DataFrame(sentiment.get("headlines") or [])
    if not headlines.empty:
        headlines["sentiment_badge"] = headlines["sentiment"].map(
            {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}
        ).fillna("Neutral")
        st.dataframe(
            headlines[["headline", "sentiment_badge", "score", "source", "published_at"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No headlines returned.")


def render_filing_tab(filing: dict[str, Any] | None) -> None:
    filing = filing or {}
    render_error_note(filing)
    latest = filing.get("latest_quarterly_result") or {}
    raw = filing.get("raw_announcements") or []
    filed_date = raw[0].get("date") if raw else None
    sec_link = raw[0].get("attachment_url") if raw else None

    meta_cols = st.columns(2)
    meta_cols[0].metric("Filed Date", filed_date or "N/A")
    meta_cols[1].metric("Period", latest.get("period") or "N/A")

    highlight_terms = filing.get("key_highlights") or []
    risk_terms = [
        item.get("headline") if isinstance(item, dict) else str(item)
        for item in raw
        if any(term in str(item).lower() for term in ("risk", "loss", "delay", "penalty", "default", "resignation", "pressure"))
    ][:5]

    risk_col, highlight_col = st.columns(2)
    with risk_col:
        st.subheader("Key Risks")
        if risk_terms:
            for risk in risk_terms:
                st.error(f"• {risk}")
        else:
            st.info("No explicit risk headlines detected in the latest filing feed.")
    with highlight_col:
        st.subheader("Key Highlights")
        if highlight_terms:
            for highlight in highlight_terms[:5]:
                st.success(f"• {highlight}")
        else:
            st.info("No highlights returned.")

    st.info(filing.get("claude_summary") or "No filing summary returned.")
    if sec_link:
        st.link_button("Open filing", sec_link)


def _peer_table_with_best_flags(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for metric, ascending in {"pe_ratio": True, "roe": False, "net_margin": False}.items():
        if metric in table:
            numeric = pd.to_numeric(table[metric], errors="coerce")
            if numeric.notna().any():
                best = numeric.min() if ascending else numeric.max()
                table[f"{metric}_best"] = numeric.eq(best).map({True: "Best", False: ""})
    return table


def render_peer_comparison(ticker: str) -> None:
    st.divider()
    st.header("Peer Comparison")
    peer_input = st.text_input("Peer tickers", value="MSFT, GOOGL, AMZN", help="Use 2 to 5 comma-separated tickers.")
    peers = tuple(_normalize_ticker(peer) for peer in peer_input.split(",") if _normalize_ticker(peer))

    if not peers:
        st.info("Add at least two peers to compare valuation and quality.")
        return

    try:
        with st.spinner("Comparing peers..."):
            peer_data = run_peer_comparison(ticker, peers)
    except Exception as exc:
        st.error(f"Peer comparison failed: {exc}")
        return

    render_error_note(peer_data)
    rows = peer_data.get("summary_table") or []
    table = _peer_table_with_best_flags(rows)
    if table.empty:
        st.info("No peer data returned.")
        return

    chart_cols = st.columns(2)
    chart_data = table.set_index("ticker")
    with chart_cols[0]:
        st.subheader("P/E Ratios")
        st.bar_chart(chart_data[["pe_ratio"]])
    with chart_cols[1]:
        st.subheader("ROE")
        st.bar_chart(chart_data[["roe"]])

    st.subheader("Summary")
    columns = [
        column
        for column in ["ticker", "company_name", "pe_ratio", "pe_ratio_best", "roe", "roe_best", "pb_ratio", "debt_to_equity", "valuation_summary"]
        if column in table.columns
    ]
    st.dataframe(table[columns], width="stretch", hide_index=True)
    st.caption("Best cells are marked in the adjacent status columns for theme-compatible highlighting.")

    winner = peer_data.get("winner_overall") or ticker
    st.caption("Winner")
    render_status_badge(winner)
    if peer_data.get("relative_valuation"):
        st.info(peer_data["relative_valuation"])


def render_doc_qa_tab() -> None:
    st.subheader("Upload Annual Report or Earnings PDF")

    file = st.file_uploader("Choose a PDF", type=["pdf"])
    default_doc_id = file.name.replace(".pdf", "").replace(" ", "_") if file else ""
    doc_id = st.text_input(
        "Document ID",
        value=default_doc_id,
        help="Alphanumeric and underscores only. E.g. tcs_annual_2024",
    )

    if st.button("Ingest Document") and file and doc_id:
        with st.spinner("📄 Extracting and embedding document..."):
            try:
                response = _request(
                    "POST",
                    url=f"{API_BASE_URL}/rag/ingest",
                    data={"doc_id": doc_id},
                    files={"file": (file.name, file.getvalue(), "application/pdf")},
                    timeout=API_LONG_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
            else:
                data = response.json()
                fetch_rag_documents.clear()
                st.success(f"✅ Ready. {data['pages_extracted']} pages, {data['chunks_created']} chunks.")
                if data.get("already_existed"):
                    st.info("This document was already ingested. Using cached version.")

    try:
        docs = fetch_rag_documents()
    except Exception as exc:
        st.error(f"Could not load documents: {exc}")
        docs = []

    if not docs:
        st.info(
            """Upload an annual report PDF above to get started.
Try the Reliance Industries FY2024 or TCS Q4 results PDF
from their investor relations pages."""
        )
        return

    st.subheader("Ask a Question")
    selected_doc = st.selectbox("Select document", docs)
    question = st.text_area(
        "Your question",
        placeholder="\n".join(
            [
                "What was the revenue for Q4?",
                "What risks did management highlight?",
                "What is the dividend policy?",
                "How did margins change year over year?",
            ]
        ),
        height=100,
    )
    k = st.slider("Sources to retrieve", 1, 10, 5)

    if st.button("Ask") and question:
        with st.spinner("🔍 Searching document..."):
            try:
                response = _request(
                    "POST",
                    url=f"{API_BASE_URL}/rag/query",
                    json={"doc_id": selected_doc, "question": question, "k": k},
                    timeout=API_LONG_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
            else:
                data = response.json()
                st.markdown("### Answer")
                st.info(data["answer"])

                with st.expander(f"📚 Sources ({len(data['sources'])} chunks used)"):
                    for i, src in enumerate(data["sources"], 1):
                        st.markdown(
                            f"**Source {i}** — Pages {src['page_start']}–"
                            f"{src['page_end']} "
                            f"(similarity: {src['similarity']:.2f})"
                        )
                        st.caption(src["excerpt_preview"] + "...")

                st.caption(
                    f"Model: {data['model_used']} · "
                    f"Tokens: {data['tokens_used']} · "
                    f"Chunks retrieved: {data['chunks_used']}"
                )

    st.subheader("Manage Documents")
    for doc in docs:
        col1, col2 = st.columns([4, 1])
        col1.text(doc)
        if col2.button("Delete", key=f"del_{doc}"):
            try:
                _request("DELETE", endpoint=f"/rag/documents/{doc}", timeout=30)
            except Exception as exc:
                st.error(f"Could not delete {doc}: {exc}")
            else:
                fetch_rag_documents.clear()
                st.rerun()


def render_empty_state() -> None:
    st.title("FinSight")
    st.caption("AI-powered stock research in seconds")
    st.info("Enter a ticker in the sidebar and click Analyse.")


def render_sidebar() -> tuple[str, str, bool]:
    with st.sidebar:
        st.title("📈 FinSight")
        st.caption("AI-powered stock research in seconds")

        if "ticker_input" not in st.session_state:
            st.session_state.ticker_input = "AAPL"
        if "preset_ticker" in st.session_state:
            st.session_state.ticker_input = st.session_state.pop("preset_ticker")

        ticker = st.text_input(
            "Stock ticker",
            key="ticker_input",
            placeholder="AAPL, RELIANCE.NS, TCS.NS",
        )

        st.caption("Quick presets")
        for row in range(0, len(PRESET_TICKERS), 3):
            cols = st.columns(3)
            for col, preset in zip(cols, PRESET_TICKERS[row : row + 3], strict=False):
                if col.button(preset, width="stretch"):
                    st.session_state.preset_ticker = preset
                    st.rerun()

        render_ticker_peer_guide()

        mode = st.radio("Research mode", ["Quick", "Full"], horizontal=True)
        analyse = st.button("Analyse", type="primary", width="stretch")
        if st.button("Clear cache", width="stretch"):
            st.cache_data.clear()
            st.success("Cache cleared.")

        st.caption(f"API: {API_BASE_URL}")
    return _normalize_ticker(ticker), mode, analyse


def main() -> None:
    ticker, mode, analyse = render_sidebar()

    if analyse:
        if not ticker:
            st.error("Please enter a ticker.")
        else:
            with st.spinner(f"🔍 Researching {ticker}..."):
                st.session_state.analysis_result = run_analysis(ticker, mode)
                st.session_state.analysis_ticker = ticker
                st.session_state.analysis_mode = mode
                st.session_state.last_ticker = ticker
                st.session_state.last_mode = mode

    if (
        st.session_state.get("analysis_result")
        and st.session_state.get("analysis_ticker") == ticker
        and st.session_state.get("analysis_mode") == mode
        and st.session_state.get("research_complete")
        and st.session_state.analysis_result.get("data", {}).get("research_status") != "complete"
    ):
        st.session_state.analysis_result = run_analysis(ticker, mode)

    result = st.session_state.get("analysis_result")
    if not result:
        render_empty_state()
        tabs = st.tabs(DASHBOARD_TABS)
        with tabs[0]:
            st.info("Run a stock analysis from the sidebar to populate the research dashboard.")
        with tabs[1]:
            st.info("Price and technical data will appear here after analysis.")
        with tabs[2]:
            st.info("Fundamentals will appear here after analysis.")
        with tabs[3]:
            st.info("News sentiment will appear here after analysis.")
        with tabs[4]:
            st.info("Filing analysis will appear here after analysis.")
        with tabs[5]:
            render_doc_qa_tab()
        return

    data = result.get("data") or {}
    active_ticker = st.session_state.get("last_ticker", ticker)
    active_mode = st.session_state.get("last_mode", mode)

    title_col, time_col = st.columns([2, 1])
    title_col.title(f"{active_ticker} Research Dashboard")
    time_col.caption(f"Last updated: {result.get('timestamp') or datetime.now().isoformat(timespec='seconds')}")
    time_col.caption(f"Mode: {active_mode}")

    tabs = st.tabs(DASHBOARD_TABS)
    with tabs[0]:
        render_research_report(data)
    with tabs[1]:
        render_price_tab(data.get("price_data"))
    with tabs[2]:
        render_fundamentals_tab(data.get("fundamentals"))
    with tabs[3]:
        render_sentiment_tab(data.get("sentiment"))
    with tabs[4]:
        render_filing_tab(data.get("filing_summary"))
    with tabs[5]:
        render_doc_qa_tab()

    if active_mode == "Full":
        render_peer_comparison(active_ticker)


if __name__ == "__main__":
    main()
