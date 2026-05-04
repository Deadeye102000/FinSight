"""Streamlit UI entrypoint for FinSight."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("FINSIGHT_API_URL", "http://localhost:8000").rstrip("/")
PRESET_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "TCS.NS", "RELIANCE.NS"]
FULL_TOOL_NAMES = {
    "get_stock_price",
    "get_fundamentals",
    "get_news_sentiment",
    "get_corporate_announcements",
    "compare_peers",
}


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


def _post(endpoint: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not body.get("success", False):
        raise RuntimeError(body.get("error") or f"{endpoint} request failed")
    return body


def _tool_data(endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return _post(endpoint, payload).get("data")
    except Exception as exc:
        return {"error": str(exc)}


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


@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(ticker: str, mode: str) -> dict[str, Any]:
    start = time.perf_counter()
    ticker = _normalize_ticker(ticker)

    price = _tool_data("/tools/price", {"ticker": ticker})
    fundamentals = _tool_data("/tools/fundamentals", {"ticker": ticker})

    if mode == "Quick":
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
            },
        }

    research_query = (
        f"Analyse {ticker} stock with a full research report. Include price and technicals, "
        "fundamentals, recent news sentiment, latest filings or corporate announcements, "
        "and peer comparison where available."
    )
    try:
        research = _post("/research", {"query": research_query}, timeout=180)
        data = research.get("data") or {}
    except Exception as exc:
        data = {
            "query": research_query,
            "tickers_mentioned": [ticker],
            "tools_called": [],
            "final_analysis": f"Research agent unavailable: {exc}",
            "confidence": "Low",
            "tokens_used": 0,
            "latency_seconds": round(time.perf_counter() - start, 2),
            "error": str(exc),
        }

    data["price_data"] = data.get("price_data") or price
    data["fundamentals"] = data.get("fundamentals") or fundamentals
    data["sentiment"] = data.get("sentiment") or _tool_data(
        "/tools/sentiment",
        {"ticker": ticker, "company_name": (fundamentals or {}).get("company_name") or ticker},
    )
    data["filing_summary"] = data.get("filing_summary") or _tool_data("/tools/filings", {"ticker": ticker})

    tools_called = set(data.get("tools_called") or [])
    tools_called.update(tool for tool in FULL_TOOL_NAMES if tool != "compare_peers")
    data["tools_called"] = sorted(tools_called)
    data["latency_seconds"] = data.get("latency_seconds") or round(time.perf_counter() - start, 2)
    data["tokens_used"] = data.get("tokens_used") or 0

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": data,
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


def render_error_note(data: dict[str, Any] | None) -> None:
    error = (data or {}).get("error")
    if error:
        st.warning(str(error), icon="⚠️")


def render_research_report(data: dict[str, Any]) -> None:
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
                    st.session_state.ticker_input = preset
                    st.rerun()

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
                st.session_state.last_ticker = ticker
                st.session_state.last_mode = mode

    result = st.session_state.get("analysis_result")
    if not result:
        render_empty_state()
        return

    data = result.get("data") or {}
    active_ticker = st.session_state.get("last_ticker", ticker)
    active_mode = st.session_state.get("last_mode", mode)

    title_col, time_col = st.columns([2, 1])
    title_col.title(f"{active_ticker} Research Dashboard")
    time_col.caption(f"Last updated: {result.get('timestamp') or datetime.now().isoformat(timespec='seconds')}")
    time_col.caption(f"Mode: {active_mode}")

    tabs = st.tabs(["📊 Research Report", "💹 Price & Technicals", "📋 Fundamentals", "📰 News Sentiment", "📁 Filing Analysis"])
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

    if active_mode == "Full":
        render_peer_comparison(active_ticker)


if __name__ == "__main__":
    main()
