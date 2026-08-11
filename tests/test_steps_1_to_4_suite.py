"""Comprehensive test suite for steps 1-4: MCP tools, FinBERT sentiment, LangGraph pipeline, and RAG retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from finsight.mcp_server.tools.filings import get_filing_text
from finsight.mcp_server.tools.peers import get_peer_comparison
from finsight.mcp_server.tools.sentiment import get_news_sentiment
from finsight.orchestrator import pipeline
from finsight.orchestrator.pipeline import run_research_pipeline, sentiment_analysis_node
from finsight.orchestrator.state import ResearchState
from finsight.rag.pipeline import RAGPipeline
from finsight.rag.retriever import retrieve_relevant_filing_context

# ---------------------------------------------------------------------------
# 1. MCP Tools Unit Tests (Mock External API Calls)
# ---------------------------------------------------------------------------


def test_mcp_get_news_sentiment_mocked() -> None:
    """Unit test for get_news_sentiment tool with mocked yfinance news."""
    mock_news = [
        {
            "content": {
                "title": "Mock Tech Stocks Rally",
                "pubDate": "2026-08-11T12:00:00Z",
                "provider": {"displayName": "Mock Wire"},
                "canonicalUrl": {"url": "https://example.com/news1"},
            }
        }
    ]

    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_instance = MagicMock()
        mock_instance.news = mock_news
        mock_ticker_class.return_value = mock_instance

        res = get_news_sentiment("AAPL", days_back=7)

        assert res["ticker"] == "AAPL"
        assert res["days_back"] == 7
        assert res["headline_count"] == 1
        assert res["headlines"][0]["title"] == "Mock Tech Stocks Rally"
        assert res["headlines"][0]["publisher"] == "Mock Wire"
        assert res["headlines"][0]["url"] == "https://example.com/news1"
        assert res["error"] is None


def test_mcp_get_peer_comparison_mocked() -> None:
    """Unit test for get_peer_comparison tool with mocked price and fundamentals tools."""
    mock_price = {"current_price": 200.0, "currency": "USD", "rsi_14": 55.0, "error": None}
    mock_fund = {
        "company_name": "Tech Corp",
        "currency": "USD",
        "reporting_currency": "USD",
        "market_cap": 3000.0,
        "pe_ratio": 30.0,
        "revenue_growth": 8.0,
        "valuation_summary": "Fairly Valued",
        "error": None,
    }

    with patch("finsight.mcp_server.tools.peers.get_stock_price", return_value=mock_price), patch(
        "finsight.mcp_server.tools.peers.get_fundamentals", return_value=mock_fund
    ):
        res = get_peer_comparison("AAPL", peer_tickers=["MSFT", "GOOGL"])

        assert res["main_ticker"] == "AAPL"
        assert res["peer_tickers"] == ["MSFT", "GOOGL"]
        assert len(res["comparison_table"]) == 3
        assert res["comparison_table"][0]["ticker"] == "AAPL"
        assert res["error"] is None


def test_mcp_get_filing_text_mocked() -> None:
    """Unit test for get_filing_text tool with mocked BSE API response."""
    mock_bse_disclosures = [
        {
            "NEWSSUB": "Reg. 34 (1) Annual Report for FY2025-26.",
            "CATEGORYNAME": "Reg. 34 (1) Annual Report.",
            "DT_TM": "2026-05-15T18:00:00.000",
            "ATTACHMENTNAME": "mock_annual_report.pdf",
            "MORE": "Tata Consultancy Services Limited Annual Report submission.",
            "SLONGNAME": "Tata Consultancy Services Limited",
        }
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"Table": mock_bse_disclosures}

    with patch("httpx.get", return_value=mock_response), patch(
        "finsight.mcp_server.tools.filings._resolve_bse_code", return_value=("532540", "TCS")
    ):
        res = get_filing_text("TCS.NS", filing_type="annual_report")

        assert res["ticker"] == "TCS.NS"
        assert res["bse_code"] == "532540"
        assert res["company_name"] == "Tata Consultancy Services Limited"
        assert res["filing_type"] == "annual_report"
        assert res["reporting_currency"] == "INR"
        assert res["normalized_currency"] == "USD"
        assert "Reg. 34 (1) Annual Report" in res["filing_text"]
        assert res["error"] is None


# ---------------------------------------------------------------------------
# 2. FinBERT Sentiment Node Unit Tests (Primary & Fallback)
# ---------------------------------------------------------------------------


def test_sentiment_analysis_node_finbert_success() -> None:
    """Unit test for sentiment_analysis_node using local FinBERT classifier."""
    mock_classifier = MagicMock()
    mock_classifier.return_value = [
        {"label": "positive", "score": 0.95},
        {"label": "negative", "score": 0.85},
    ]

    with patch("finsight.orchestrator.pipeline.load_finbert_model", return_value=mock_classifier):
        state = ResearchState(
            ticker="AAPL",
            raw_news={
                "headlines": [
                    {"title": "Apple reports record revenue beat", "publisher": "Bloomberg"},
                    {"title": "Supply chain issues hamper production", "publisher": "Reuters"},
                ]
            },
        )
        res = sentiment_analysis_node(state)
        sent = res["sentiment_analysis"]

        assert sent["mode"] == "finbert"
        assert sent["headline_count"] == 2
        assert sent["positive_count"] == 1
        assert sent["negative_count"] == 1
        assert len(sent["per_headline_details"]) == 2
        assert sent["per_headline_details"][0]["sentiment"] == "Positive"
        assert sent["per_headline_details"][0]["score"] == 0.95
        assert sent["per_headline_details"][1]["sentiment"] == "Negative"
        assert sent["per_headline_details"][1]["score"] == -0.85


def test_sentiment_analysis_node_lexicon_fallback() -> None:
    """Unit test for sentiment_analysis_node gracefully falling back to lexicon mode on model error."""
    with patch("finsight.orchestrator.pipeline.load_finbert_model", side_effect=RuntimeError("GPU OOM")):
        state = ResearchState(
            ticker="AAPL",
            raw_news={
                "headlines": [
                    {"title": "Company reports record profit growth and dividend surge", "publisher": "News"},
                    {"title": "Firm faces lawsuit and revenue drop warning", "publisher": "News"},
                ]
            },
        )
        res = sentiment_analysis_node(state)
        sent = res["sentiment_analysis"]

        assert sent["mode"] == "lexicon_fallback"
        assert sent["headline_count"] == 2
        assert sent["positive_count"] == 1
        assert sent["negative_count"] == 1
        assert sent["per_headline_details"][0]["sentiment"] == "Positive"
        assert sent["per_headline_details"][1]["sentiment"] == "Negative"


# ---------------------------------------------------------------------------
# 3. LangGraph Pipeline End-to-End Integration Test
# ---------------------------------------------------------------------------


def test_langgraph_pipeline_end_to_end_mocked() -> None:
    """Integration test verifying full LangGraph StateGraph pipeline execution with mocked tool calls."""
    mock_price = {"current_price": 180.0, "currency": "USD", "rsi_14": 52.0, "macd_signal": "bullish", "error": None}
    mock_fund = {
        "company_name": "Apple Inc.",
        "currency": "USD",
        "reporting_currency": "USD",
        "market_cap": 2800.0,
        "pe_ratio": 28.5,
        "gross_margin": 45.0,
        "net_margin": 25.0,
        "roe": 30.0,
        "valuation_summary": "Fairly Valued",
        "error": None,
    }
    mock_news = {
        "ticker": "AAPL",
        "days_back": 7,
        "headline_count": 1,
        "headlines": [{"title": "Apple launches new AI expansion", "publisher": "Wire", "published_at": "2026-08-11", "url": "https://example.com"}],
        "error": None,
    }
    mock_peers = {
        "main_ticker": "AAPL",
        "peer_tickers": ["MSFT"],
        "comparison_table": [{"ticker": "AAPL", "pe_ratio": 28.5}, {"ticker": "MSFT", "pe_ratio": 32.0}],
        "relative_valuation": "AAPL is cheap vs MSFT",
        "winner_overall": "AAPL",
        "error": None,
    }

    with patch("finsight.orchestrator.pipeline.get_stock_price", return_value=mock_price), patch(
        "finsight.orchestrator.pipeline.get_fundamentals", return_value=mock_fund
    ), patch("finsight.orchestrator.pipeline.get_news_sentiment", return_value=mock_news), patch(
        "finsight.orchestrator.pipeline.get_peer_comparison", return_value=mock_peers
    ), patch(
        "finsight.rag.retriever.retrieve_relevant_filing_context", return_value="Grounded Filing Context: Record R&D expenditure."
    ):

        final_state = run_research_pipeline("AAPL", peer_tickers=["MSFT"])

        assert isinstance(final_state, ResearchState)
        assert final_state.ticker == "AAPL"
        assert final_state.peer_tickers == ["MSFT"]

        assert final_state.raw_price == mock_price
        assert final_state.raw_fundamentals == mock_fund
        assert final_state.raw_news == mock_news
        assert final_state.raw_peers == mock_peers

        assert final_state.fundamentals_analysis["valuation_summary"] == "Fairly Valued"
        assert final_state.sentiment_analysis["headline_count"] == 1
        assert final_state.peer_analysis["winner_overall"] == "AAPL"

        assert final_state.thesis is not None
        assert "Investment Thesis for AAPL" in final_state.thesis
        assert final_state.report is not None
        assert "# FinSight Stock Research Report: AAPL" in final_state.report


# ---------------------------------------------------------------------------
# 4. RAG Retrieval Unit Test Against Small Fixture Corpus
# ---------------------------------------------------------------------------


def test_rag_retrieval_fixture_corpus(tmp_path) -> None:
    """Test retrieve_relevant_filing_context against a small isolated ChromaDB fixture corpus."""
    persist_dir = str(tmp_path / ".chroma_test")
    rag_pipeline = RAGPipeline(persist_directory=persist_dir)

    fixture_chunks = [
        {
            "chunk_id": "TEST_TICKER_1",
            "doc_id": "TEST_TICKER",
            "text": "The Company reported a revenue growth of 18.5% year over year driven by cloud demand.",
            "page_start": 1,
            "page_end": 1,
            "token_estimate": 15,
        },
        {
            "chunk_id": "TEST_TICKER_2",
            "doc_id": "TEST_TICKER",
            "text": "Risk factors include supply chain disruptions and foreign exchange volatility in emerging markets.",
            "page_start": 2,
            "page_end": 2,
            "token_estimate": 18,
        },
    ]

    rag_pipeline.ingest_chunks(doc_id="TEST_TICKER", chunks=fixture_chunks, batch_size=32)

    context = retrieve_relevant_filing_context(
        ticker="TEST_TICKER",
        query="revenue growth cloud demand",
        k=2,
        persist_directory=persist_dir,
    )

    assert "revenue growth of 18.5%" in context
    assert "Doc: TEST_TICKER" in context
    assert "Pages 1-1" in context
