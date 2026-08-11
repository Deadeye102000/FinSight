"""One-time filing text ingestion script to populate ChromaDB from NSE watchlist."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsight.mcp_server.tools.filings import get_filing_text
from finsight.rag.chunker import chunk_document
from finsight.rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ingest_filings")

WATCHLIST = [
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
]


def ingest_watchlist(
    watchlist: list[str] = WATCHLIST,
    persist_dir: str = ".chroma",
) -> dict[str, Any]:
    """Execute one-time ingestion of filing disclosures across the watchlist into ChromaDB."""
    logger.info("Starting one-time filing ingestion for %d watchlist companies...", len(watchlist))

    pipeline = RAGPipeline(persist_directory=persist_dir)

    successful_companies: list[str] = []
    failed_companies: list[str] = []
    total_chunks_indexed = 0
    total_docs_indexed = 0

    t_start = time.perf_counter()

    for index, ticker in enumerate(watchlist, start=1):
        logger.info("[%d/%d] Fetching filing text for %s...", index, len(watchlist), ticker)
        try:
            filing_data = get_filing_text(ticker=ticker, filing_type="annual_report")
            filing_text = filing_data.get("filing_text", "").strip()

            if not filing_text or len(filing_text) < 50 or filing_data.get("error"):
                logger.warning("No fetchable filing text returned for %s: %s", ticker, filing_data.get("error"))
                failed_companies.append(ticker)
                continue

            doc_id = ticker.strip().upper().replace(".", "_")

            # Convert filing_text into pseudo-page structure for chunker
            pages = [{"page_number": 1, "text": filing_text}]
            chunks = chunk_document(
                pages=pages,
                doc_id=doc_id,
                chunk_size=512,
                overlap=64,
                min_chunk_size=30,
            )

            if not chunks:
                logger.warning("Zero chunks generated for %s", ticker)
                failed_companies.append(ticker)
                continue

            result = pipeline.ingest_chunks(doc_id=doc_id, chunks=chunks, batch_size=32)
            indexed_count = result.get("chunks_indexed", 0)

            total_chunks_indexed += indexed_count
            total_docs_indexed += 1
            successful_companies.append(ticker)
            logger.info("Successfully ingested %s: %d chunks into ChromaDB", ticker, indexed_count)

        except Exception as exc:
            logger.error("Failed ingestion for %s: %s", ticker, exc)
            failed_companies.append(ticker)

        # Respectful pacing between tickers
        time.sleep(0.5)

    duration = time.perf_counter() - t_start

    stats = {
        "total_watchlist_attempted": len(watchlist),
        "successful_companies_count": len(successful_companies),
        "failed_companies_count": len(failed_companies),
        "successful_companies": successful_companies,
        "failed_companies": failed_companies,
        "total_documents_indexed": total_docs_indexed,
        "total_chunks_stored": total_chunks_indexed,
        "duration_seconds": round(duration, 2),
    }

    print("\n==================================================")
    print("      FinSight Filing RAG Ingestion Statistics    ")
    print("==================================================")
    print(f"Watchlist Attempted:      {stats['total_watchlist_attempted']}")
    print(f"Successfully Ingested:    {stats['successful_companies_count']}")
    print(f"Failed / Unfetchable:    {stats['failed_companies_count']}")
    print(f"Total Documents Index:    {stats['total_documents_indexed']}")
    print(f"Total Chunks Stored:      {stats['total_chunks_stored']}")
    print(f"Ingestion Duration:       {stats['duration_seconds']}s")
    print("==================================================\n")

    return stats


if __name__ == "__main__":
    ingest_watchlist()
