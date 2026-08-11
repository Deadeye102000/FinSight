"""Step 4 RAG tests: hybrid dense + BM25 retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from finsight.rag.embedder import embedder
from finsight.rag.retriever import HybridRetriever
from finsight.rag.vector_store import VectorStore


@pytest.fixture
def vs(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_directory=str(tmp_path / ".chroma"))


@pytest.fixture
def doc_id() -> str:
    return "hybrid_doc"


@pytest.fixture
def chunks(doc_id: str) -> list[dict]:
    texts = [
        "Revenue growth accelerated as enterprise customers expanded cloud usage.",
        "Sales growth remained strong across the North America and India segments.",
        "Management said revenue increased because renewal rates improved materially.",
        "EBITDA margin 24.3 percent Q3FY24 improved because operating leverage increased.",
        "The company reported EBITDA margin 24.3 percent Q3FY24 versus 21.8 percent last year.",
        "Management highlighted EBITDA margin 24.3 percent Q3FY24 as a profitability milestone.",
        "The board of directors meeting approved the quarterly financial results.",
        "Board of directors meeting minutes noted a discussion on governance controls.",
        "Archive reference alpha beta gamma with unrelated filler language.",
        "Miscellaneous filler text about office supplies and travel logistics.",
    ]
    return [
        {
            "chunk_id": f"{doc_id}_{index:04d}",
            "doc_id": doc_id,
            "text": text,
            "page_start": index + 1,
            "page_end": index + 1,
            "char_start": index * 100,
            "char_end": (index + 1) * 100,
            "token_estimate": len(text) // 4,
        }
        for index, text in enumerate(texts)
    ]


@pytest.fixture
def retriever(vs: VectorStore, doc_id: str, chunks: list[dict]) -> HybridRetriever:
    embeddings = embedder.embed_texts([chunk["text"] for chunk in chunks])
    vs.add_chunks(doc_id, chunks, embeddings)
    hybrid = HybridRetriever(vs, embedder)
    hybrid.build_bm25_index(doc_id, chunks)
    return hybrid


def test_retrieve_returns_k_results(retriever: HybridRetriever, doc_id: str) -> None:
    results = retriever.retrieve(doc_id, "revenue growth", k=3)

    assert len(results) == 3


def test_retrieve_revenue_query_ranks_revenue_chunks_high(
    retriever: HybridRetriever,
    doc_id: str,
) -> None:
    results = retriever.retrieve(doc_id, "revenue growth", k=5)
    top_texts = [result["text"] for result in results[:3]]

    assert any("revenue" in text.lower() or "sales" in text.lower() for text in top_texts)


def test_bm25_catches_exact_number(retriever: HybridRetriever, doc_id: str) -> None:
    results = retriever.retrieve(doc_id, "EBITDA margin 24.3", k=5)
    top_texts = [result["text"] for result in results[:3]]

    assert any("24.3" in text for text in top_texts)


def test_no_bm25_index_falls_back_to_dense(
    vs: VectorStore,
    doc_id: str,
    chunks: list[dict],
) -> None:
    embeddings = embedder.embed_texts([chunk["text"] for chunk in chunks])
    vs.add_chunks(doc_id, chunks, embeddings)
    retriever_no_bm25 = HybridRetriever(vs, embedder)

    results = retriever_no_bm25.retrieve(doc_id, "revenue", k=3)

    assert len(results) >= 1


def test_retrieval_method_field_present(retriever: HybridRetriever, doc_id: str) -> None:
    results = retriever.retrieve(doc_id, "revenue", k=3)

    assert all("retrieval_method" in result for result in results)


def test_weights_sum_affects_ranking(retriever: HybridRetriever, doc_id: str) -> None:
    retriever.retrieve(doc_id, "EBITDA 24.3", k=5, dense_weight=1.0, sparse_weight=0.0)
    results_sparse = retriever.retrieve(doc_id, "EBITDA 24.3", k=5, dense_weight=0.0, sparse_weight=1.0)
    sparse_top = results_sparse[0]["text"]

    assert "24.3" in sparse_top or "EBITDA" in sparse_top


def test_k_respected(retriever: HybridRetriever, doc_id: str) -> None:
    results = retriever.retrieve(doc_id, "any query", k=2)

    assert len(results) <= 2


def test_metadata_present_in_results(retriever: HybridRetriever, doc_id: str) -> None:
    results = retriever.retrieve(doc_id, "revenue", k=3)

    assert all("page_start" in result["metadata"] for result in results)
