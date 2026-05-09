"""Step 3 RAG tests: ChromaDB vector store wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finsight.rag.vector_store import VectorStore


@pytest.fixture
def vs(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_directory=str(tmp_path / ".chroma"))


@pytest.fixture
def dummy_embeddings() -> list[list[float]]:
    rng = np.random.default_rng(42)
    base = rng.normal(size=384)
    vectors = []
    for _ in range(12):
        vector = base + rng.normal(scale=0.01, size=384)
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector.astype(float).tolist())
    return vectors


def _chunks(doc_id: str, count: int, page_start: int = 1) -> list[dict]:
    return [
        {
            "chunk_id": f"{doc_id}_{index:04d}",
            "doc_id": doc_id,
            "text": f"Financial performance chunk {index} for {doc_id}.",
            "page_start": page_start + index,
            "page_end": page_start + index,
            "char_start": index * 100,
            "char_end": (index + 1) * 100,
            "token_estimate": 25,
        }
        for index in range(count)
    ]


def test_add_and_query_returns_results(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    chunks = _chunks("test_doc", 5)
    vs.add_chunks("test_doc", chunks, dummy_embeddings[:5])

    results = vs.query("test_doc", dummy_embeddings[0], k=5)

    assert len(results) >= 1


def test_similarity_is_between_0_and_1(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    chunks = _chunks("test_doc", 5)
    vs.add_chunks("test_doc", chunks, dummy_embeddings[:5])

    results = vs.query("test_doc", dummy_embeddings[0], k=5)

    assert results
    assert all(0.0 <= result["similarity"] <= 1.0 for result in results)


def test_document_exists_after_add(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    chunks = _chunks("test_doc", 5)
    vs.add_chunks("test_doc", chunks, dummy_embeddings[:5])

    assert vs.document_exists("test_doc") is True


def test_document_not_exists_before_add(vs: VectorStore) -> None:
    assert vs.document_exists("ghost") is False


def test_delete_removes_document(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    chunks = _chunks("test_doc", 5)
    vs.add_chunks("test_doc", chunks, dummy_embeddings[:5])

    vs.delete_document("test_doc")

    assert vs.document_exists("test_doc") is False


def test_list_documents_returns_doc_ids(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    vs.add_chunks("doc_a", _chunks("doc_a", 2), dummy_embeddings[:2])
    vs.add_chunks("doc_b", _chunks("doc_b", 2), dummy_embeddings[2:4])

    docs = vs.list_documents()

    assert "doc_a" in docs
    assert "doc_b" in docs


def test_query_empty_collection_returns_empty(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    vs.get_or_create_collection("empty_doc")

    results = vs.query("empty_doc", dummy_embeddings[0], k=5)

    assert results == []


def test_k_limits_results(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    chunks = _chunks("test_doc", 10)
    vs.add_chunks("test_doc", chunks, dummy_embeddings[:10])

    results = vs.query("test_doc", dummy_embeddings[0], k=3)

    assert len(results) <= 3


def test_chunk_count_accurate(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    chunks = _chunks("test_doc", 7)
    vs.add_chunks("test_doc", chunks, dummy_embeddings[:7])

    assert vs.chunk_count("test_doc") == 7


def test_metadata_preserved(vs: VectorStore, dummy_embeddings: list[list[float]]) -> None:
    chunks = _chunks("test_doc", 1, page_start=42)
    vs.add_chunks("test_doc", chunks, dummy_embeddings[:1])

    results = vs.query("test_doc", dummy_embeddings[0], k=1)

    assert results[0]["metadata"]["page_start"] == 42
