"""Step 5 RAG tests: generator and end-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from finsight.rag.generator import NOT_FOUND_ANSWER, RAGGenerator
from finsight.rag.pipeline import RAGPipeline
from finsight.rag.vector_store import VectorStore


class FakeEmbedder:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [0.0] * 384
        if "ebitda" in lowered or "24.3" in lowered:
            vector[1] = 1.0
        elif "revenue" in lowered or "sales" in lowered:
            vector[0] = 1.0
        else:
            vector[2] = 1.0
        return vector


class FakeGenerator:
    def generate(self, query: str, retrieved_chunks: list[dict]) -> dict:
        if not retrieved_chunks:
            return {"answer": NOT_FOUND_ANSWER, "citations": [], "chunks_used": 0, "error": None}

        page = retrieved_chunks[0]["metadata"]["page_start"]
        return {
            "answer": f"{retrieved_chunks[0]['text']} [Page {page}]",
            "citations": [page],
            "chunks_used": len(retrieved_chunks),
            "error": None,
        }


class _FakeAnthropicMessages:
    def create(self, **kwargs):
        self.kwargs = kwargs

        class Block:
            type = "text"
            text = "Revenue increased according to the uploaded filing. [Page 2]"

        class Response:
            content = [Block()]

        return Response()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeAnthropicMessages()


def _chunks(doc_id: str, count: int) -> list[dict]:
    chunks = []
    for index in range(count):
        if index == 0:
            text = "Revenue increased as sales growth improved across enterprise customers."
        elif index == 1:
            text = "EBITDA margin 24.3 percent Q3FY24 improved due to operating leverage."
        else:
            text = f"General filing commentary and governance update number {index}."
        chunks.append(
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
        )
    return chunks


def test_generator_returns_not_found_without_chunks() -> None:
    generator = RAGGenerator(anthropic_client=FakeAnthropicClient())

    result = generator.generate("What was revenue?", [])

    assert result["answer"] == NOT_FOUND_ANSWER
    assert result["citations"] == []


def test_generator_uses_injected_llm_and_extracts_citations() -> None:
    generator = RAGGenerator(anthropic_client=FakeAnthropicClient())
    chunks = [
        {
            "text": "Revenue increased according to the uploaded filing.",
            "metadata": {"page_start": 2, "page_end": 2, "doc_id": "doc"},
        }
    ]

    result = generator.generate("What happened to revenue?", chunks)

    assert "Revenue increased" in result["answer"]
    assert result["citations"] == [2]


def test_pipeline_ingest_chunks_batches_32(tmp_path: Path) -> None:
    fake_embedder = FakeEmbedder()
    pipeline = RAGPipeline(
        vector_store=VectorStore(str(tmp_path / ".chroma")),
        embedder=fake_embedder,
        generator=FakeGenerator(),
    )

    result = pipeline.ingest_chunks("batch_doc", _chunks("batch_doc", 65))

    assert result["chunks_indexed"] == 65
    assert fake_embedder.batch_sizes == [32, 32, 1]


def test_pipeline_query_returns_answer_and_sources(tmp_path: Path) -> None:
    pipeline = RAGPipeline(
        vector_store=VectorStore(str(tmp_path / ".chroma")),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
    )
    pipeline.ingest_chunks("query_doc", _chunks("query_doc", 5))

    result = pipeline.query("query_doc", "revenue growth", k=2)

    assert result["answer"]
    assert result["citations"]
    assert len(result["sources"]) <= 2


def test_pipeline_rebuilds_bm25_from_chroma_after_restart(tmp_path: Path) -> None:
    persist_directory = str(tmp_path / ".chroma")
    first = RAGPipeline(
        vector_store=VectorStore(persist_directory),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
    )
    first.ingest_chunks("restart_doc", _chunks("restart_doc", 5))

    second = RAGPipeline(
        vector_store=VectorStore(persist_directory),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
    )
    result = second.query("restart_doc", "EBITDA margin 24.3", k=3, dense_weight=0.0, sparse_weight=1.0)

    assert any("24.3" in source["text"] for source in result["sources"])
    assert "restart_doc" in second._chunk_cache


def test_pipeline_missing_document_returns_not_found(tmp_path: Path) -> None:
    pipeline = RAGPipeline(
        vector_store=VectorStore(str(tmp_path / ".chroma")),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
    )

    result = pipeline.query("missing_doc", "What was revenue?")

    assert result["answer"] == NOT_FOUND_ANSWER
    assert result["sources"] == []
