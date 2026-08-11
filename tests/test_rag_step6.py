"""Step 6 RAG tests: FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finsight.api import main
from finsight.rag.generator import NOT_FOUND_ANSWER
from finsight.rag.pipeline import RAGPipeline
from finsight.rag.vector_store import VectorStore


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 384
        lowered = text.lower()
        if "revenue" in lowered or "sales" in lowered:
            vector[0] = 1.0
        elif "margin" in lowered or "ebitda" in lowered:
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector


class FakeGenerator:
    def generate(self, query: str, retrieved_chunks: list[dict]) -> dict:
        if not retrieved_chunks:
            return {
                "answer": NOT_FOUND_ANSWER,
                "citations": [],
                "chunks_used": 0,
                "model_used": "fake",
                "tokens_used": 0,
                "error": None,
            }

        page = retrieved_chunks[0]["metadata"]["page_start"]
        return {
            "answer": f"{retrieved_chunks[0]['text']} [Page {page}]",
            "citations": [page],
            "chunks_used": len(retrieved_chunks),
            "model_used": "fake",
            "tokens_used": 12,
            "error": None,
        }


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _minimal_pdf_bytes(page_texts: list[str]) -> bytes:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_ids: list[int] = []

    for text in page_texts:
        page_id = len(objects) + 1
        content_id = page_id + 1
        page_ids.append(page_id)
        text_ops = "\n".join(f"({_escape_pdf_text(line)}) Tj T*" for line in text.splitlines())
        stream = f"BT /F1 12 Tf 72 720 Td 16 TL {text_ops} ET".encode("latin-1")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


@pytest.fixture(autouse=True)
def isolated_rag_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    main.rate_limit_store.clear()
    pipeline = RAGPipeline(
        vector_store=VectorStore(str(tmp_path / ".chroma")),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
    )
    monkeypatch.setattr(main, "rag_pipeline", pipeline)


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture
def valid_pdf() -> bytes:
    text = (
        "Revenue increased significantly during the quarter as sales growth improved. "
        "Management also discussed margin pressure and operating leverage. "
        "The annual report includes risks, dividends, and financial performance commentary."
    )
    return _minimal_pdf_bytes([text])


def _ingest(client: TestClient, doc_id: str, pdf_bytes: bytes) -> dict:
    response = client.post(
        "/rag/ingest",
        data={"doc_id": doc_id},
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    return response.json()


def test_ingest_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/rag/ingest",
        data={"doc_id": "test_doc"},
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400


def test_ingest_rejects_invalid_doc_id(client: TestClient, valid_pdf: bytes) -> None:
    response = client.post(
        "/rag/ingest",
        data={"doc_id": "my doc!"},
        files={"file": ("report.pdf", valid_pdf, "application/pdf")},
    )

    assert response.status_code == 400


def test_ingest_rejects_oversized_file(client: TestClient) -> None:
    response = client.post(
        "/rag/ingest",
        data={"doc_id": "large_doc"},
        files={"file": ("large.pdf", b"0" * (21 * 1024 * 1024), "application/pdf")},
    )

    assert response.status_code == 400


def test_ingest_success(client: TestClient, valid_pdf: bytes) -> None:
    data = _ingest(client, "test_doc", valid_pdf)

    assert data["success"] is True
    assert data["chunks_created"] >= 1


def test_query_success(client: TestClient, valid_pdf: bytes) -> None:
    _ingest(client, "query_doc", valid_pdf)

    response = client.post("/rag/query", json={"doc_id": "query_doc", "question": "What happened to revenue?", "k": 3})

    assert response.status_code == 200
    assert len(response.json()["answer"]) > 5


def test_query_unknown_doc_returns_404(client: TestClient) -> None:
    response = client.post("/rag/query", json={"doc_id": "unknown_doc", "question": "What was revenue?", "k": 3})

    assert response.status_code == 404


def test_query_short_question_returns_422(client: TestClient) -> None:
    response = client.post("/rag/query", json={"doc_id": "unknown_doc", "question": "hi", "k": 3})

    assert response.status_code == 422


def test_list_documents_returns_list(client: TestClient) -> None:
    response = client.get("/rag/documents")

    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_delete_document(client: TestClient, valid_pdf: bytes) -> None:
    _ingest(client, "delete_doc", valid_pdf)

    delete_response = client.delete("/rag/documents/delete_doc")
    list_response = client.get("/rag/documents")

    assert delete_response.status_code == 200
    assert "delete_doc" not in list_response.json()["documents"]


def test_delete_unknown_doc_returns_404(client: TestClient) -> None:
    response = client.delete("/rag/documents/nonexistent")

    assert response.status_code == 404
