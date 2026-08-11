"""Step 1 RAG tests: PDF extraction and sentence-aware chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from finsight.rag.chunker import chunk_document, extract_text_from_pdf


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

        lines = text.splitlines() or [""]
        text_ops = "\n".join(f"({_escape_pdf_text(line)}) Tj T*" for line in lines)
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


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    page_texts = [
        "Page one revenue increased by eighteen percent. "
        "Management attributed the growth to strong cloud demand. "
        "Margins also improved during the quarter.",
        "Page two cash flow remained healthy. "
        "The company reduced debt and improved working capital. "
        "Guidance remained stable for the next quarter.",
        "Page three risks include currency pressure. "
        "Management also discussed competitive pricing. "
        "Capital allocation remains disciplined.",
    ]
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_minimal_pdf_bytes(page_texts))
    return pdf_path


@pytest.fixture
def chunk_pages() -> list[dict]:
    sentence = (
        "Revenue grew steadily as enterprise customers expanded contracts "
        "and management preserved operating margins."
    )
    pages = []
    for page_number in range(1, 4):
        text = " ".join(f"{sentence} Page {page_number} sentence {idx}." for idx in range(1, 11))
        pages.append({"page_number": page_number, "text": text})
    return pages


def test_extract_returns_list_of_dicts(sample_pdf: Path) -> None:
    pages = extract_text_from_pdf(str(sample_pdf))

    assert pages
    assert all({"page_number", "text"} <= set(item) for item in pages)


def test_extract_skips_blank_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank_middle.pdf"
    pdf_path.write_bytes(
        _minimal_pdf_bytes(
            [
                "Page one contains more than enough text for extraction. "
                "This paragraph should remain in the result.",
                "",
                "Page three contains more than enough text for extraction. "
                "This paragraph should also remain in the result.",
            ]
        )
    )

    pages = extract_text_from_pdf(str(pdf_path))

    assert [page["page_number"] for page in pages] == [1, 3]


def test_chunk_ids_are_unique(chunk_pages: list[dict]) -> None:
    chunks = chunk_document(chunk_pages, "annual_report", chunk_size=80, overlap=16)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_has_required_fields(chunk_pages: list[dict]) -> None:
    chunks = chunk_document(chunk_pages, "annual_report", chunk_size=80, overlap=16)
    required_fields = {
        "chunk_id",
        "doc_id",
        "text",
        "page_start",
        "page_end",
        "char_start",
        "char_end",
        "token_estimate",
    }

    assert chunks
    assert all(required_fields <= set(chunk) for chunk in chunks)


def test_chunk_size_respected(chunk_pages: list[dict]) -> None:
    chunk_size = 80
    chunks = chunk_document(chunk_pages, "annual_report", chunk_size=chunk_size, overlap=16)

    assert all(chunk["token_estimate"] <= chunk_size + 20 for chunk in chunks)


def test_overlap_creates_continuity(chunk_pages: list[dict]) -> None:
    chunks = chunk_document(chunk_pages, "annual_report", chunk_size=80, overlap=64)

    assert len(chunks) > 1
    assert chunks[0]["text"][-32:] in chunks[1]["text"]


def test_min_chunk_filter_removes_tiny_chunks(chunk_pages: list[dict]) -> None:
    chunks = chunk_document(chunk_pages, "annual_report", chunk_size=80, overlap=16, min_chunk_size=120)

    assert chunks
    assert all(len(chunk["text"]) >= 120 for chunk in chunks)


def test_doc_id_in_chunk_ids(chunk_pages: list[dict]) -> None:
    doc_id = "annual_report"
    chunks = chunk_document(chunk_pages, doc_id, chunk_size=80, overlap=16)

    assert chunks
    assert all(chunk["chunk_id"].startswith(doc_id) for chunk in chunks)
