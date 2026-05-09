"""PDF extraction and sentence-aware chunking for FinSight RAG."""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber


def _normalize_page_text(text: str) -> str:
    """Clean page text while keeping paragraph boundaries readable."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Use pdfplumber to extract text page by page.
    Return list of {page_number: int, text: str}.
    Skip pages with fewer than 50 characters.
    Strip excessive whitespace, preserve paragraph breaks.
    """
    pages: list[dict] = []
    with pdfplumber.open(Path(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = _normalize_page_text(page.extract_text() or "")
            if len(text) < 50:
                continue
            pages.append({"page_number": index, "text": text})
    return pages


def _join_pages_with_boundaries(pages: list[dict]) -> tuple[str, list[dict]]:
    parts: list[str] = []
    boundaries: list[dict] = []
    offset = 0

    for page in pages:
        if parts:
            parts.append("\n\n")
            offset += 2

        text = str(page["text"]).strip()
        start = offset
        parts.append(text)
        offset += len(text)
        boundaries.append(
            {
                "page_number": int(page["page_number"]),
                "char_start": start,
                "char_end": offset,
            }
        )

    return "".join(parts), boundaries


def _split_sentences(text: str) -> list[dict]:
    sentences: list[dict] = []
    start = 0
    length = len(text)
    index = 0

    while index < length:
        if text[index] == "." and index + 1 < length and text[index + 1] in {" ", "\n"}:
            end = index + 1
            sentence = text[start:end].strip()
            if sentence:
                leading = len(text[start:end]) - len(text[start:end].lstrip())
                trailing = len(text[start:end]) - len(text[start:end].rstrip())
                sentences.append(
                    {
                        "text": sentence,
                        "char_start": start + leading,
                        "char_end": end - trailing,
                    }
                )
            start = end
            while start < length and text[start] in {" ", "\n"}:
                start += 1
            index = start
            continue
        index += 1

    if start < length:
        sentence = text[start:].strip()
        if sentence:
            leading = len(text[start:]) - len(text[start:].lstrip())
            sentences.append(
                {
                    "text": sentence,
                    "char_start": start + leading,
                    "char_end": length,
                }
            )

    return sentences


def _page_for_offset(boundaries: list[dict], offset: int) -> int:
    if not boundaries:
        return 0

    for boundary in boundaries:
        if boundary["char_start"] <= offset < boundary["char_end"]:
            return boundary["page_number"]

    return boundaries[-1]["page_number"]


def chunk_document(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_size: int = 100,
) -> list[dict]:
    # CHUNKING STRATEGY — for interview reference
    # chunk_size=512 tokens: large enough for financial context,
    #   small enough to stay precise in retrieval
    # overlap=64 tokens: prevents answer truncation at boundaries.
    #   Without overlap, a sentence spanning two chunks gets lost.
    # sentence-aware split: avoids cutting mid-sentence which
    #   degrades both embedding quality and LLM readability.
    # min_chunk_size filter: removes header/footer fragments
    #   (page numbers, company names) that pollute the index.
    """
    Sentence-aware sliding window chunker.
    """
    if not pages:
        return []

    max_chars = chunk_size * 4
    overlap_chars = overlap * 4
    full_text, boundaries = _join_pages_with_boundaries(pages)
    sentences = _split_sentences(full_text)
    chunks: list[dict] = []
    sentence_index = 0

    while sentence_index < len(sentences):
        chunk_start = sentence_index
        chunk_end = sentence_index

        while chunk_end < len(sentences):
            candidate_start = sentences[chunk_start]["char_start"]
            candidate_end = sentences[chunk_end]["char_end"]
            candidate_length = candidate_end - candidate_start
            if chunk_end > chunk_start and candidate_length > max_chars:
                break
            chunk_end += 1

        if chunk_end == chunk_start:
            chunk_end += 1

        char_start = sentences[chunk_start]["char_start"]
        char_end = sentences[chunk_end - 1]["char_end"]
        chunk_text = full_text[char_start:char_end].strip()

        if len(chunk_text) >= min_chunk_size:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{len(chunks):04d}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "page_start": _page_for_offset(boundaries, char_start),
                    "page_end": _page_for_offset(boundaries, max(char_end - 1, char_start)),
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_estimate": len(chunk_text) // 4,
                }
            )

        if chunk_end >= len(sentences):
            break

        next_start = chunk_end
        tail_chars = 0
        while next_start > chunk_start and tail_chars < overlap_chars:
            next_start -= 1
            tail_chars = sentences[chunk_end - 1]["char_end"] - sentences[next_start]["char_start"]

        if next_start <= chunk_start:
            next_start = chunk_start + 1
        sentence_index = next_start

    return chunks
