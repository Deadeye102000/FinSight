"""End-to-end RAG ingestion and query pipeline."""

from __future__ import annotations

from finsight.rag.chunker import chunk_document, extract_text_from_pdf
from finsight.rag.embedder import embedder as default_embedder
from finsight.rag.generator import RAGGenerator
from finsight.rag.retriever import HybridRetriever
from finsight.rag.vector_store import VectorStore


class RAGPipeline:
    """Orchestrate PDF extraction, chunking, embedding, retrieval, and generation."""

    def __init__(
        self,
        persist_directory: str = ".chroma",
        vector_store: VectorStore | None = None,
        embedder=None,
        generator: RAGGenerator | None = None,
    ):
        self.vector_store = vector_store or VectorStore(persist_directory=persist_directory)
        self.embedder = embedder or default_embedder
        self.generator = generator or RAGGenerator()
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        self._chunk_cache: dict[str, list[dict]] = {}

    def ingest_pdf(
        self,
        pdf_path: str,
        doc_id: str,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 100,
        batch_size: int = 32,
    ) -> dict:
        """Extract, chunk, embed, store, and index one PDF."""
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_document(
            pages,
            doc_id=doc_id,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
        result = self.ingest_chunks(doc_id, chunks, batch_size=batch_size)
        result["pages_extracted"] = len(pages)
        return result

    def ingest_chunks(self, doc_id: str, chunks: list[dict], batch_size: int = 32) -> dict:
        """Embed, store, and index already-created chunks."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if self.vector_store.document_exists(doc_id):
            self.vector_store.delete_document(doc_id)

        embeddings = self._embed_chunks_in_batches(chunks, batch_size=batch_size)
        if chunks:
            self.vector_store.add_chunks(doc_id, chunks, embeddings)
            self.retriever.build_bm25_index(doc_id, chunks)
        self._chunk_cache[doc_id] = chunks

        return {
            "doc_id": doc_id,
            "chunks_indexed": len(chunks),
            "batch_size": batch_size,
        }

    def query(
        self,
        doc_id: str,
        question: str,
        k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> dict:
        """Retrieve grounded chunks and generate an answer."""
        if not self.vector_store.document_exists(doc_id):
            generated = self.generator.generate(question, [])
            return {
                "doc_id": doc_id,
                "question": question,
                "answer": generated["answer"],
                "citations": generated["citations"],
                "sources": [],
                "chunks_retrieved": 0,
                "error": generated.get("error"),
            }

        self._ensure_bm25_index(doc_id)
        sources = self.retriever.retrieve(
            doc_id,
            question,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        generated = self.generator.generate(question, sources)
        return {
            "doc_id": doc_id,
            "question": question,
            "answer": generated["answer"],
            "citations": generated["citations"],
            "sources": sources,
            "chunks_retrieved": len(sources),
            "error": generated.get("error"),
        }

    def _embed_chunks_in_batches(self, chunks: list[dict], batch_size: int) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings.extend(self.embedder.embed_texts([chunk["text"] for chunk in batch]))
        return embeddings

    def _ensure_bm25_index(self, doc_id: str) -> None:
        if doc_id in self._chunk_cache and doc_id in self.retriever._bm25_index:
            return

        chunks = self._chunk_cache.get(doc_id)
        if chunks is None:
            chunks = self._load_chunks_from_vector_store(doc_id)
            self._chunk_cache[doc_id] = chunks

        if chunks and doc_id not in self.retriever._bm25_index:
            self.retriever.build_bm25_index(doc_id, chunks)

    def _load_chunks_from_vector_store(self, doc_id: str) -> list[dict]:
        collection = self.vector_store._get_collection(doc_id)
        if collection is None or collection.count() == 0:
            return []

        raw = collection.get(include=["documents", "metadatas"], limit=collection.count())
        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        chunks = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", doc_id),
                    "text": text,
                    "page_start": metadata.get("page_start", 0),
                    "page_end": metadata.get("page_end", metadata.get("page_start", 0)),
                    "char_start": 0,
                    "char_end": len(text or ""),
                    "token_estimate": metadata.get("token_estimate", len(text or "") // 4),
                }
            )
        return chunks
