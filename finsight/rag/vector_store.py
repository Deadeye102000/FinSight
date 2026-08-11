# VECTOR STORE CHOICE — for interview reference
# ChromaDB in PersistentClient mode
# Data stored in .chroma/ directory (gitignored)
#
# Why ChromaDB over alternatives:
#   FAISS: faster ANN search but no built-in persistence or metadata
#          filtering without extra code
#   Pinecone: managed, fast, but paid and requires internet
#   Weaviate: self-hosted but heavier operational overhead
#   ChromaDB: persistent, metadata-filterable, runs in-process,
#             zero infra — right choice for a portfolio project
#
# One collection per document (doc_id = collection name).
# This keeps retrieval scoped — querying TCS annual report
# cannot accidentally return Reliance chunks.

from __future__ import annotations


class VectorStore:
    def __init__(self, persist_directory: str = ".chroma"):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(self, doc_id: str):
        """
        Each document gets its own collection.
        Use cosine similarity space: {"hnsw:space": "cosine"}
        """
        return self._client.get_or_create_collection(
            name=doc_id,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        """
        Add chunks + embeddings to the collection.
        ChromaDB requires:
          ids: list[str]          — use chunk["chunk_id"]
          embeddings: list[list[float]]
          documents: list[str]    — use chunk["text"]
          metadatas: list[dict]   — use page_start, page_end,
                                    doc_id, token_estimate
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        collection = self.get_or_create_collection(doc_id)
        collection.add(
            ids=[str(chunk["chunk_id"]) for chunk in chunks],
            embeddings=embeddings,
            documents=[str(chunk["text"]) for chunk in chunks],
            metadatas=[
                {
                    "page_start": int(chunk["page_start"]),
                    "page_end": int(chunk["page_end"]),
                    "doc_id": str(chunk["doc_id"]),
                    "token_estimate": int(chunk["token_estimate"]),
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        doc_id: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """
        Retrieve top-k chunks by cosine similarity.
        Use collection.query() with include=["documents","metadatas","distances"]
        Convert distance to similarity: similarity = 1 - distance
        Return list of:
        {
          text: str,
          metadata: dict,
          distance: float,
          similarity: float
        }
        Handle edge case: if collection is empty return [].
        n_results must be min(k, collection.count()) — ChromaDB
        raises if you ask for more results than exist.
        """
        collection = self._get_collection(doc_id)
        if collection is None or k <= 0:
            return []

        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "text": document,
                "metadata": metadata,
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def document_exists(self, doc_id: str) -> bool:
        """Return True if collection exists AND has at least 1 chunk."""
        collection = self._get_collection(doc_id)
        return collection is not None and collection.count() > 0

    def list_documents(self) -> list[str]:
        """Return all collection names (= all ingested doc_ids)."""
        collections = self._client.list_collections()
        return [collection if isinstance(collection, str) else collection.name for collection in collections]

    def delete_document(self, doc_id: str):
        """Delete the collection for this doc_id."""
        if self._get_collection(doc_id) is None:
            return
        self._client.delete_collection(doc_id)

    def chunk_count(self, doc_id: str) -> int:
        """Return number of chunks stored for this doc_id."""
        collection = self._get_collection(doc_id)
        if collection is None:
            return 0
        return collection.count()

    def _get_collection(self, doc_id: str):
        try:
            return self._client.get_collection(doc_id)
        except Exception:
            return None
