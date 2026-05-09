# MODEL CHOICE — for interview reference
# Model: sentence-transformers/all-MiniLM-L6-v2
# Dimensions: 384
# Size: ~80MB, runs on CPU, ~50ms per chunk after warmup
# Why this model:
#   - Standard benchmark model for semantic similarity tasks
#   - Fast enough for real-time query embedding
#   - 384 dims keeps ChromaDB storage small vs 1536 (OpenAI ada-002)
# Production upgrade path:
#   - FinancialBERT embeddings for domain-specific financial terms
#   - text-embedding-3-small (OpenAI) for higher accuracy at low cost
#   - Matryoshka embeddings if storage becomes a constraint

from __future__ import annotations


class Embedder:
    def __init__(self):
        self._model = None

    def _load(self):
        """Load model once, cache as instance variable."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.
        Returns list of 384-dimensional float vectors.
        Use model.encode() with convert_to_numpy=True then .tolist().
        """
        model = self._load()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        Must use the SAME model as embed_texts — this is critical.
        Asymmetric embedding (different models for query vs doc)
        is a common mistake that destroys retrieval quality.
        """
        return self.embed_texts([query])[0]


# Module-level singleton — import this everywhere
embedder = Embedder()
