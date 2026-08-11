# HYBRID SEARCH — for interview reference
#
# Why not dense-only:
#   Dense retrieval embeds "EBITDA margin 24.3%" and
#   "profitability metric 24.3" similarly — numbers and
#   abbreviations don't embed distinctively enough.
#   Exact financial figures like "₹4,200 crore" or "Q3FY24"
#   are rare in the model's training data and embed poorly.
#
# Why not BM25-only:
#   BM25 misses semantic queries. "What did management say
#   about cost pressure" won't match "operational expenses
#   weighed on margins" because the words don't overlap.
#
# Solution: Reciprocal Rank Fusion (RRF)
#   score = dense_weight * 1/(rank_dense + 60)
#         + sparse_weight * 1/(rank_sparse + 60)
#   The constant 60 is standard RRF — it dampens rank-1
#   dominance so neither method overwhelms the other.
#   Default: dense_weight=0.7, sparse_weight=0.3
#   Rationale: financial narrative is mostly semantic,
#   but specific numbers need keyword backup.

from __future__ import annotations

from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, vector_store, embedder):
        self._vs = vector_store
        self._embedder = embedder
        self._bm25_index: dict[str, BM25Okapi] = {}
        self._corpus: dict[str, list[dict]] = {}

    def build_bm25_index(self, doc_id: str, chunks: list[dict]):
        """
        Tokenize each chunk's text (lowercase split on whitespace).
        Build BM25Okapi from tokenized corpus.
        Store both index and original chunks for later retrieval.
        """
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self._bm25_index[doc_id] = BM25Okapi(tokenized_corpus)
        self._corpus[doc_id] = chunks

    def retrieve(
        self,
        doc_id: str,
        query: str,
        k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict]:
        """
        Hybrid retrieval with RRF fusion.

        Step 1 — Dense:
          embed query with self._embedder.embed_query()
          get top k*2 results from self._vs.query()
          each result has: text, metadata, similarity

        Step 2 — Sparse (BM25):
          if doc_id not in self._bm25_index: skip sparse,
          use dense only
          tokenize query (lowercase split)
          get BM25 scores for all corpus chunks
          sort descending, take top k*2

        Step 3 — RRF fusion:
          For each result in dense list (rank 0..):
            key = first 60 chars of text (stable identifier)
            scores[key] += dense_weight * (1 / (rank + 60))

          For each result in sparse list (rank 0..):
            key = first 60 chars of text
            scores[key] += sparse_weight * (1 / (rank + 60))

          Sort by fused score descending, take top k
          Return the chunk dicts in that order

        Return list of dicts, each with:
          text: str
          metadata: dict (page_start, page_end, doc_id)
          similarity: float (from dense, or 0.0 if sparse-only)
          retrieval_method: str  "dense" | "sparse" | "hybrid"
        """
        if k <= 0:
            return []

        dense_results = self._dense_results(doc_id, query, k)
        sparse_results = self._sparse_results(doc_id, query, k)

        scores: dict[str, float] = {}
        ranked_results: dict[str, dict] = {}
        methods: dict[str, set[str]] = {}

        for rank, result in enumerate(dense_results):
            key = self._key(result["text"])
            scores[key] = scores.get(key, 0.0) + dense_weight * (1.0 / (rank + 60))
            ranked_results[key] = result
            methods.setdefault(key, set()).add("dense")

        for rank, result in enumerate(sparse_results):
            key = self._key(result["text"])
            scores[key] = scores.get(key, 0.0) + sparse_weight * (1.0 / (rank + 60))
            if key not in ranked_results:
                ranked_results[key] = result
            methods.setdefault(key, set()).add("sparse")

        sorted_keys = sorted(scores, key=scores.get, reverse=True)[:k]
        fused_results = []
        for key in sorted_keys:
            result = dict(ranked_results[key])
            method_set = methods[key]
            if method_set == {"dense", "sparse"}:
                retrieval_method = "hybrid"
            elif method_set == {"dense"}:
                retrieval_method = "dense"
            else:
                retrieval_method = "sparse"
            result["retrieval_method"] = retrieval_method
            fused_results.append(result)

        return fused_results

    def _dense_results(self, doc_id: str, query: str, k: int) -> list[dict]:
        query_embedding = self._embedder.embed_query(query)
        results = self._vs.query(doc_id, query_embedding, k=k * 2)
        return [
            {
                "text": result["text"],
                "metadata": result["metadata"],
                "similarity": float(result.get("similarity", 0.0)),
            }
            for result in results
        ]

    def _sparse_results(self, doc_id: str, query: str, k: int) -> list[dict]:
        if doc_id not in self._bm25_index:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index[doc_id].get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        chunks = self._corpus[doc_id]

        results = []
        for index in ranked_indices[: k * 2]:
            chunk = chunks[index]
            results.append(
                {
                    "text": chunk["text"],
                    "metadata": {
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "doc_id": chunk["doc_id"],
                        "token_estimate": chunk.get("token_estimate", 0),
                    },
                    "similarity": 0.0,
                }
            )
        return results

    def _tokenize(self, text: str) -> list[str]:
        return str(text).lower().split()

    def _key(self, text: str) -> str:
        return text[:60]


def retrieve_relevant_filing_context(
    ticker: str,
    query: str,
    k: int = 3,
    persist_directory: str = ".chroma",
) -> str:
    """Retrieve relevant filing context chunks for a ticker from ChromaDB."""
    try:
        from finsight.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(persist_directory=persist_directory)
        normalized_doc = ticker.strip().upper().replace(".", "_")

        doc_ids_to_try = [normalized_doc, ticker.strip().upper()]
        selected_doc_id = None

        for doc_id in doc_ids_to_try:
            if pipeline.vector_store.document_exists(doc_id):
                selected_doc_id = doc_id
                break

        if not selected_doc_id:
            try:
                existing_cols = pipeline.vector_store._client.list_collections()
                for col in existing_cols:
                    col_name = col.name if hasattr(col, "name") else str(col)
                    if normalized_doc in col_name or ticker.strip().upper() in col_name:
                        selected_doc_id = col_name
                        break
            except Exception:
                pass

        if not selected_doc_id:
            return f"No filing context collection found in vector store for ticker '{ticker}'."

        res = pipeline.query(doc_id=selected_doc_id, question=query, k=k)
        sources = res.get("sources") or []
        if not sources:
            return f"No relevant filing excerpts matched query '{query}' for ticker '{ticker}'."

        formatted_excerpts = []
        for index, source in enumerate(sources, start=1):
            text = str(source.get("text") or "").strip()
            meta = source.get("metadata") or {}
            page_start = meta.get("page_start", "?")
            page_end = meta.get("page_end", "?")
            doc_id = meta.get("doc_id", selected_doc_id)
            formatted_excerpts.append(
                f"[{index}] [Doc: {doc_id} | Pages {page_start}-{page_end}]\n{text}"
            )

        return "\n\n".join(formatted_excerpts)
    except Exception as exc:
        return f"Unable to retrieve filing context for '{ticker}': {exc}"
