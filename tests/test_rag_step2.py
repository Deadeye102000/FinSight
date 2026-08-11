"""Step 2 RAG tests: embedding layer."""

from __future__ import annotations

import numpy as np

from finsight.rag.embedder import Embedder, embedder


def test_embed_query_returns_384_dims() -> None:
    result = embedder.embed_query("test query")

    assert len(result) == 384


def test_embed_texts_batch_shape() -> None:
    result = embedder.embed_texts(["hello", "world", "finance"])

    assert len(result) == 3
    assert len(result[0]) == 384


def test_same_model_for_query_and_docs() -> None:
    v1 = embedder.embed_query("revenue growth")
    v2 = embedder.embed_texts(["revenue growth"])[0]

    assert all(abs(a - b) < 1e-5 for a, b in zip(v1, v2))


def test_similar_texts_have_higher_cosine() -> None:
    v_profit = embedder.embed_query("company reported strong profit growth")
    v_revenue = embedder.embed_query("revenue increased significantly this quarter")
    v_weather = embedder.embed_query("it rained heavily in Mumbai yesterday")

    def cosine(a: list[float], b: list[float]) -> float:
        a_array = np.array(a)
        b_array = np.array(b)
        return np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))

    assert cosine(v_profit, v_revenue) > cosine(v_profit, v_weather)


def test_lazy_load_works() -> None:
    e = Embedder()

    assert e._model is None
    e.embed_query("test")
    assert e._model is not None


def test_embeddings_are_floats() -> None:
    result = embedder.embed_query("test")

    assert all(isinstance(x, float) for x in result)


def test_empty_string_does_not_crash() -> None:
    result = embedder.embed_query("")

    assert len(result) == 384


def test_long_text_truncated_gracefully() -> None:
    long_text = "word " * 1000
    result = embedder.embed_query(long_text)

    assert len(result) == 384
