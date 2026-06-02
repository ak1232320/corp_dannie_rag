"""Тесты гибридного поиска: нормировка, компоненты скоров, ранжирование."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import app.hybrid_retriever as hr
from app.hybrid_retriever import HybridRetriever, _minmax


def test_minmax_basic():
    out = _minmax(np.array([0.0, 5.0, 10.0]))
    assert out.min() == 0.0 and out.max() == 1.0
    assert abs(out[1] - 0.5) < 1e-9


def test_minmax_constant_is_zero():
    out = _minmax(np.array([3.0, 3.0, 3.0]))
    assert np.allclose(out, 0.0)


@pytest.fixture
def mini_hybrid(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """Мини-индекс: TF-IDF + рукотворные эмбеддинги, embed_query замокан."""
    chunks = [
        {"chunk_id": "0_0", "doc_id": "0", "name": "Diabetes trial",
         "text": "insulin glucose diabetes blood sugar"},
        {"chunk_id": "1_0", "doc_id": "1", "name": "Glaucoma trial",
         "text": "eye pressure vision glaucoma"},
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    vectorizer_path = tmp_path / "vectorizer.pkl"
    matrix_path = tmp_path / "matrix.npz"
    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)
    scipy.sparse.save_npz(matrix_path, matrix)

    # L2-нормированные эмбеддинги: doc0=[1,0], doc1=[0,1]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    # запрос семантически совпадает с doc0
    monkeypatch.setattr(hr, "embed_query", lambda q: np.array([1.0, 0.0], dtype="float32"))

    return {
        "vectorizer_path": vectorizer_path,
        "matrix_path": matrix_path,
        "embeddings_path": embeddings_path,
        "chunks_path": chunks_path,
    }


def test_hybrid_returns_score_components(mini_hybrid):
    r = HybridRetriever(**mini_hybrid, alpha=0.5)
    results = r.search("insulin", k=2)
    assert len(results) == 2
    for hit in results:
        assert {"score", "lex_score", "sem_score"} <= set(hit)


def test_hybrid_pure_semantic_ranks_match_first(mini_hybrid):
    # alpha=1.0 -> только семантика; запрос без лексических пересечений
    r = HybridRetriever(**mini_hybrid, alpha=1.0)
    results = r.search("zzz", k=2)
    assert results[0]["doc_id"] == "0"


def test_hybrid_empty_query_returns_empty(mini_hybrid):
    r = HybridRetriever(**mini_hybrid)
    assert r.search("") == []


def test_hybrid_size_mismatch_raises(tmp_path, mini_hybrid):
    # подменяем эмбеддинги на неверный размер -> ошибка согласованности
    bad = tmp_path / "bad_emb.npy"
    np.save(bad, np.zeros((5, 2), dtype="float32"))
    args = {**mini_hybrid, "embeddings_path": bad}
    with pytest.raises(ValueError):
        HybridRetriever(**args)
