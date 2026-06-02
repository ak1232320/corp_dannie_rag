"""Гибридный поиск: слияние лексического (TF-IDF) и семантического (эмбеддинги) скоров.

Зачем: TF-IDF находит точные совпадения слов, но не ловит синонимы/перефраз;
эмбеддинги ловят смысл, но иногда промахиваются по точным терминам/кодам.
Комбинация устойчивее к разным типам запросов.

Как: оба «сырых» косинуса нормируются min-max по всем чанкам (чтобы привести
к одной шкале), затем линейно смешиваются:  score = alpha*sem + (1-alpha)*lex.
Ранжируем по смешанному скору, но в результат кладём и «сырые» компоненты
(lex_score/sem_score) — по ним генератор принимает решение об отказе
(min-max теряет абсолютную релевантность, а отказ должен опираться на неё).
"""

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from app.chunker import load_documents
from app.config import (
    EMBEDDINGS_NPY,
    HYBRID_ALPHA,
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    TOP_K,
    USE_HYBRID,
    VECTORIZER_PKL,
)
from app.embedder import embed_query


def _minmax(x: np.ndarray) -> np.ndarray:
    """Нормировка в [0, 1]; если все значения равны — нули."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    def __init__(
        self,
        vectorizer_path: Path = VECTORIZER_PKL,
        matrix_path: Path = MATRIX_NPZ,
        embeddings_path: Path = EMBEDDINGS_NPY,
        chunks_path: Path = INDEX_CHUNKS_JSONL,
        alpha: float = HYBRID_ALPHA,
    ) -> None:
        self.alpha = alpha
        with Path(vectorizer_path).open("rb") as f:
            self.vectorizer = pickle.load(f)
        self.matrix = scipy.sparse.load_npz(matrix_path)
        self.embeddings = np.load(embeddings_path)
        self.chunks = load_documents(Path(chunks_path))

        if not (self.matrix.shape[0] == len(self.chunks) == self.embeddings.shape[0]):
            raise ValueError("Размеры матрицы / эмбеддингов / чанков не совпадают")

    def search(self, query: str, k: int = TOP_K, alpha: float | None = None) -> list[dict]:
        if not query.strip():
            return []

        a = self.alpha if alpha is None else alpha
        k = min(k, len(self.chunks))
        q = query.strip()

        lex = cosine_similarity(self.vectorizer.transform([q]), self.matrix).flatten()
        # эмбеддинги L2-нормированы -> скалярное произведение = косинус
        sem = (self.embeddings @ embed_query(q)).astype(float)

        fused = a * _minmax(sem) + (1 - a) * _minmax(lex)

        top_indices = fused.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "name": chunk["name"],
                    "score": float(fused[idx]),
                    "lex_score": float(lex[idx]),
                    "sem_score": float(sem[idx]),
                }
            )
        return results


def get_retriever(use_hybrid: bool | None = None):
    """Фабрика: гибридный ретривер, если включён и есть эмбеддинги, иначе TF-IDF."""
    from app.retriever import Retriever

    use = USE_HYBRID if use_hybrid is None else use_hybrid
    if use and EMBEDDINGS_NPY.exists():
        return HybridRetriever()
    return Retriever()
