"""Retrieval: загрузка TF-IDF индекса и поиск top-k по cosine similarity."""

import pickle
from pathlib import Path

import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from app.chunker import load_documents
from app.config import (
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    TOP_K,
    VECTORIZER_PKL,
)


class Retriever:
    def __init__(
        self,
        vectorizer_path: Path = VECTORIZER_PKL,
        matrix_path: Path = MATRIX_NPZ,
        chunks_path: Path = INDEX_CHUNKS_JSONL,
    ) -> None:
        self.vectorizer = self._load_vectorizer(vectorizer_path)
        self.matrix = self._load_matrix(matrix_path)
        self.chunks = load_documents(chunks_path)

        if self.matrix.shape[0] != len(self.chunks):
            raise ValueError("Число строк матрицы не совпадает с числом чанков")

    @staticmethod
    def _load_vectorizer(path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {path}. Запустите: uv run python scripts/build_index.py"
            )
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_matrix(path: Path) -> scipy.sparse.csr_matrix:
        if not path.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {path}. Запустите: uv run python scripts/build_index.py"
            )
        return scipy.sparse.load_npz(path)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        if not query.strip():
            return []

        k = min(k, len(self.chunks))
        query_vec = self.vectorizer.transform([query.strip()])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        top_indices = scores.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "name": chunk["name"],
                    "score": float(scores[idx]),
                }
            )
        return results
