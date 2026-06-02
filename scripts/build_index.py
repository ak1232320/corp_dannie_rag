"""Build index: ingest + chunk + TF-IDF + семантические эмбеддинги, сохранение артефактов.

Расширяет эталонный build_index: помимо TF-IDF-матрицы считает и сохраняет
матрицу эмбеддингов (embeddings.npy) для гибридного/семантического поиска.
"""

import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.chunker import load_documents, run as chunk_run
from app.config import (
    CHUNKS_JSONL,
    DATA_INDEX,
    EMBEDDINGS_NPY,
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    VECTORIZER_PKL,
)
from app.embedder import embed_texts
from ingest import run as ingest_run


def build_tfidf(texts: list[str]) -> tuple[TfidfVectorizer, scipy.sparse.csr_matrix]:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_index(
    vectorizer: TfidfVectorizer,
    matrix: scipy.sparse.csr_matrix,
    embeddings: np.ndarray,
    chunks_path: Path = CHUNKS_JSONL,
) -> int:
    DATA_INDEX.mkdir(parents=True, exist_ok=True)

    with VECTORIZER_PKL.open("wb") as f:
        pickle.dump(vectorizer, f)

    scipy.sparse.save_npz(MATRIX_NPZ, matrix)
    np.save(EMBEDDINGS_NPY, embeddings)
    shutil.copy2(chunks_path, INDEX_CHUNKS_JSONL)
    return matrix.shape[0]


def run() -> int:
    doc_count = ingest_run()
    chunk_count = chunk_run()
    chunks = load_documents(CHUNKS_JSONL)
    texts = [c["text"] for c in chunks]

    if not texts:
        raise ValueError("Нет чанков для индексации")

    vectorizer, matrix = build_tfidf(texts)
    print(f"TF-IDF матрица: {matrix.shape}. Считаю эмбеддинги для {len(texts)} чанков…")
    embeddings = embed_texts(texts)

    save_index(vectorizer, matrix, embeddings)
    print(
        f"Документов: {doc_count}, чанков: {chunk_count}, "
        f"TF-IDF: {matrix.shape}, эмбеддинги: {embeddings.shape}"
    )
    return chunk_count


def main() -> None:
    run()
    print(f"Индекс сохранён -> {DATA_INDEX}")


if __name__ == "__main__":
    main()
