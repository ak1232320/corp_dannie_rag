"""Семантические эмбеддинги (sentence-transformers) для гибридного поиска.

Модель кэшируется в памяти, чтобы не загружать её повторно. Возвращаем
L2-нормированные векторы — тогда косинусная близость = скалярное произведение,
что даёт быстрый поиск через одно матричное умножение.
"""

import numpy as np

from app.config import EMB_BATCH_SIZE, EMB_MODEL_NAME

_model = None


def get_model(name: str = EMB_MODEL_NAME):
    """Лениво загружает и кэширует модель эмбеддингов."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(name)
    return _model


def embed_texts(texts: list[str], batch_size: int = EMB_BATCH_SIZE) -> np.ndarray:
    """Кодирует список текстов в матрицу (N, d) из L2-нормированных float32."""
    model = get_model()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
    )
    return np.asarray(emb, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Кодирует один запрос в вектор (d,)."""
    return embed_texts([text])[0]
