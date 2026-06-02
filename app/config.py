"""Конфигурация проекта: пути к данным/индексу и параметры пайплайна.

Базовые параметры (TOP_K, CHUNK_*) — как в эталонном rag-tutorial.
Ниже добавлены параметры трёх улучшений: семантические эмбеддинги + гибридный
поиск, оценка качества и LLM-генерация через OpenRouter. Все они конфигурируются
через переменные окружения (см. .env.example), что упрощает эксперименты.
"""

import os
from pathlib import Path

# .env подгружаем рано, чтобы переменные ниже уже видели значения из файла.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # python-dotenv не установлен или .env отсутствует — не критично
    pass

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INDEX = ROOT / "data" / "index"

RAW_DATASETS = DATA_RAW / "datasets.json"
DOCUMENTS_JSONL = DATA_PROCESSED / "documents.jsonl"
CHUNKS_JSONL = DATA_PROCESSED / "chunks.jsonl"

VECTORIZER_PKL = DATA_INDEX / "vectorizer.pkl"
MATRIX_NPZ = DATA_INDEX / "matrix.npz"
EMBEDDINGS_NPY = DATA_INDEX / "embeddings.npy"
INDEX_CHUNKS_JSONL = DATA_INDEX / "chunks.jsonl"

# --- Источник данных: Clinical Trial Eligibility Criteria (Kaggle, скачан вручную) ---
# https://www.kaggle.com/datasets/harrachimustapha/clinical-trial-eligibility-criteria-dataset
CLINICAL_DIR = ROOT / "data" / "clinical trial eligibility dataset"
CLINICAL_TRIALS_CSV = CLINICAL_DIR / "trials_clean.csv"
CLINICAL_CHUNKS_CSV = CLINICAL_DIR / "eligibility_criteria_chunks.csv"
# Размер стратифицированной по заболеванию подвыборки (в датасете 8 заболеваний).
SAMPLE_SIZE = int(os.getenv("RAG_SAMPLE_SIZE", "1200"))
RANDOM_SEED = int(os.getenv("RAG_RANDOM_SEED", "42"))

# --- Базовые параметры (как в эталоне) ---
TOP_K = 3
CHUNK_MAX_CHARS = 400
CHUNK_OVERLAP = 50

# --- Улучшение 1: семантические эмбеддинги + гибридный поиск ---
# Мультиязычная модель (EN+RU): данные англоязычные, но запросы могут быть и на русском.
EMB_MODEL_NAME = os.getenv(
    "RAG_EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMB_BATCH_SIZE = int(os.getenv("RAG_EMB_BATCH_SIZE", "64"))
# Вес семантики в гибридном скоре: score = alpha*sem + (1-alpha)*lex
HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.5"))
# Использовать гибридный поиск по умолчанию (иначе чистый TF-IDF).
USE_HYBRID = os.getenv("RAG_USE_HYBRID", "1") == "1"

# --- Улучшение 3: LLM-генерация через OpenRouter (OpenAI-совместимый API) ---
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
# "auto" — использовать LLM, если задан ключ; "1" — всегда; "0" — никогда (экстрактив).
USE_LLM = os.getenv("RAG_USE_LLM", "auto")

# --- Улучшение 2: оценка качества ---
EVAL_QUERIES = ROOT / "eval" / "eval_queries.json"
DOC_CONDITIONS = ROOT / "eval" / "doc_conditions.json"  # doc_id -> заболевание
