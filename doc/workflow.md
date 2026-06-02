# Workflow (поток данных и процесс разработки)

## Поток данных

```
Kaggle: trials_clean.csv (60k испытаний)
        │  scripts/prepare_datasets.py  (выборка 1200, сборка текста)
        ▼
data/raw/datasets.json            { datasets: [{id,name,text}] }
        │  scripts/ingest.py  (clean_text)
        ▼
data/processed/documents.jsonl    { doc_id,name,text,source_file }
        │  app/chunker.py  (абзацы → ≤400 симв., overlap=50)
        ▼
data/processed/chunks.jsonl       { chunk_id,doc_id,name,text }  (8585)
        │  scripts/build_index.py
        ├─▶ TF-IDF:    data/index/vectorizer.pkl + matrix.npz
        ├─▶ эмбеддинги: data/index/embeddings.npy
        └─▶ копия чанков: data/index/chunks.jsonl

Запрос ──▶ Retriever / HybridRetriever (top-k)
        ──▶ generator.ask: relevant? ─нет─▶ отказ
                            │да
                            ▼
                 llm.generate (OpenRouter)  ─ошибка/нет ключа─▶  экстрактив
                            ▼
                 ответ + источники (UI / CLI)
```

## Команды (всё через локальный `.venv`)

```powershell
uv sync                                   # окружение
uv run python scripts/prepare_datasets.py # данные -> datasets.json
uv run python scripts/build_index.py      # индекс (TF-IDF + эмбеддинги)
uv run python scripts/check_retrieval.py  # проверка поиска
uv run python scripts/check_generator.py  # проверка ответа/отказа
uv run python scripts/evaluate.py         # метрики качества
uv run pytest                             # тесты
uv run streamlit run app/main.py          # UI: http://localhost:8501
```

## Цикл разработки

1. Изменил параметр/код → при необходимости пересобрал индекс (`build_index.py`).
2. Быстрая проверка: `check_retrieval` / `check_generator`.
3. Числовая проверка качества: `evaluate.py` (TF-IDF vs гибрид).
4. Регрессии: `pytest`.
5. Демонстрация: `streamlit`.

Артефакты (`data/index`, `data/processed`) генерируются скриптами и не коммитятся;
для воспроизведения достаточно `datasets.json` + два скрипта сборки.
