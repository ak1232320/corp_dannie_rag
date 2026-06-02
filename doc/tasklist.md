# Tasklist (план работ и статус)

Итерации по мотивам `homework/01_implementation` эталона. Все пункты выполнены.

## Планирование
- [x] Идея проекта — [00_project_idea.md](00_project_idea.md)
- [x] Vision и объём — [vision.md](vision.md)
- [x] Соглашения — [conventions.md](conventions.md)
- [x] Tasklist (этот файл) и [workflow.md](workflow.md)

## Реализация (iter 0–8)
- [x] **iter 0 — Каркас:** структура, `app/config.py`, `.gitignore`, `.env.example`, `pyproject.toml`, локальный `.venv`.
- [x] **iter 1 — Данные:** `scripts/prepare_datasets.py` — выборка 1200 испытаний из Clinical Trials → `datasets.json`.
- [x] **iter 2 — Ингест:** `scripts/ingest.py` → `documents.jsonl` (нормализация текста).
- [x] **iter 3 — Чанкинг:** `app/chunker.py` → `chunks.jsonl` (8585 чанков, `max=400`, `overlap=50`).
- [x] **iter 4 — Индекс:** `scripts/build_index.py` + `app/embedder.py` → TF-IDF (`matrix.npz`) + эмбеддинги (`embeddings.npy`).
- [x] **iter 5 — Поиск:** `app/retriever.py` (TF-IDF) + `app/hybrid_retriever.py` (гибрид) + `scripts/check_retrieval.py`.
- [x] **iter 6 — Ответ:** `app/prompts.py`, `app/generator.py`, `app/llm.py` (OpenRouter + фолбэк) + `scripts/check_generator.py`.
- [x] **iter 7 — UI:** `app/main.py` (Streamlit) — тумблеры гибрид/LLM, скоры, источники, демо-вопросы.
- [x] **iter 8 — Оценка, тесты, README:** `scripts/evaluate.py`, `eval/eval_queries.json`, 34 теста, `README.md`.

## Улучшения (≥2 для «Отлично»)
- [x] **Гибридный поиск** (TF-IDF + семантические эмбеддинги)
- [x] **Оценка качества** (retrieval@k / hit-rate / MRR, TF-IDF vs гибрид)
- [x] **LLM-генерация** (OpenRouter, graceful fallback)

## Сдача
- [x] `homework/IMPROVEMENTS.md`
- [x] `homework/SUBMISSION.md` — ссылка на репозиторий вписана
- [x] Опубликовать репозиторий: https://github.com/ak1232320/corp_dannie_rag
- [ ] Открыть PR в `MaratNotes/rag-tutorial` (правка их `homework/SUBMISSION.md` со ссылкой)
