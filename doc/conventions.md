# Conventions (соглашения)

## Структура

- `app/` — библиотека (устанавливается как пакет): конфиг, чанкинг, эмбеддинги,
  ретриверы, LLM, генератор, промпты, UI.
- `scripts/` — исполняемые шаги пайплайна и проверки.
- `tests/` — pytest.
- `data/` — `raw/` (входные), `processed/` (jsonl), `index/` (артефакты).
- `eval/`, `doc/`, `homework/` — оценка, документация, сдача.

## Схемы данных (контракты между шагами)

```
datasets.json   { "datasets": [ {"id": int, "name": str, "text": str} ] }
documents.jsonl { "doc_id": str, "name": str, "text": str, "source_file": str }
chunks.jsonl    { "chunk_id": "<doc_id>_<i>", "doc_id": str, "name": str, "text": str }
hit (поиск)     { "text", "doc_id", "name", "score"[, "lex_score", "sem_score"] }
```

`doc_id` — строка; `chunk_id = f"{doc_id}_{i}"`. NCT-id хранится внутри `name`.

## Код и стиль

- Код, имена переменных и пути — **на английском**; документация и комментарии — **на русском**.
- Чистые функции для логики (тестируемость); побочные эффекты — в `main()`/`run()`.
- Параметры — централизованно в [`app/config.py`](../app/config.py); секреты и
  тумблеры — через переменные окружения / `.env` (см. `.env.example`).
- Кодировка файлов — UTF-8 (`ensure_ascii=False` при записи JSON).

## Запуск и окружение

- Только локальное окружение: `uv sync` → `.venv`; запуск через `uv run …`.
- Никаких глобальных установок; `.venv`, `.env`, артефакты `data/index`,
  большой исходник — в `.gitignore`. Коммитим `data/raw/datasets.json`.

## Пороговые значения

`MIN_SCORE=0.15` (TF-IDF), `SEM_MIN_SCORE=0.45` / `LEX_STRONG_SCORE=0.45` (гибрид) —
калиброваны по реальным распределениям скоров (см. `homework/IMPROVEMENTS.md`).
