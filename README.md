# Clinical Trials RAG

Учебный Retrieval-Augmented Generation поверх **критериев приемлемости клинических
исследований** (ClinicalTrials.gov, датасет с Kaggle). Повторяет пайплайн
[rag-tutorial](https://github.com/MaratNotes/rag-tutorial) на своих данных
(`данные → чанки → индекс → поиск → ответ`) и добавляет три улучшения:
**гибридный поиск** (TF-IDF + семантические эмбеддинги), **оценку качества**
(retrieval@k / hit-rate / MRR) и **LLM-генерацию** через OpenRouter.

```
datasets.json ──ingest──▶ documents.jsonl ──chunk──▶ chunks.jsonl ──┬─▶ TF-IDF (vectorizer.pkl + matrix.npz)
                                                                     └─▶ эмбеддинги (embeddings.npy)
                                          запрос ──▶ [гибридный поиск top-k] ──▶ [LLM | экстрактив] ──▶ ответ + источники
```

- **Данные:** 1200 испытаний (стратифицированно, 150 на каждое из 8 заболеваний) → **8585 чанков**. Подробно — [doc/DATA.md](doc/DATA.md).
- **Стек:** Python 3.12, `uv`, scikit-learn (TF-IDF), sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), OpenRouter (openai SDK), Streamlit, pytest.
- **Улучшения:** [homework/IMPROVEMENTS.md](homework/IMPROVEMENTS.md).

## Установка и запуск

Всё ставится в **локальное окружение проекта** (`.venv`), глобально ничего не нужно.

```powershell
# 1. Зависимости в локальный .venv
uv sync

# 2. Данные: скачать с Kaggle и распаковать в "data/clinical trial eligibility dataset/"
#    https://www.kaggle.com/datasets/harrachimustapha/clinical-trial-eligibility-criteria-dataset
#    (нужен файл trials_clean.csv)

# 3. Подготовить корпус -> data/raw/datasets.json (1200 записей)
uv run python scripts/prepare_datasets.py

# 4. Собрать индекс (TF-IDF + эмбеддинги); первый запуск скачает модель (~470 МБ)
uv run python scripts/build_index.py

# 5. Запустить UI -> http://localhost:8501
uv run streamlit run app/main.py
```

Опционально — **LLM-генерация** (иначе работает экстрактивный фолбэк):

```powershell
copy .env.example .env       # затем впишите OPENROUTER_API_KEY (ключ: https://openrouter.ai/keys)
```

Проверки и тесты:

```powershell
uv run python scripts/check_retrieval.py    # быстрая проверка поиска (TF-IDF)
uv run python scripts/check_generator.py    # ответ + источники + отказ на офтопе
uv run python scripts/evaluate.py           # метрики качества: TF-IDF vs гибрид
uv run pytest                               # 34 теста
```

## Демонстрация (3 вопроса + 1 негативный)

> Режим `extractive` — без ключа OpenRouter; с ключом ответ генерирует LLM (`llm`).

**1. «Inclusion criteria for breast cancer surgery trials»** → найдено, источники:
- `NCT05981716` Young Breast Cancer Survivors Study (sem=0.73) — Inclusion: female breast cancer survivors, diagnosed < 50 лет; Exclusion: male survivors, …
- `NCT05370300` SNAPS Breast Cancer Patient Study; `NCT00194779` HER2-positive, pre-surgery chemo.

**2. «COVID-19 trials with hospitalized patients»** → найдено, источники:
- `NCT04542694` Favipiravir vs Standard of Care in Hospitalized COVID-19 (sem=0.78)
- `NCT05445934` FB2001 (BRIGHT Study); `NCT04421027` Baricitinib in hospitalized COVID-19.

**3. «Eligibility criteria for type 2 diabetes studies using insulin»** → найдено, источники:
- `Diabetes Self-Management Education` (sem=0.79); `Biphasic Insulin` PK/PD; `Pioglitazone + Metformin`.

**4. «How do I cook borscht?» (негатив, офтоп)** → **отказ**:
```
В базе не найдено релевантных фрагментов. Ответить по данным невозможно.
```
(семантические скоры топ-фрагментов −0.06 / −0.07 / 0.06 — ниже порога 0.45.)

## Оценка качества (retrieval@k, k=3)

`uv run python scripts/evaluate.py` (57 запросов: 40 title + 8 EN-перефраз + 8 RU-перефраз + 1 негатив):

| набор | TF-IDF hit@3 | Hybrid hit@3 |
|---|---|---|
| title | 1.000 | 1.000 |
| paraphrase (EN) | 1.000 | 1.000 |
| **ru_paraphrase** | **0.125** | **1.000** |
| **negative (отказ)** | **✗** | **✓** |
| **итого** | **0.875** | **1.000** |

Вывод: на лексически «лёгких» запросах TF-IDF и гибрид равны, но на
кросс-язычных (русский запрос → англоязычный корпус) и при отказе от офтопа
лексика проваливается, а **семантика/гибрид решают задачу**.

## Структура проекта

```
app/        config, chunker, embedder, retriever, hybrid_retriever, llm, generator, prompts, main(Streamlit)
scripts/    prepare_datasets, ingest, build_index, check_retrieval, check_generator, evaluate
tests/      test_chunking, test_retrieval, test_hybrid, test_eval, test_generation  (34 теста)
data/       raw/datasets.json · processed/*.jsonl · index/{vectorizer.pkl,matrix.npz,embeddings.npy,chunks.jsonl}
eval/       eval_queries.json · doc_conditions.json
doc/        DATA.md + документы планирования (00_project_idea, vision, conventions, tasklist, workflow)
homework/   IMPROVEMENTS.md · SUBMISSION.md
```

## Конфигурация (env / `.env`)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `OPENROUTER_API_KEY` | — | ключ LLM; пусто → экстрактивный фолбэк |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | модель OpenRouter |
| `RAG_EMB_MODEL` | `…/paraphrase-multilingual-MiniLM-L12-v2` | модель эмбеддингов |
| `RAG_HYBRID_ALPHA` | `0.5` | вес семантики в гибридном скоре |
| `RAG_USE_HYBRID` | `1` | гибрид (1) или чистый TF-IDF (0) |
| `RAG_SAMPLE_SIZE` | `1200` | размер выборки испытаний |

Параметры пайплайна (`TOP_K=3`, `CHUNK_MAX_CHARS=400`, `CHUNK_OVERLAP=50`, пороги
`MIN_SCORE=0.15`, `SEM_MIN_SCORE=0.45`) — в [app/config.py](app/config.py) и
[app/prompts.py](app/prompts.py).
