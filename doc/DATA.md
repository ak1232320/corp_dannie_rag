# Данные

## Источник

**Clinical Trial Eligibility Criteria Dataset** (Kaggle):
<https://www.kaggle.com/datasets/harrachimustapha/clinical-trial-eligibility-criteria-dataset>

- **Происхождение:** очищенные и структурированные записи клинических исследований
  с публичного реестра **ClinicalTrials.gov**.
- **Лицензия:** CC BY 4.0 (свободное использование с указанием авторства).
- **Язык:** английский. **Домен:** медицина (критерии включения/исключения в КИ).

### Объём исходника (`statistics/dataset_overview.csv`)

| метрика | значение |
|---|---|
| испытаний (trials) | **60 337** |
| уникальных NCT id | 60 337 |
| заболеваний (condition queries) | 8 |
| eligibility-чанков критериев | 825 507 |

Заболевания: breast cancer, COVID-19, chronic obstructive pulmonary disease,
rheumatoid arthritis, type 2 diabetes, glaucoma, anxiety, sickle cell anemia.

Файлы: `trials_clean.csv` (по испытанию на строку, 24 колонки),
`eligibility_criteria_chunks.csv` (критерии по одному на строку), `statistics/*`.
Большой исходник **не коммитится** (в `.gitignore`); скачивается с Kaggle локально.

## Что берём в корпус

Скрипт [`scripts/prepare_datasets.py`](../scripts/prepare_datasets.py) делает
**стратифицированную по заболеванию выборку**: `SAMPLE_SIZE=1200`, по **150 испытаний
на каждое из 8 заболеваний** (детерминированно, `RANDOM_SEED=42`). Для каждого
испытания собирается читаемый многоабзацный текст из полей:

```
{title}

Condition: … | Phase: … | Study type: … | Sex: … | Age: …–…

Brief summary:
{brief_summary}

Inclusion criteria:
{inclusion_criteria}

Exclusion criteria:
{exclusion_criteria}
```

Результат — `data/raw/datasets.json` в схеме эталона:

```json
{ "datasets": [ { "id": 0, "name": "Title [Condition] (NCT01234567)", "text": "…" }, … ] }
```

`NCT`-идентификатор кладётся в `name` — по нему UI строит ссылку на
`clinicaltrials.gov/study/NCT…`. Сопутствующий файл `eval/doc_conditions.json`
(doc_id → заболевание) используется для оценки качества.

## Охват индексации

| этап | объём |
|---|---|
| записей в `datasets.json` | **1200** |
| документов (`documents.jsonl`) | 1200 |
| чанков (`chunks.jsonl`, `CHUNK_MAX_CHARS=400`, `overlap=50`) | **8585** |
| TF-IDF матрица | 8585 × 18421 |
| эмбеддинги (`paraphrase-multilingual-MiniLM-L12-v2`) | 8585 × 384 |

Распределение по заболеваниям в выборке — ровно по 150 на каждое (8 × 150 = 1200).

## Воспроизведение

```powershell
# 1) положить trials_clean.csv в "data/clinical trial eligibility dataset/"
uv run python scripts/prepare_datasets.py   # -> data/raw/datasets.json (+ eval/doc_conditions.json)
uv run python scripts/build_index.py        # -> data/index/*
```

Размер выборки настраивается: `RAG_SAMPLE_SIZE=3000 uv run python scripts/prepare_datasets.py`.
