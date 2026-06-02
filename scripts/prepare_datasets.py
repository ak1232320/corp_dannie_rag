"""Готовит data/raw/datasets.json из локального датасета Clinical Trials (Kaggle).

Источник (скачан вручную с Kaggle):
    data/clinical trial eligibility dataset/trials_clean.csv
    — 60 337 клинических испытаний с ClinicalTrials.gov по 8 заболеваниям.

Берём стратифицированную по заболеванию подвыборку (SAMPLE_SIZE, по умолчанию 1200:
~150 на каждое из 8 заболеваний), собираем из полей читаемый текст и приводим к схеме
эталонного rag-tutorial:

    { "datasets": [ {"id": int, "name": str, "text": str}, ... ] }

NCT-идентификатор кладём в name — по нему UI строит ссылку на clinicaltrials.gov.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (
    CLINICAL_TRIALS_CSV,
    DOC_CONDITIONS,
    RANDOM_SEED,
    RAW_DATASETS,
    SAMPLE_SIZE,
)

USECOLS = [
    "source_condition_query",
    "nct_id",
    "title",
    "conditions",
    "study_type",
    "phase",
    "sex",
    "minimum_age",
    "maximum_age",
    "brief_summary",
    "inclusion_criteria",
    "exclusion_criteria",
    "eligibility_criteria",
    "has_eligibility_criteria",
]

MIN_TEXT_LEN = 120  # отбрасываем слишком короткие записи


def _clean(value) -> str:
    """str() с отбрасыванием NaN/пустышек."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def build_text(row: pd.Series) -> str:
    """Собирает читаемый многоабзацный текст одного испытания из полей CSV."""
    title = _clean(row.get("title"))
    conditions = _clean(row.get("conditions")) or _clean(row.get("source_condition_query"))
    phase = _clean(row.get("phase"))
    study_type = _clean(row.get("study_type"))
    sex = _clean(row.get("sex"))
    min_age = _clean(row.get("minimum_age"))
    max_age = _clean(row.get("maximum_age"))
    summary = _clean(row.get("brief_summary"))
    incl = _clean(row.get("inclusion_criteria"))
    excl = _clean(row.get("exclusion_criteria"))
    elig = _clean(row.get("eligibility_criteria"))

    meta_bits = []
    if conditions:
        meta_bits.append(f"Condition: {conditions}")
    if phase:
        meta_bits.append(f"Phase: {phase}")
    if study_type:
        meta_bits.append(f"Study type: {study_type}")
    if sex:
        meta_bits.append(f"Sex: {sex}")
    if min_age or max_age:
        meta_bits.append(f"Age: {min_age}–{max_age}".strip("–"))

    parts: list[str] = []
    if title:
        parts.append(title)
    if meta_bits:
        parts.append(" | ".join(meta_bits))
    if summary:
        parts.append("Brief summary:\n" + summary)
    if incl:
        parts.append("Inclusion criteria:\n" + incl)
    if excl:
        parts.append("Exclusion criteria:\n" + excl)
    if not incl and not excl and elig:
        parts.append("Eligibility criteria:\n" + elig)

    return "\n\n".join(parts).strip()


def make_name(row: pd.Series) -> str:
    """Заголовок записи: 'Title [Condition] (NCT...)'."""
    nct = _clean(row.get("nct_id"))
    title = _clean(row.get("title")) or "(no title)"
    cond = _clean(row.get("conditions")) or _clean(row.get("source_condition_query"))
    name = f"{title} [{cond}]" if cond else title
    if nct:
        name = f"{name} ({nct})"
    return name


def _is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_sample() -> pd.DataFrame:
    if not CLINICAL_TRIALS_CSV.exists():
        raise FileNotFoundError(
            f"Не найден файл данных: {CLINICAL_TRIALS_CSV}\n"
            "Скачайте датасет 'Clinical Trial Eligibility Criteria' с Kaggle и распакуйте "
            "в 'data/clinical trial eligibility dataset/'."
        )

    print(f"Читаю {CLINICAL_TRIALS_CSV.name} (это может занять несколько секунд) …")
    df = pd.read_csv(CLINICAL_TRIALS_CSV, usecols=lambda c: c in USECOLS)

    df = df[df["has_eligibility_criteria"].map(_is_true)]
    df = df[df["title"].notna()]

    n_conditions = max(1, df["source_condition_query"].nunique())
    per_cond = max(1, SAMPLE_SIZE // n_conditions)
    rng = random.Random(RANDOM_SEED)

    picked: list[int] = []
    for _, group in df.groupby("source_condition_query"):
        idx = list(group.index)
        rng.shuffle(idx)
        picked.extend(idx[:per_cond])

    return df.loc[picked]


def main() -> None:
    sample = load_sample()

    datasets: list[dict] = []
    domains: Counter = Counter()
    conditions: dict[str, str] = {}  # doc_id -> заболевание (для оценки качества)
    for _, row in sample.iterrows():
        text = build_text(row)
        if len(text) < MIN_TEXT_LEN:
            continue
        cond = _clean(row.get("source_condition_query"))
        doc_id = str(len(datasets))
        datasets.append(
            {"id": len(datasets), "name": make_name(row), "text": text}
        )
        conditions[doc_id] = cond
        domains[cond] += 1

    RAW_DATASETS.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATASETS.write_text(
        json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    DOC_CONDITIONS.parent.mkdir(parents=True, exist_ok=True)
    DOC_CONDITIONS.write_text(
        json.dumps(conditions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nЗаписано {len(datasets)} записей -> {RAW_DATASETS}")
    print(f"Карта заболеваний -> {DOC_CONDITIONS}")
    print("Распределение по заболеваниям:")
    for cond, n in sorted(domains.items()):
        print(f"  {cond:<28} {n}")


if __name__ == "__main__":
    main()
