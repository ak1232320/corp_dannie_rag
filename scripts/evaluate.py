"""Оценка качества ретривала (улучшение 2): retrieval@k / hit-rate@k / MRR.

Сравнивает baseline (TF-IDF) и гибридный поиск на трёх типах запросов:
  - title       — заголовок испытания -> его же doc_id (лексически «лёгкие»);
  - paraphrase  — описание заболевания своими словами -> любой doc этого
                  заболевания (здесь проявляется польза семантики/гибрида);
  - negative    — офтоп-вопрос -> система должна «отказаться» (нет релевантных).

Набор сохраняется в eval/eval_queries.json для воспроизводимости.
Запуск:  uv run python scripts/evaluate.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunker import load_documents
from app.config import (
    DOC_CONDITIONS,
    DOCUMENTS_JSONL,
    EVAL_QUERIES,
    RANDOM_SEED,
    TOP_K,
)
from app.generator import is_relevant

N_TITLE_QUERIES = 40

# Перефраз заболевания «своими словами» (без терминов из заголовков) -> заболевание.
PARAPHRASE_QUERIES = {
    "type 2 diabetes": "treatment for high blood sugar disease",
    "rheumatoid arthritis": "studies on joint inflammation and morning stiffness",
    "glaucoma": "increased pressure inside the eye causing vision loss",
    "covid-19": "coronavirus respiratory infection in patients",
    "chronic obstructive pulmonary disease": "smoking related chronic lung breathing disease",
    "sickle cell anemia": "inherited red blood cell disorder with pain crises",
    "anxiety": "excessive worry, fear and panic disorder",
    "breast cancer": "malignant tumor of the breast in women",
}

# Те же темы по-русски: у TF-IDF нет общих токенов с англоязычным корпусом,
# поэтому лексика проваливается, а мультиязычные эмбеддинги находят нужное —
# наглядная демонстрация пользы семантики/гибрида.
RU_PARAPHRASE_QUERIES = {
    "type 2 diabetes": "сахарный диабет 2 типа лечение инсулином",
    "rheumatoid arthritis": "ревматоидный артрит воспаление суставов",
    "glaucoma": "повышенное внутриглазное давление и потеря зрения",
    "covid-19": "коронавирусная инфекция у госпитализированных пациентов",
    "chronic obstructive pulmonary disease": "хроническая обструктивная болезнь лёгких у курильщиков",
    "sickle cell anemia": "серповидноклеточная анемия и болевые кризы",
    "anxiety": "тревожное расстройство, паника и беспокойство",
    "breast cancer": "злокачественная опухоль молочной железы у женщин",
}

NEGATIVE_QUERY = "How do I cook borscht at home?"


# ----------------------- метрики (чистые функции, тестируемы) -----------------------

def first_rank(ranked_doc_ids: list[str], relevant: set[str]) -> int | None:
    """1-based ранг первого релевантного документа, либо None."""
    for i, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant:
            return i
    return None


def evaluate_retriever(retriever, queries: list[dict], k: int) -> dict:
    """hit-rate@k и MRR@k по позитивным запросам (с непустым relevant_doc_ids)."""
    positives = [q for q in queries if q["relevant_doc_ids"]]
    hits = 0
    rr_sum = 0.0
    for q in positives:
        results = retriever.search(q["query"], k=k)
        ranked = [r["doc_id"] for r in results]
        rank = first_rank(ranked, set(q["relevant_doc_ids"]))
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
    n = len(positives)
    return {
        "n": n,
        "hit_rate": hits / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
    }


def evaluate_negative(retriever, query: str, k: int) -> bool:
    """True, если система корректно «отказывается» (нет релевантных хитов)."""
    results = retriever.search(query, k=k)
    return not any(is_relevant(h) for h in results)


# ----------------------------- построение набора запросов -----------------------------

def title_of(name: str) -> str:
    return name.split(" [")[0].strip()


def build_queries() -> list[dict]:
    docs = load_documents(DOCUMENTS_JSONL)
    rng = random.Random(RANDOM_SEED)

    queries: list[dict] = []
    for d in rng.sample(docs, min(N_TITLE_QUERIES, len(docs))):
        queries.append(
            {"query": title_of(d["name"]), "relevant_doc_ids": [d["doc_id"]], "type": "title"}
        )

    if DOC_CONDITIONS.exists():
        cond_map = json.loads(DOC_CONDITIONS.read_text(encoding="utf-8"))
        by_cond: dict[str, list[str]] = {}
        for doc_id, cond in cond_map.items():
            by_cond.setdefault(cond, []).append(doc_id)
        for cond, q in PARAPHRASE_QUERIES.items():
            ids = by_cond.get(cond, [])
            if ids:
                queries.append({"query": q, "relevant_doc_ids": ids, "type": "paraphrase"})
        for cond, q in RU_PARAPHRASE_QUERIES.items():
            ids = by_cond.get(cond, [])
            if ids:
                queries.append({"query": q, "relevant_doc_ids": ids, "type": "ru_paraphrase"})

    queries.append({"query": NEGATIVE_QUERY, "relevant_doc_ids": [], "type": "negative"})
    return queries


def _by_type(queries: list[dict], t: str) -> list[dict]:
    return [q for q in queries if q["type"] == t]


def main() -> None:
    from app.hybrid_retriever import HybridRetriever
    from app.retriever import Retriever

    queries = build_queries()
    EVAL_QUERIES.parent.mkdir(parents=True, exist_ok=True)
    EVAL_QUERIES.write_text(
        json.dumps({"queries": queries}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    k = TOP_K
    retrievers = {"TF-IDF": Retriever(), "Hybrid": HybridRetriever()}

    print(f"Набор: {len(queries)} запросов  (k={k})  -> {EVAL_QUERIES.name}\n")
    header = f"{'Retriever':<10} {'set':<11} {'n':>4} {'hit@k':>8} {'MRR@k':>8}"
    print(header)
    print("-" * len(header))
    for name, retr in retrievers.items():
        for label, subset in (
            ("all", [q for q in queries if q["relevant_doc_ids"]]),
            ("title", _by_type(queries, "title")),
            ("paraphrase", _by_type(queries, "paraphrase")),
            ("ru_paraphrase", _by_type(queries, "ru_paraphrase")),
        ):
            m = evaluate_retriever(retr, subset, k)
            print(f"{name:<10} {label:<11} {m['n']:>4} {m['hit_rate']:>8.3f} {m['mrr']:>8.3f}")
        ok = evaluate_negative(retr, NEGATIVE_QUERY, k)
        print(f"{name:<10} {'negative':<11} {'1':>4}  отказ={'✓' if ok else '✗'}")
        print("-" * len(header))


if __name__ == "__main__":
    main()
