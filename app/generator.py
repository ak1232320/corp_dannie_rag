"""Ответ по top-k чанкам: LLM-генерация (если доступна) или экстрактив, плюс источники.

Логика отказа опирается на абсолютную релевантность:
- для TF-IDF-хитов — по score (косинус) >= MIN_SCORE;
- для гибридных хитов — по «сырым» компонентам: lex >= MIN_SCORE ИЛИ sem >= SEM_MIN_SCORE.
Если релевантных фрагментов нет — отказываемся (LLM не вызываем).
"""

from app import llm
from app.config import TOP_K
from app.hybrid_retriever import get_retriever
from app.prompts import (
    LEX_STRONG_SCORE,
    MIN_SCORE,
    REFUSAL_EMPTY_QUESTION,
    REFUSAL_NO_CONTEXT,
    SEM_MIN_SCORE,
)


def is_relevant(hit: dict) -> bool:
    """Релевантен ли фрагмент (по абсолютным скорам).

    Гибрид: семантика — надёжный сигнал на этом корпусе; лексика учитывается
    только как сильный довод (офтоп редко даёт высокий лексический скор).
    Чистый TF-IDF: как в эталоне, по порогу MIN_SCORE.
    """
    if "sem_score" in hit:  # гибридный хит
        return hit["sem_score"] >= SEM_MIN_SCORE or hit["lex_score"] >= LEX_STRONG_SCORE
    return hit.get("score", 0.0) >= MIN_SCORE


def relevant_hits(hits: list[dict]) -> list[dict]:
    return [h for h in hits if is_relevant(h)]


def build_answer(hits: list[dict]) -> str:
    """Экстрактивный ответ только из релевантных чанков (как в эталоне)."""
    relevant = relevant_hits(hits)
    if not relevant:
        return REFUSAL_NO_CONTEXT

    parts = ["На основании найденных фрагментов:"]
    for i, hit in enumerate(relevant, 1):
        parts.append(f"\n[{i}] {hit['name']}")
        parts.append(f"doc_id={hit['doc_id']}, score={hit['score']:.2f}")
        parts.append(hit["text"])
    return "\n".join(parts)


def format_sources(hits: list[dict]) -> list[dict]:
    sources = []
    for hit in hits:
        src = {
            "doc_id": hit["doc_id"],
            "name": hit.get("name", ""),
            "text": hit["text"],
            "score": hit["score"],
        }
        if "sem_score" in hit:  # прозрачность гибрида
            src["lex_score"] = hit["lex_score"]
            src["sem_score"] = hit["sem_score"]
        sources.append(src)
    return sources


def ask(
    question: str,
    k: int = TOP_K,
    retriever=None,
    use_hybrid: bool | None = None,
    use_llm: bool | None = None,
) -> dict:
    """Вопрос -> ответ, источники и режим ('llm' | 'extractive' | 'refusal')."""
    if not question.strip():
        return {"answer": REFUSAL_EMPTY_QUESTION, "sources": [], "mode": "refusal"}

    r = retriever or get_retriever(use_hybrid)
    hits = r.search(question.strip(), k=k)
    relevant = relevant_hits(hits)

    if not relevant:
        return {
            "answer": REFUSAL_NO_CONTEXT,
            "sources": format_sources(hits),
            "mode": "refusal",
        }

    mode = "extractive"
    answer = None
    if llm.llm_enabled(use_llm):
        answer = llm.generate(question.strip(), relevant)
        if answer:
            mode = "llm"
    if answer is None:
        answer = build_answer(hits)

    return {"answer": answer, "sources": format_sources(hits), "mode": mode}
