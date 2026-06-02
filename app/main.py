"""Streamlit UI: вопрос -> фрагменты -> ответ (LLM/экстрактив) -> источники.

Переключатели гибридного поиска и LLM-генерации; для каждого источника
показываем скоры (включая лексическую/семантическую компоненты гибрида)
и ссылку на испытание на ClinicalTrials.gov по NCT-идентификатору.
"""

import re

import streamlit as st

from app.config import (
    EMBEDDINGS_NPY,
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    TOP_K,
    USE_HYBRID,
    VECTORIZER_PKL,
)
from app.generator import ask, is_relevant
from app.hybrid_retriever import get_retriever

NCT_RE = re.compile(r"NCT\d{8}")

DEMO_QUESTIONS = [
    "Inclusion criteria for breast cancer surgery trials",
    "COVID-19 trials with hospitalized patients",
    "Eligibility criteria for type 2 diabetes studies using insulin",
    "How do I cook borscht?",  # негативный (офтоп) — ожидаем отказ
]

MODE_BADGE = {"llm": "🤖 LLM", "extractive": "📄 Экстрактив", "refusal": "🚫 Отказ"}


def index_exists() -> bool:
    return all(p.exists() for p in (VECTORIZER_PKL, MATRIX_NPZ, INDEX_CHUNKS_JSONL))


@st.cache_resource
def load_retriever(use_hybrid: bool):
    return get_retriever(use_hybrid)


def nct_url(name: str) -> str | None:
    m = NCT_RE.search(name or "")
    return f"https://clinicaltrials.gov/study/{m.group(0)}" if m else None


def render_source(i: int, src: dict, expanded: bool) -> None:
    label = f"[{i}] doc_id={src['doc_id']} · score={src['score']:.4f}"
    if "sem_score" in src:
        label += f" · lex={src['lex_score']:.3f} · sem={src['sem_score']:.3f}"
    with st.expander(label, expanded=expanded):
        st.markdown(f"**{src['name']}**")
        url = nct_url(src["name"])
        if url:
            st.markdown(f"[Открыть на ClinicalTrials.gov]({url})")
        st.text(src["text"])


def main() -> None:
    st.set_page_config(page_title="Clinical Trials RAG", layout="wide")
    st.title("Clinical Trials RAG")
    st.caption(
        "Учебный RAG по критериям клинических исследований: "
        "TF-IDF + семантика (гибрид) + LLM, с источниками"
    )

    if not index_exists():
        st.error(
            "Индекс не собран. Сначала выполните:\n\n"
            "`uv run python scripts/build_index.py`"
        )
        st.stop()

    # --- Настройки ---
    st.sidebar.header("Настройки")
    has_emb = EMBEDDINGS_NPY.exists()
    use_hybrid = st.sidebar.checkbox(
        "Гибридный поиск (TF-IDF + эмбеддинги)",
        value=USE_HYBRID and has_emb,
        disabled=not has_emb,
    )
    if not has_emb:
        st.sidebar.caption("Эмбеддинги не найдены — доступен только TF-IDF.")

    llm_available = bool(OPENROUTER_API_KEY)
    use_llm = st.sidebar.checkbox(
        f"LLM-генерация ({OPENROUTER_MODEL})",
        value=llm_available,
        disabled=not llm_available,
    )
    if not llm_available:
        st.sidebar.caption("OPENROUTER_API_KEY не задан — ответ экстрактивный (фолбэк).")

    top_k = st.sidebar.slider("top-k", min_value=1, max_value=10, value=TOP_K)

    st.sidebar.header("Demo-вопросы")
    for q in DEMO_QUESTIONS:
        if st.sidebar.button(q, use_container_width=True):
            st.session_state["question"] = q

    # --- Запрос ---
    question = st.text_input("Ваш вопрос", key="question")

    if st.button("Спросить", type="primary"):
        if not question.strip():
            st.warning("Введите вопрос.")
            st.stop()

        with st.spinner("Поиск и генерация..."):
            result = ask(
                question.strip(),
                k=top_k,
                retriever=load_retriever(use_hybrid),
                use_llm=use_llm,
            )

        st.subheader("Ответ")
        badge = MODE_BADGE.get(result["mode"], result["mode"])
        st.caption(f"Режим: {badge} · Поиск: {'гибрид' if use_hybrid else 'TF-IDF'}")
        if result["mode"] == "llm":
            st.markdown(result["answer"])
        else:
            st.text(result["answer"])

        st.subheader("Источники (top-k)")
        sources = result["sources"]
        if not sources:
            st.info("Источники не найдены.")
        else:
            for i, src in enumerate(sources, 1):
                render_source(i, src, expanded=is_relevant(src))


if __name__ == "__main__":
    main()
