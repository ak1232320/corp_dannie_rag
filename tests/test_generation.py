"""Тесты генерации: релевантность, включение LLM, graceful fallback ask()."""

from app import llm
from app.generator import ask, is_relevant
from app.prompts import REFUSAL_EMPTY_QUESTION, REFUSAL_NO_CONTEXT


class HitsRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=3):
        return self.hits[:k]


# --- is_relevant ---

def test_is_relevant_tfidf():
    assert is_relevant({"doc_id": "1", "score": 0.2})
    assert not is_relevant({"doc_id": "1", "score": 0.05})


def test_is_relevant_hybrid_by_strong_lexical():
    assert is_relevant({"score": 0.9, "lex_score": 0.5, "sem_score": 0.1})


def test_is_relevant_hybrid_by_semantic():
    assert is_relevant({"score": 0.9, "lex_score": 0.0, "sem_score": 0.5})


def test_is_relevant_hybrid_neither():
    # средний лексический скор офтопа (0.38) и низкая семантика -> не релевантно
    assert not is_relevant({"score": 0.9, "lex_score": 0.38, "sem_score": 0.2})


# --- llm_enabled ---

def test_llm_disabled_without_key(monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm, "USE_LLM", "auto")
    assert llm.llm_enabled() is False


def test_llm_enabled_with_key_auto(monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "USE_LLM", "auto")
    assert llm.llm_enabled() is True


# --- ask(): отказ / экстрактив / пустой вопрос ---

def test_ask_empty_question():
    res = ask("   ", retriever=HitsRetriever([]))
    assert res["answer"] == REFUSAL_EMPTY_QUESTION
    assert res["mode"] == "refusal"


def test_ask_refuses_on_low_scores():
    r = HitsRetriever([{"doc_id": "1", "name": "X", "text": "t", "score": 0.01}])
    res = ask("вопрос", retriever=r)
    assert res["answer"] == REFUSAL_NO_CONTEXT
    assert res["mode"] == "refusal"


def test_ask_extractive_without_key(monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "")
    r = HitsRetriever(
        [{"doc_id": "1", "name": "Trial X", "text": "relevant context text", "score": 0.5}]
    )
    res = ask("вопрос", retriever=r)
    assert res["mode"] == "extractive"
    assert "relevant context text" in res["answer"]
