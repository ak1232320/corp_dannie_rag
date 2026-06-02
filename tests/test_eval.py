"""Тесты метрик качества (улучшение 2): first_rank, hit-rate/MRR, отказ."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402


class HitsRetriever:
    """Фейковый ретривер: всегда отдаёт заданные хиты."""

    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=3):
        return self.hits[:k]


def test_first_rank_found():
    assert ev.first_rank(["a", "b", "c"], {"b"}) == 2


def test_first_rank_not_found():
    assert ev.first_rank(["a", "b"], {"z"}) is None


def test_evaluate_retriever_perfect():
    queries = [{"query": "q", "relevant_doc_ids": ["1"], "type": "title"}]
    r = HitsRetriever([{"doc_id": "1", "score": 1.0}, {"doc_id": "2", "score": 0.5}])
    m = ev.evaluate_retriever(r, queries, k=3)
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 1.0


def test_evaluate_retriever_rank_two():
    queries = [{"query": "q", "relevant_doc_ids": ["9"], "type": "title"}]
    r = HitsRetriever([{"doc_id": "1", "score": 1.0}, {"doc_id": "9", "score": 0.5}])
    m = ev.evaluate_retriever(r, queries, k=3)
    assert m["hit_rate"] == 1.0
    assert abs(m["mrr"] - 0.5) < 1e-9


def test_evaluate_retriever_miss():
    queries = [{"query": "q", "relevant_doc_ids": ["42"], "type": "title"}]
    r = HitsRetriever([{"doc_id": "1", "score": 1.0}])
    m = ev.evaluate_retriever(r, queries, k=3)
    assert m["hit_rate"] == 0.0
    assert m["mrr"] == 0.0


def test_evaluate_negative_refuses_on_low_scores():
    r = HitsRetriever([{"doc_id": "1", "score": 0.01}])
    assert ev.evaluate_negative(r, "off-topic", k=3) is True


def test_evaluate_negative_not_refuses_when_relevant():
    r = HitsRetriever([{"doc_id": "1", "score": 0.9}])
    assert ev.evaluate_negative(r, "q", k=3) is False


def test_title_of_strips_suffix():
    assert ev.title_of("Some Trial Name [Breast Cancer] (NCT12345678)") == "Some Trial Name"
