"""Проверка ответа (итерация 6): гибридный поиск + LLM/экстрактив + корректный отказ."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.generator import ask


def show(label: str, question: str) -> None:
    print(f"\n--- {label}: «{question}» ---")
    result = ask(question)
    print(f"Режим: {result['mode']}")
    print(f"Ответ:\n{result['answer']}\n")
    print(f"Источников: {len(result['sources'])}")
    for i, src in enumerate(result["sources"], 1):
        extra = ""
        if "sem_score" in src:
            extra = f", lex={src['lex_score']:.3f}, sem={src['sem_score']:.3f}"
        print(
            f"  [{i}] doc_id={src['doc_id']}, score={src['score']:.4f}{extra}, "
            f"name={src['name'][:60]}..."
        )


if __name__ == "__main__":
    show("Есть контекст", "Inclusion criteria for breast cancer surgery trials")
    show("Есть контекст", "COVID-19 trials with hospitalized patients")
    show("Negative (офтоп)", "How do I cook borscht?")
