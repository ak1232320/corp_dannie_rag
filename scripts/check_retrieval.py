"""Проверка retrieval (итерация 5) — понятный вывод (TF-IDF baseline)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever import Retriever


def print_hit(i: int, hit: dict) -> None:
    preview = hit["text"][:120].replace("\n", " ")
    print(f"  [{i}] doc_id={hit['doc_id']}, score={hit['score']:.4f}")
    print(f"      {preview}...")


def main() -> None:
    print("=== Проверка Retriever (итерация 5, TF-IDF) ===\n")

    r = Retriever()
    print("OK: индекс загружен (vectorizer.pkl + matrix.npz + chunks.jsonl)\n")

    queries = [
        ("breast cancer surgery inclusion criteria", "термины есть в данных -> score > 0"),
        ("how to cook borscht", "офтоп -> низкий score, но top-k всё равно возвращается"),
    ]

    for query, hint in queries:
        print(f"Запрос: «{query}»")
        print(f"Ожидание: {hint}")
        results = r.search(query, k=3)
        print(f"Получено результатов: {len(results)}")
        for i, hit in enumerate(results, 1):
            print_hit(i, hit)
        print()

    print("=== Итог ===")
    print("Если видите 3 результата с полями doc_id / score / text — итерация 5 работает.")


if __name__ == "__main__":
    main()
