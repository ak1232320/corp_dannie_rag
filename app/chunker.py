"""Chunking: documents.jsonl → chunks.jsonl."""

import json
from pathlib import Path

from app.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP, DOCUMENTS_JSONL, CHUNKS_JSONL


def split_paragraphs(text: str) -> list[str]:
    """Разбивает текст на непустые абзацы."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Длинный абзац без переносов — жёсткая нарезка; overlap добавляется позже."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        start = end
    return parts


def apply_overlap(chunks: list[str], overlap: int, max_chars: int) -> list[str]:
    """Добавляет overlap из предыдущего чанка в начало следующего."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prefix = chunks[i - 1][-overlap:]
        combined = prefix + chunks[i]
        if len(combined) > max_chars:
            combined = prefix + chunks[i][: max_chars - len(prefix)]
        result.append(combined)
    return result


def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Нарезка по абзацам с ограничением длины и overlap между чанками."""
    if not text.strip():
        return []

    raw_chunks: list[str] = []
    current_parts: list[str] = []

    def flush() -> None:
        if current_parts:
            raw_chunks.append("\n\n".join(current_parts))
            current_parts.clear()

    for para in split_paragraphs(text):
        for piece in split_long_text(para, max_chars):
            candidate_parts = current_parts + [piece]
            candidate = "\n\n".join(candidate_parts)
            if len(candidate) <= max_chars:
                current_parts = candidate_parts
            else:
                flush()
                if len(piece) <= max_chars:
                    current_parts = [piece]
                else:
                    raw_chunks.extend(split_long_text(piece, max_chars))

    flush()
    return apply_overlap(raw_chunks, overlap, max_chars)


def chunk_document(doc: dict) -> list[dict]:
    """Один документ → список чанков с метаданными."""
    chunks = []
    for i, text in enumerate(chunk_text(doc["text"])):
        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_{i}",
                "doc_id": doc["doc_id"],
                "name": doc["name"],
                "text": text,
            }
        )
    return chunks


def load_documents(path: Path) -> list[dict]:
    documents = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def write_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def run(
    input_path: Path = DOCUMENTS_JSONL,
    output_path: Path = CHUNKS_JSONL,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл: {input_path}")

    documents = load_documents(input_path)
    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    write_chunks(all_chunks, output_path)
    return len(all_chunks)


def main() -> None:
    count = run()
    print(f"Записано {count} чанков -> {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
