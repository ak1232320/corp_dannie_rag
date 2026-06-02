"""LLM-генерация ответа через OpenRouter (OpenAI-совместимый API).

Улучшение 3. Безопасный фолбэк: если ключа нет или вызов упал — возвращаем None,
и генератор откатывается на экстрактивный ответ. Приложение всегда работает.
"""

from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    USE_LLM,
)
from app.prompts import LLM_PROMPT_TEMPLATE, SYSTEM_RULES


def llm_enabled(override: bool | None = None) -> bool:
    """Включена ли LLM-генерация.

    override:  None  -> по настройке USE_LLM ("auto"|"1"|"0");
               True/False -> явно (но всё равно нужен ключ).
    """
    if override is not None:
        return bool(override) and bool(OPENROUTER_API_KEY)
    if USE_LLM == "0":
        return False
    if USE_LLM == "1":
        return True
    return bool(OPENROUTER_API_KEY)  # "auto"


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h['name']} (doc_id={h['doc_id']})\n{h['text']}")
    return "\n\n".join(blocks)


def generate(question: str, hits: list[dict], model: str | None = None) -> str | None:
    """Ответ LLM по контексту, либо None при отсутствии ключа/ошибке (фолбэк выше)."""
    if not OPENROUTER_API_KEY:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        user = LLM_PROMPT_TEMPLATE.format(
            context=format_context(hits), question=question
        )
        resp = client.chat.completions.create(
            model=model or OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:  # сеть/квоты/ключ — откатываемся на экстрактив
        print(f"[llm] фолбэк на экстрактивный ответ: {exc}")
        return None
