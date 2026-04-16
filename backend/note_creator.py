"""Structured note generation from web sources via LLM."""
from __future__ import annotations
from typing import Literal

from .knowledge_graph import GRAPH, extract_relations_from_note_body, parse_entity_relations
from .knowledge_manager import KM, _slug
from .llm import TaskType, router
from .models import Category, Note
from .tools.web_search import fetch_url, web_search

NOTE_SYSTEM = """Ты — аккуратный ведущий технических заметок.
Задача: сжать сырые источники в краткую структурированную заметку в формате ниже.
Только ключевые факты, цифры, ограничения — БЕЗ воды и обобщений.
Пиши на русском языке."""

NOTE_TEMPLATE = """# {topic}

## Что это
(краткое, фактическое определение)

## Ключевые параметры
(цифры, лимиты, спецификации)

## Практические заметки
(реальные советы по применению)

## Частые ошибки
(что идёт не так и как починить)

## Причинно-следственные связи
(формат: причина -> следствие)
- X causes Y
- A enables B

## Связанные темы
- [[тема1]]
- [[тема2]]
"""


def _build_prompt(topic: str, sources_text: str, depth: str) -> str:
    detail = "подробно, с примерами и числами" if depth == "deep" else "кратко, только самое главное"
    return f"""Тема: {topic}
Уровень детализации: {detail}

Используй приведённые источники. Создай заметку строго по шаблону:

{NOTE_TEMPLATE}

Источники:
\"\"\"
{sources_text}
\"\"\"

Верни ТОЛЬКО тело заметки (без frontmatter), начиная с "# {topic}".
В конце добавь секцию "## keywords_extracted:" и перечисли через запятую
5-10 ключевых слов для поиска."""


import re as _re

# Patterns for causal relations: "X causes Y", "X -> Y", "X enables Y", etc.
_CAUSAL_PATTERNS = [
    _re.compile(r"[-•]\s*(.+?)\s+causes?\s+(.+)", _re.IGNORECASE),
    _re.compile(r"[-•]\s*(.+?)\s+enables?\s+(.+)", _re.IGNORECASE),
    _re.compile(r"[-•]\s*(.+?)\s+prevents?\s+(.+)", _re.IGNORECASE),
    _re.compile(r"[-•]\s*(.+?)\s+leads?\s+to\s+(.+)", _re.IGNORECASE),
    _re.compile(r"[-•]\s*(.+?)\s*->\s*(.+)"),
    _re.compile(r"[-•]\s*(.+?)\s*→\s*(.+)"),
]


def _extract_causal_edges(body: str) -> list[tuple[str, str, str]]:
    """Extract causal edges from a note body's causal relations section."""
    triples: list[tuple[str, str, str]] = []
    # Find the causal section
    causal_start = -1
    for marker in ("## Причинно-следственные связи", "## Causal", "## causal_relations"):
        idx = body.lower().find(marker.lower())
        if idx >= 0:
            causal_start = idx
            break
    if causal_start < 0:
        return triples

    # Get the section text (up to next ## or end)
    section = body[causal_start:]
    next_section = section.find("\n## ", 3)
    if next_section > 0:
        section = section[:next_section]

    for line in section.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for pattern in _CAUSAL_PATTERNS:
            m = pattern.match(line)
            if m:
                cause = m.group(1).strip().lower().rstrip(".")
                effect = m.group(2).strip().lower().rstrip(".")
                if len(cause) > 2 and len(effect) > 2:
                    # Determine relation type from the pattern
                    if "prevent" in line.lower():
                        rel = "prevents"
                    elif "enable" in line.lower():
                        rel = "enables"
                    else:
                        rel = "causes"
                    triples.append((cause, rel, effect))
                break

    return triples


def _extract_keywords(body: str) -> tuple[str, list[str]]:
    marker = "## keywords_extracted:"
    idx = body.lower().find(marker.lower())
    if idx == -1:
        return body, []
    raw = body[idx + len(marker) :].strip()
    kws = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
    return body[:idx].rstrip(), kws


def learn_topic(
    topic: str,
    *,
    depth: Literal["deep", "quick"] = "quick",
    category: Category = "profession",
    project: str | None = None,
    max_sources: int = 3,
) -> Note:
    """Исследовать тему в вебе и создать заметку."""
    results = web_search(topic, max_results=max_sources)
    sources_text_parts: list[str] = []
    source_urls: list[str] = []
    for r in results:
        source_urls.append(r.url)
        page = fetch_url(r.url, max_chars=5000)
        sources_text_parts.append(f"### {r.title}\n{r.url}\n{r.snippet}\n\n{page}")

    if not sources_text_parts:
        sources_text = f"(веб-источники недоступны — используй общие знания о теме '{topic}')"
    else:
        sources_text = "\n\n---\n\n".join(sources_text_parts)

    body_raw = router().call(
        TaskType.NOTE_CREATION,
        NOTE_SYSTEM, _build_prompt(topic, sources_text, depth),
        max_tokens=2000, temperature=0.2,
    )
    body, keywords = _extract_keywords(body_raw)

    confidence = "partial" if sources_text_parts else "unverified"
    source = "; ".join(source_urls) if source_urls else "internal"

    note = KM.save_note(
        topic=topic,
        body=body,
        category=category,
        keywords=keywords or [topic.lower()],
        source=source,
        confidence=confidence,  # type: ignore
        project=project,
    )

    # Index into knowledge graph — extract entities/relations from note body.
    # This is free: we parse [[wiki-links]], **bold terms**, and causal
    # relations from the note the LLM already created. No extra LLM call.
    try:
        note_slug = _slug(topic)
        triples = extract_relations_from_note_body(body, topic)
        # Extract causal edges from "## Причинно-следственные связи" section
        triples.extend(_extract_causal_edges(body))
        # Also link keywords as entities
        for kw in (keywords or []):
            if kw.lower() != topic.lower():
                triples.append((topic.lower(), "keyword", kw.lower()))
        GRAPH.add_relations(triples, source_note=note_slug)
    except Exception:
        pass  # graph indexing is best-effort

    return note
