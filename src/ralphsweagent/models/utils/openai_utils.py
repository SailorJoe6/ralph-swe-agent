"""Utilities for OpenAI Responses API output parsing."""

from __future__ import annotations

from typing import Any


def _normalize_item(item: Any) -> dict | None:
    if item is None:
        return None
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return None


def _extract_text_from_content(content_items: Any) -> list[str]:
    texts: list[str] = []
    if not content_items:
        return texts
    for entry in content_items:
        normalized = _normalize_item(entry)
        if not isinstance(normalized, dict):
            continue
        entry_type = normalized.get("type")
        if entry_type in {"output_text", "text"}:
            text = normalized.get("text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def coerce_responses_text(output: Any) -> str:
    """Extract concatenated text from Responses API output items."""
    if not output:
        return ""
    texts: list[str] = []
    for item in output:
        normalized = _normalize_item(item)
        if not isinstance(normalized, dict):
            continue
        item_type = normalized.get("type")
        if item_type in {"output_text", "text"}:
            text = normalized.get("text")
            if isinstance(text, str):
                texts.append(text)
        elif item_type == "message":
            texts.extend(_extract_text_from_content(normalized.get("content")))
    return "".join(texts)
