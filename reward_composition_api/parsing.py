from __future__ import annotations

import ast
import json
from typing import Any


def parse_int_tuple(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)

    text = str(value).strip()
    if not text:
        return ()
    text = text.replace("x", ",")
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if any(item <= 0 for item in values):
        raise ValueError("sizes must be positive integers")
    return values


def parse_key_value_mapping(value: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None or isinstance(value, dict):
        return value

    text = value.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _parse_loose_mapping(text)

    if not isinstance(parsed, dict):
        raise ValueError("expected a mapping")
    return {str(key): item for key, item in parsed.items()}


def _parse_loose_mapping(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    if not text:
        return {}

    result: dict[str, Any] = {}
    for item in _split_top_level(text, ","):
        key, raw_value = _split_key_value(item)
        result[_clean_key(key)] = _parse_scalar(raw_value)
    return result


def _split_key_value(item: str) -> tuple[str, str]:
    for separator in (":", "="):
        parts = _split_top_level(item, separator, maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"expected key:value in '{item}'")


def _split_top_level(text: str, separator: str, maxsplit: int | None = None) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    splits = 0

    for index, char in enumerate(text):
        if quote:
            if char == quote and text[index - 1 : index] != "\\":
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth -= 1
            continue
        if char == separator and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
            splits += 1
            if maxsplit is not None and splits >= maxsplit:
                break

    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _clean_key(value: str) -> str:
    return value.strip().strip("'\"")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip("'\"")
