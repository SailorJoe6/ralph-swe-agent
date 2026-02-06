"""Context window utilities for model-aware UI and prompts."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from minisweagent import global_config_dir

_BUILTIN_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CONTEXT_WINDOW_FILENAME = "model_context_windows.yaml"
_DATE_SUFFIX_RE = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$")
_SUFFIXES = ("-preview", "-beta", "-latest")
_QUANT_SUFFIX_RE = re.compile(
    r"-(?:fp8|fp16|bf16|int8|int4|awq|gptq|gguf)(?:-[a-z0-9_]+)?$"
)
_Q_SUFFIX_RE = re.compile(r"-q\d+(?:_[a-z0-9_]+)?$")


def get_seed_context_window_path() -> Path:
    return _BUILTIN_CONFIG_DIR / _CONTEXT_WINDOW_FILENAME


def get_live_context_window_path(config_dir: Path | None = None) -> Path:
    base_dir = Path(config_dir) if config_dir is not None else global_config_dir
    return base_dir / _CONTEXT_WINDOW_FILENAME


def ensure_live_context_window_map(config_dir: Path | None = None) -> Path:
    """Ensure the live context window map exists, copying the seed file if needed."""
    live_path = get_live_context_window_path(config_dir)
    if live_path.exists():
        return live_path

    seed_path = get_seed_context_window_path()
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed context window map not found at {seed_path}")

    live_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed_path, live_path)
    return live_path


def load_context_window_map(config_dir: Path | None = None) -> dict[str, int]:
    """Load the live context window map (creating it if missing)."""
    live_path = ensure_live_context_window_map(config_dir)
    data = yaml.safe_load(live_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Context window map must be a mapping of model names to token limits")

    cleaned: dict[str, int] = {}
    for key, value in data.items():
        if key is None:
            continue
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid context window value for {key!r}: {value!r}")
        if int_value <= 0:
            raise ValueError(f"Context window value for {key!r} must be positive")
        cleaned[str(key)] = int_value
    return cleaned


def normalize_model_name(model_name: str) -> str:
    """Normalize model names to improve matching against the context window map."""
    normalized = model_name.strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/")[-1]

    while True:
        updated = _DATE_SUFFIX_RE.sub("", normalized)
        for suffix in _SUFFIXES:
            if updated.endswith(suffix):
                updated = updated[: -len(suffix)]
        updated = _QUANT_SUFFIX_RE.sub("", updated)
        updated = _Q_SUFFIX_RE.sub("", updated)
        if updated == normalized:
            break
        normalized = updated
    return normalized


def lookup_context_window(model_name: str, context_map: dict[str, int]) -> int | None:
    """Resolve a model's context window using exact or longest-prefix match."""
    normalized_name = normalize_model_name(model_name)
    normalized_map = {normalize_model_name(key): int(value) for key, value in context_map.items()}

    if normalized_name in normalized_map:
        return normalized_map[normalized_name]

    best_key = ""
    for key in normalized_map:
        if normalized_name.startswith(key) and len(key) > len(best_key):
            best_key = key

    return normalized_map.get(best_key) if best_key else None


def save_context_window_map(context_map: dict[str, int], config_dir: Path | None = None) -> Path:
    """Persist the context window map to the live config file."""
    live_path = ensure_live_context_window_map(config_dir)
    sanitized: dict[str, int] = {}
    for key, value in context_map.items():
        if key is None:
            continue
        sanitized[str(key)] = int(value)
    live_path.write_text(yaml.safe_dump(sanitized, sort_keys=True))
    return live_path


def update_context_window_map(model_name: str, max_tokens: int, config_dir: Path | None = None) -> Path:
    """Update the live context window map with a resolved token limit."""
    context_map = load_context_window_map(config_dir)
    normalized_name = normalize_model_name(model_name)
    context_map[normalized_name] = int(max_tokens)
    return save_context_window_map(context_map, config_dir)
