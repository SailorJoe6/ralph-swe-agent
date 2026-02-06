from pathlib import Path

from ralphsweagent.models.context_window import (
    ensure_live_context_window_map,
    get_seed_context_window_path,
    load_context_window_map,
    lookup_context_window,
    normalize_model_name,
)


def test_normalize_model_name_strips_provider_and_suffixes():
    assert normalize_model_name("openai/gpt-4o-2024-08-06") == "gpt-4o"
    assert normalize_model_name("anthropic/claude-3-5-sonnet-20240620") == "claude-3-5-sonnet"
    assert normalize_model_name("gpt-4o-mini-preview") == "gpt-4o-mini"
    assert normalize_model_name("gpt-4o-mini-latest") == "gpt-4o-mini"
    assert normalize_model_name("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8") == "qwen3-coder-30b-a3b-instruct"
    assert normalize_model_name("llama-3.1-70b-instruct-awq") == "llama-3.1-70b-instruct"
    assert normalize_model_name("mistral-7b-instruct-v0.2-q4_k_m") == "mistral-7b-instruct-v0.2"


def test_lookup_context_window_exact_and_prefix_matches():
    context_map = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "qwen2.5-32b": 131072,
    }
    assert lookup_context_window("openai/gpt-4o-2024-08-06", context_map) == 128000
    assert lookup_context_window("gpt-4o-mini-2024-08-06", context_map) == 128000
    assert lookup_context_window("qwen2.5-32b-instruct", context_map) == 131072
    assert lookup_context_window("unknown-model", context_map) is None


def test_load_context_window_map_bootstraps_from_seed(tmp_path: Path):
    seed_path = get_seed_context_window_path()
    assert seed_path.exists()

    live_path = ensure_live_context_window_map(tmp_path)
    assert live_path.exists()

    loaded = load_context_window_map(tmp_path)
    assert loaded
    assert all(isinstance(value, int) for value in loaded.values())
