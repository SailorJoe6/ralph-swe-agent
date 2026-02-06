# Global Configuration

ralph-swe-agent inherits global configuration from mini-swe-agent (API keys,
default model, cost limits, etc.) and adds its own enhancements described below.

## Context Window Map

ralph-swe-agent tracks the context window size for each model so that agents can
report how much of the context window has been consumed. This helps templates
warn the user when context is running low.

### How It Works

1. On first use, a seed file (`model_context_windows.yaml`) is copied to the
   mini-swe-agent global config directory (typically
   `~/.config/mini-swe-agent/`).
2. When an agent starts, it looks up the model name in this map.
3. After each model query, the agent calculates `context_left_percent` — the
   percentage of the context window still available — and attaches it to the
   response message.
4. If a model is resolved via prefix matching but is not yet in the map, it is
   added automatically for faster future lookups.

### Seeded Models

The seed file ships with mappings for common models:

| Model Pattern | Context Window |
|---------------|---------------|
| `gpt-4o` / `gpt-4o-mini` / `gpt-4.1` / `gpt-4.1-mini` / `gpt-4-turbo` | 128,000 |
| `gpt-4-32k` | 32,768 |
| `gpt-4` | 8,192 |
| `claude-3-5-sonnet` / `claude-3-5-haiku` / `claude-3-opus` | 200,000 |
| `gemini-1.5-pro` | 2,000,000 |
| `gemini-1.5-flash` | 1,000,000 |
| `llama-3.1-70b-instruct` / `llama-3.1-8b-instruct` | 131,072 |
| `qwen2.5-72b-instruct` | 131,072 |
| `qwen3-coder-30b-a3b-instruct` | 262,144 |

### Model Name Normalization

Model names are normalized before lookup to improve matching:

- Provider prefixes are stripped (`openai/gpt-4o` → `gpt-4o`)
- Date suffixes are removed (`gpt-4o-2024-08-06` → `gpt-4o`)
- Version suffixes are removed (`-preview`, `-beta`, `-latest`)
- Quantization suffixes are removed (`-fp8`, `-int4`, `-awq`, `-q4_k_m`, etc.)
- Everything is lowercased

If an exact match is not found, the longest prefix match is used.

### Customizing the Map

Edit the live copy at `~/.config/mini-swe-agent/model_context_windows.yaml`:

```yaml
# Add or update entries
my-custom-model: 65536
```

### Template Variables

The following variables are available in agent templates:

| Variable | Type | Description |
|----------|------|-------------|
| `context_window_max` | `int \| None` | Maximum context window tokens for the model |
| `context_window_prompt_tokens` | `int \| None` | Prompt tokens used in the last query |
| `context_left_percent` | `int \| None` | Percentage of context window remaining (0–100) |
