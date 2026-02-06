# ralph-swe-agent

Thin extension layer over mini-swe-agent.

## Design

- `vendor/mini-swe-agent` is a git submodule (upstream base).
- `ralph-swe-agent` mirrors mini-swe-agent's module layout and keeps customizations in local packages:
  - `src/ralphsweagent/agents/...`
  - `src/ralphsweagent/models/...`
  - `src/ralphsweagent/run/...`
- `src/ralphsweagent/run/...` provides entrypoints and benchmark runners:
  - `ralphsweagent/run/mini.py`
  - `ralphsweagent/run/utilities/mini_extra.py`
  - `ralphsweagent/run/benchmarks/swebench.py`
- Custom agent/model behavior from the old `tool-calling-only` fork commit is implemented in `ralphsweagent`:
  - `ReasoningToolCallAgent` enables required `reasoning` tool-call args.
  - Model overrides add `require_reasoning` + `tool_choice` support and tool schema enforcement.
  - Shortcut model names (`litellm`, `openrouter`, etc.) are patched to the ralph model implementations.
- Custom SWE-bench runner keeps upstream behavior while applying ralph agent/model resolution.

## Install

```bash
pip install -e .
```

## Docs

- `docs/README.md`

## Compatibility

This package provides `mini`, `mini-swe-agent`, and `mini-extra` console scripts.
`mini`, `mini-extra swebench`, and `mini-extra swebench-single` route to ralph entrypoints.
Utility subcommands (`mini-extra config`, `mini-extra inspect`) delegate to vendored mini-swe-agent modules.
