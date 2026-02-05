# ralph-swe-agent

Thin extension layer over mini-swe-agent.

## Design

- `vendor/mini-swe-agent` is a git submodule (upstream base).
- `ralph-swe-agent` adds a custom SWE-bench runner hook:
  - Supports `run.agent_class` in config.
  - Default behavior remains `ProgressTrackingAgent(DefaultAgent)`.

## Install

```bash
pip install -e .
```

## Compatibility

This package provides `mini`, `mini-swe-agent`, and `mini-extra` console scripts.
`mini-extra swebench` routes to the custom runner; other subcommands delegate to vendored mini-swe-agent modules.
