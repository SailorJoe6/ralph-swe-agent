# YAML Configuration

`ralph-swe-agent` supports custom agent classes via `agent.agent_class`.

Example:

```yaml
agent:
  agent_class: ralphsweagent.agents.reasoning_tool_call.ReasoningToolCallAgent
```

Notes:

- `agent.agent_class` can be a full import path.
- Model shortcuts (`litellm`, `openrouter`, `portkey`, `requesty`, and response variants)
  are mapped to ralph model overrides during ralph run entrypoints.
- For tool-call reasoning runs, see `usage/swebench.md`.
