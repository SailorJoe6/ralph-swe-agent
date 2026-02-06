"""Agent that requires reasoning in bash tool call arguments."""

from __future__ import annotations

from minisweagent.agents.default import DefaultAgent


class ReasoningToolCallAgent(DefaultAgent):
    """Enable required reasoning fields for bash tool calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_reasoning_tool_calls()

    def _configure_reasoning_tool_calls(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None:
            return
        if hasattr(config, "require_reasoning"):
            config.require_reasoning = True
        if hasattr(config, "retry_missing_tool_calls"):
            config.retry_missing_tool_calls = True
