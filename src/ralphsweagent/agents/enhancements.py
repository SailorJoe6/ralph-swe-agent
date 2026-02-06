"""Monkeypatch DefaultAgent with ralph-swe-agent enhancements.

This module adds context window tracking and live trajectory streaming to
DefaultAgent so all agent classes (default, interactive, reasoning_tool_call)
inherit the behavior regardless of which class is resolved from config.
"""

from __future__ import annotations

from ralphsweagent.models.context_window import (
    load_context_window_map,
    lookup_context_window,
    normalize_model_name,
    update_context_window_map,
)

_patched = False


def register_agent_enhancements() -> None:
    """Monkeypatch DefaultAgent with context window tracking and live trajectory streaming.

    Safe to call multiple times; only patches once.
    """
    global _patched
    if _patched:
        return
    _patched = True

    import json
    from pathlib import Path

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.utils.serialize import to_jsonable

    _original_init = DefaultAgent.__init__
    _original_run = DefaultAgent.run
    _original_query = DefaultAgent.query
    _original_get_template_vars = DefaultAgent.get_template_vars
    _original_add_messages = DefaultAgent.add_messages

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.context_window_max: int | None = None
        self.context_window_prompt_tokens: int | None = None
        self.context_left_percent: int | None = None
        self._live_trajectory_path: Path | None = None

    def _set_live_trajectory_path(self, path: Path | None) -> None:
        """Set a live JSONL trajectory path and clear any existing file."""
        self._live_trajectory_path = path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
        except Exception as exc:
            self.logger.warning("Failed to initialize live trajectory file %s: %s", path, exc)

    def _patched_add_messages(self, *messages: dict) -> list[dict]:
        result = _original_add_messages(self, *messages)
        if self._live_trajectory_path:
            try:
                self._live_trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                with self._live_trajectory_path.open("a", encoding="utf-8") as handle:
                    for message in messages:
                        handle.write(json.dumps(to_jsonable(message)) + "\n")
            except Exception as exc:
                self.logger.warning(
                    "Failed to write live trajectory to %s: %s", self._live_trajectory_path, exc
                )
        return result

    def _patched_get_template_vars(self, **kwargs):
        base = _original_get_template_vars(self, **kwargs)
        base.update({
            "context_window_max": self.context_window_max,
            "context_window_prompt_tokens": self.context_window_prompt_tokens,
            "context_left_percent": self.context_left_percent,
        })
        return base

    def _patched_run(self, task: str = "", **kwargs):
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._resolve_context_window_max()
        self.add_messages(
            self.model.format_message(
                role="system", content=self._render_template(self.config.system_template)
            ),
            self.model.format_message(
                role="user", content=self._render_template(self.config.instance_template)
            ),
        )
        from minisweagent.exceptions import InterruptAgentFlow

        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def _patched_query(self):
        from minisweagent.exceptions import LimitsExceeded

        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if self.context_window_max is None:
            self._resolve_context_window_max()
        self.n_calls += 1
        message = self.model.query(self.messages)
        self._update_context_window_stats(message)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def _resolve_context_window_max(self) -> None:
        if self.context_window_max is not None:
            return
        model_name = getattr(self.model, "config", None)
        model_name = getattr(model_name, "model_name", None)
        if not model_name or not isinstance(model_name, str):
            return
        model_name = model_name.strip()
        if not model_name:
            return
        context_map = load_context_window_map()
        resolved = lookup_context_window(model_name, context_map)
        normalized_name = normalize_model_name(model_name)
        normalized_keys = {normalize_model_name(key) for key in context_map}
        if resolved is None and getattr(self, "context_window_mode", "auto") == "interactive":
            resolved = self._prompt_for_context_window(model_name)
            if resolved is not None:
                update_context_window_map(model_name, resolved)
        if resolved is not None:
            self.context_window_max = int(resolved)
            if normalized_name not in normalized_keys:
                update_context_window_map(model_name, resolved)

    def _prompt_for_context_window(self, model_name: str) -> int | None:
        return None

    def _update_context_window_stats(self, message: dict) -> None:
        prompt_tokens = self._extract_prompt_tokens(message)
        if prompt_tokens is None:
            return
        self.context_window_prompt_tokens = prompt_tokens
        if not self.context_window_max:
            return
        left = int(100 * (1 - (prompt_tokens / self.context_window_max)))
        self.context_left_percent = max(0, min(100, left))
        message["context_left_percent"] = self.context_left_percent

    @staticmethod
    def _extract_prompt_tokens(message: dict) -> int | None:
        usage = None
        extra = message.get("extra") or {}
        response = extra.get("response")
        if response and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None and isinstance(message.get("usage"), dict):
            usage = message.get("usage")
        if usage is None and hasattr(message.get("usage"), "model_dump"):
            usage = message["usage"].model_dump()
        if usage is None and hasattr(message.get("usage"), "__dict__"):
            usage = message["usage"].__dict__
        if not isinstance(usage, dict):
            return None
        prompt_tokens = usage.get("prompt_tokens")
        return prompt_tokens if isinstance(prompt_tokens, int) else None

    DefaultAgent.__init__ = _patched_init
    DefaultAgent.run = _patched_run
    DefaultAgent.query = _patched_query
    DefaultAgent.get_template_vars = _patched_get_template_vars
    DefaultAgent.add_messages = _patched_add_messages
    DefaultAgent.set_live_trajectory_path = _set_live_trajectory_path
    DefaultAgent.context_window_mode = "auto"
    DefaultAgent._resolve_context_window_max = _resolve_context_window_max
    DefaultAgent._prompt_for_context_window = _prompt_for_context_window
    DefaultAgent._update_context_window_stats = _update_context_window_stats
    DefaultAgent._extract_prompt_tokens = _extract_prompt_tokens
