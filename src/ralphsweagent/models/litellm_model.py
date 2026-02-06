import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from ralphsweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    BASH_TOOL_WITH_REASONING,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")

CLOSING_TAG_RE = re.compile(r"</[^>]+>")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class _StreamingMessage:
    def __init__(self, *, role: str, content: str | None, tool_calls: list):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


class _StreamingChoice:
    def __init__(self, *, index: int, message: _StreamingMessage, finish_reason: str | None):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason

    def model_dump(self) -> dict:
        return {
            "index": self.index,
            "message": self.message.model_dump(),
            "finish_reason": self.finish_reason,
        }


class _StreamingResponse(dict):
    def __init__(self, *, choices: list[_StreamingChoice], usage: dict | None, model: str | None, **kwargs):
        data: dict[str, Any] = {"choices": choices, **kwargs}
        if usage is not None:
            data["usage"] = usage
        if model is not None:
            data["model"] = model
        super().__init__(data)
        self.choices = choices
        self.usage = usage
        self.model = model

    def model_dump(self) -> dict:
        data = dict(self)
        data["choices"] = [choice.model_dump() for choice in self.choices]
        return data


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    use_streaming: bool = _env_flag("MSWEA_USE_STREAMING", False)
    """Stream responses from LiteLLM to avoid long-response timeouts."""
    stream_include_usage: bool = _env_flag("MSWEA_STREAM_INCLUDE_USAGE", True)
    """Include usage data in stream chunks when supported."""
    stream_guard_enabled: bool = _env_flag("MSWEA_STREAM_GUARD_ENABLED", False)
    """Enable stream guard to stop pathological closing-tag repetition."""
    stream_guard_window: int = _env_int("MSWEA_STREAM_GUARD_WINDOW", 8192)
    """Window size in characters for stream guard repetition detection."""
    stream_guard_tag_threshold: int = _env_int("MSWEA_STREAM_GUARD_TAG_THRESHOLD", 50)
    """Closing-tag repetition threshold in the rolling window before truncation."""
    tool_choice: Any | None = None
    """Tool choice configuration passed to the API (e.g., "required")."""
    require_reasoning: bool = False
    """Require non-empty reasoning in bash tool calls."""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            if self.config.use_streaming:
                return self._query_streaming(messages, **kwargs)
            return self._query_non_streaming(messages, **kwargs)
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _query_non_streaming(self, messages: list[dict[str, str]], **kwargs):
        request_kwargs = self.config.model_kwargs | kwargs
        tool_choice = self.config.tool_choice
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL_WITH_REASONING] if self.config.require_reasoning else [BASH_TOOL],
            **request_kwargs,
        )

    def _query_streaming(self, messages: list[dict[str, str]], **kwargs):
        stream_kwargs = self.config.model_kwargs | kwargs
        tool_choice = self.config.tool_choice
        if tool_choice is not None:
            stream_kwargs["tool_choice"] = tool_choice
        if self.config.stream_include_usage:
            stream_options = dict(stream_kwargs.get("stream_options") or {})
            stream_options.setdefault("include_usage", True)
            stream_kwargs["stream_options"] = stream_options
        stream = litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL_WITH_REASONING] if self.config.require_reasoning else [BASH_TOOL],
            stream=True,
            **stream_kwargs,
        )
        response = self._reconstruct_stream_response(stream)
        if self.config.stream_include_usage and not self._is_usage_valid(response.usage):
            logger.warning("Streaming response missing usage; retrying non-streaming completion for cost tracking.")
            return self._query_non_streaming(messages, **kwargs)
        return response

    @staticmethod
    def _is_usage_valid(usage: dict | None) -> bool:
        if not isinstance(usage, dict):
            return False
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if not isinstance(value, int) or value <= 0:
                return False
        return True

    @staticmethod
    def _normalize_usage(usage: Any) -> dict | None:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "__dict__"):
            return usage.__dict__
        return None

    @staticmethod
    def _normalize_tool_call_delta(tool_call: Any) -> dict | None:
        if tool_call is None:
            return None
        if isinstance(tool_call, dict):
            return tool_call
        if hasattr(tool_call, "model_dump"):
            return tool_call.model_dump()
        if hasattr(tool_call, "__dict__"):
            return tool_call.__dict__
        return None

    def _accumulate_tool_calls(self, tool_calls_by_index: dict[int, dict], delta_tool_calls: list[Any]) -> None:
        for raw_tool_call in delta_tool_calls:
            tool_call = self._normalize_tool_call_delta(raw_tool_call)
            if not tool_call:
                continue
            index = tool_call.get("index", 0)
            entry = tool_calls_by_index.setdefault(
                index, {"id": None, "type": None, "function": {"name": None, "arguments": ""}}
            )
            if tool_call.get("id"):
                entry["id"] = tool_call["id"]
            if tool_call.get("type"):
                entry["type"] = tool_call["type"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                entry["function"]["name"] = function["name"]
            if function.get("arguments"):
                entry["function"]["arguments"] += function["arguments"]

    def _build_tool_calls(self, tool_calls_by_index: dict[int, dict]) -> list:
        tool_calls = []
        for index in sorted(tool_calls_by_index):
            data = tool_calls_by_index[index]
            function = SimpleNamespace(
                name=data["function"].get("name"),
                arguments=data["function"].get("arguments", ""),
            )
            tool_call = SimpleNamespace(id=data.get("id"), function=function, type=data.get("type"))
            tool_calls.append(tool_call)
        return tool_calls

    def _reconstruct_stream_response(self, stream) -> _StreamingResponse:
        content_text = ""
        tool_calls_by_index: dict[int, dict] = {}
        usage: dict | None = None
        finish_reason: str | None = None
        role = "assistant"
        model = None
        response_id = None
        created = None
        for chunk in stream:
            if chunk is None:
                continue
            model = getattr(chunk, "model", model)
            response_id = getattr(chunk, "id", response_id)
            created = getattr(chunk, "created", created)
            chunk_usage = self._normalize_usage(getattr(chunk, "usage", None))
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", finish_reason)
            delta = getattr(choice, "delta", None) or getattr(choice, "message", None)
            if not delta:
                continue
            delta_role = getattr(delta, "role", None)
            if delta_role:
                role = delta_role
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                self._accumulate_tool_calls(tool_calls_by_index, delta_tool_calls)
            delta_content = getattr(delta, "content", None)
            if delta_content:
                content_text += delta_content
                if self._should_trigger_stream_guard(content_text):
                    content_text = self._truncate_stream_content(content_text)
                    logger.warning("Stream guard triggered; truncating streamed content.")
                    break

        tool_calls = self._build_tool_calls(tool_calls_by_index) if tool_calls_by_index else []
        content = content_text if content_text else None
        message = _StreamingMessage(role=role, content=content, tool_calls=tool_calls)
        choice = _StreamingChoice(index=0, message=message, finish_reason=finish_reason)
        return _StreamingResponse(
            choices=[choice],
            usage=usage,
            model=model,
            id=response_id,
            created=created,
        )

    def _should_trigger_stream_guard(self, content: str) -> bool:
        if not self.config.stream_guard_enabled:
            return False
        window_size = self.config.stream_guard_window
        threshold = self.config.stream_guard_tag_threshold
        if window_size <= 0 or threshold <= 0:
            return False
        window = content[-window_size:] if len(content) > window_size else content
        matches = list(CLOSING_TAG_RE.finditer(window))
        return len(matches) >= threshold

    def _truncate_stream_content(self, content: str) -> str:
        window_size = self.config.stream_guard_window
        threshold = self.config.stream_guard_tag_threshold
        if window_size <= 0 or threshold <= 0:
            return content
        window = content[-window_size:] if len(content) > window_size else content
        matches = list(CLOSING_TAG_RE.finditer(window))
        if len(matches) < threshold:
            return content
        cutoff = len(content) - len(window) + matches[threshold - 1].start()
        return content[:cutoff]

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            # Preserve raw assistant response for debugging (appears in live + final trajectories)
            debug_message = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
                "extra": {
                    "parse_error": True,
                    "response": response.model_dump(),
                    **cost_output,
                    "timestamp": time.time(),
                },
            }
            raise FormatError(debug_message, *e.messages) from e
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            require_reasoning=self.config.require_reasoning,
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
