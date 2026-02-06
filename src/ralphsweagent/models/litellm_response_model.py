import logging
import time
from collections.abc import Callable

import litellm

from minisweagent.models import GLOBAL_MODEL_STATS
from ralphsweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from ralphsweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    BASH_TOOL_RESPONSE_API_WITH_REASONING,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from ralphsweagent.models.utils.openai_utils import coerce_responses_text
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_response_model")


class LitellmResponseModelConfig(LitellmModelConfig):
    pass


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self._previous_response_id: str | None = None

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Flatten response objects into their output items for stateless API calls."""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            request_kwargs = self.config.model_kwargs | kwargs
            if self._previous_response_id:
                request_kwargs.setdefault("previous_response_id", self._previous_response_id)
            if self.config.tool_choice is not None:
                request_kwargs["tool_choice"] = self.config.tool_choice
            return litellm.responses(
                model=self.config.model_name,
                input=messages,
                tools=(
                    [BASH_TOOL_RESPONSE_API_WITH_REASONING]
                    if self.config.require_reasoning
                    else [BASH_TOOL_RESPONSE_API]
                ),
                **request_kwargs,
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        response_id = message.get("id")
        if response_id:
            self._previous_response_id = response_id
        output_items = message.get("output", [])
        content = coerce_responses_text(output_items)
        if content:
            message["content"] = content
        message["extra"] = {
            "actions": self._parse_actions(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        return parse_toolcall_actions_response(
            getattr(response, "output", []),
            format_error_template=self.config.format_error_template,
            require_reasoning=self.config.require_reasoning,
        )

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
