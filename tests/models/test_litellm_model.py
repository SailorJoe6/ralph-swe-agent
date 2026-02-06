from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from ralphsweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from ralphsweagent.models.utils.actions_toolcall import BASH_TOOL, BASH_TOOL_WITH_REASONING


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


def _make_stream_chunk(*, content=None, tool_calls=None, usage=None, finish_reason=None, model="gpt-4"):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model,
        id="resp_1",
        created=123,
    )


class TestLitellmModel:
    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_reasoning_tool_schema_and_choice(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test", "reasoning": "inspect"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", require_reasoning=True, tool_choice="required")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL_WITH_REASONING]
        assert mock_completion.call_args.kwargs["tool_choice"] == "required"

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_streaming_reconstructs_content_and_usage(self, mock_cost, mock_completion):
        tool_call_delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo test"}'},
        }
        stream = iter(
            [
                _make_stream_chunk(content="Hello ", tool_calls=[tool_call_delta]),
                _make_stream_chunk(
                    content="world",
                    tool_calls=None,
                    usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    finish_reason="stop",
                ),
            ]
        )
        mock_completion.return_value = stream
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", use_streaming=True)
        result = model.query([{"role": "user", "content": "test"}])

        assert mock_completion.call_args.kwargs["stream"] is True
        assert mock_completion.call_args.kwargs["stream_options"]["include_usage"] is True
        assert result["content"] == "Hello world"
        assert result["extra"]["response"]["usage"]["prompt_tokens"] == 3
        assert result["extra"]["actions"] == [{"command": "echo test", "tool_call_id": "call_1"}]

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_streaming_falls_back_without_usage(self, mock_cost, mock_completion):
        tool_call_delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo test"}'},
        }
        stream = iter([_make_stream_chunk(content="Hello ", tool_calls=[tool_call_delta])])
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.side_effect = [stream, _mock_litellm_response([tool_call])]
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", use_streaming=True)
        result = model.query([{"role": "user", "content": "test"}])

        assert mock_completion.call_count == 2
        assert mock_completion.call_args_list[0].kwargs["stream"] is True
        assert mock_completion.call_args_list[1].kwargs.get("stream") is None
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_streaming_falls_back_on_zero_usage(self, mock_cost, mock_completion):
        tool_call_delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo test"}'},
        }
        stream = iter(
            [
                _make_stream_chunk(
                    content="Hello",
                    tool_calls=[tool_call_delta],
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    finish_reason="stop",
                )
            ]
        )
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.side_effect = [stream, _mock_litellm_response([tool_call])]
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", use_streaming=True)
        result = model.query([{"role": "user", "content": "test"}])

        assert mock_completion.call_count == 2
        assert mock_completion.call_args_list[0].kwargs["stream"] is True
        assert mock_completion.call_args_list[1].kwargs.get("stream") is None
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_stream_guard_truncates_repeated_closing_tags(self, mock_cost, mock_completion, caplog):
        tool_call_delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo test"}'},
        }
        stream = iter(
            [
                _make_stream_chunk(
                    content="</final></final>",
                    tool_calls=[tool_call_delta],
                    usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    finish_reason="stop",
                ),
            ]
        )
        mock_completion.return_value = stream
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            use_streaming=True,
            stream_guard_enabled=True,
            stream_guard_window=200,
            stream_guard_tag_threshold=2,
        )
        with caplog.at_level("WARNING"):
            result = model.query([{"role": "user", "content": "test"}])

        assert "Stream guard triggered" in caplog.text
        assert result["content"] == "</final>"

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_stream_guard_allows_normal_content(self, mock_cost, mock_completion):
        tool_call_delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo test"}'},
        }
        stream = iter(
            [
                _make_stream_chunk(content="Hello ", tool_calls=[tool_call_delta], finish_reason=None),
                _make_stream_chunk(
                    content="world</final>",
                    usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    finish_reason="stop",
                ),
            ]
        )
        mock_completion.return_value = stream
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            use_streaming=True,
            stream_guard_enabled=True,
            stream_guard_window=50,
            stream_guard_tag_threshold=3,
        )
        result = model.query([{"role": "user", "content": "test"}])

        assert result["content"] == "Hello world</final>"


class TestRetryMissingToolCalls:
    """Tests for the graceful tool-call recovery feature."""

    @staticmethod
    def _valid_tool_call():
        tc = MagicMock()
        tc.function.name = "bash"
        tc.function.arguments = '{"command": "echo hello"}'
        tc.id = "call_retry"
        return tc

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_retry_missing_tool_calls_succeeds(self, mock_cost, mock_completion):
        """First call returns no tool calls, retry returns valid tool call."""
        no_tools_resp = _mock_litellm_response(None)
        no_tools_resp.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "Let me think about this...",
        }
        valid_resp = _mock_litellm_response([self._valid_tool_call()])
        mock_completion.side_effect = [no_tools_resp, valid_resp]
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", retry_missing_tool_calls=True)
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        # Should succeed without raising FormatError
        assert result["extra"]["actions"] == [{"command": "echo hello", "tool_call_id": "call_retry"}]
        # Cost should include both calls
        assert result["extra"]["cost"] == 0.002
        # litellm.completion called twice (original + retry)
        assert mock_completion.call_count == 2
        # Nudge messages should remain in messages list
        assert len(messages) == 3  # original + assistant + nudge
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "You must respond using the bash tool."
        # Retry call should have tool_choice="required"
        assert mock_completion.call_args_list[1].kwargs["tool_choice"] == "required"

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_retry_missing_tool_calls_fails(self, mock_cost, mock_completion):
        """First call returns no tool calls, retry also returns no tool calls."""
        no_tools_resp1 = _mock_litellm_response(None)
        no_tools_resp1.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "Thinking...",
        }
        no_tools_resp2 = _mock_litellm_response(None)
        no_tools_resp2.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "Still thinking...",
        }
        mock_completion.side_effect = [no_tools_resp1, no_tools_resp2]
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", retry_missing_tool_calls=True)
        messages = [{"role": "user", "content": "test"}]
        with pytest.raises(FormatError):
            model.query(messages)

        # litellm.completion called twice
        assert mock_completion.call_count == 2
        # Messages should be restored (nudge messages removed)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "test"}

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_retry_skipped_when_tool_choice_required(self, mock_cost, mock_completion):
        """tool_choice='required' in config — no retry, FormatError raised immediately."""
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            retry_missing_tool_calls=True,
            tool_choice="required",
        )
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

        # Only one call (no retry)
        assert mock_completion.call_count == 1

    @patch("ralphsweagent.models.litellm_model.litellm.completion")
    @patch("ralphsweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_retry_skipped_when_flag_disabled(self, mock_cost, mock_completion):
        """retry_missing_tool_calls=False (default) — no retry, FormatError raised immediately."""
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")  # default: retry_missing_tool_calls=False
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

        # Only one call (no retry)
        assert mock_completion.call_count == 1
