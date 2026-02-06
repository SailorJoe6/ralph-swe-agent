import pytest

from minisweagent.exceptions import FormatError
from ralphsweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    BASH_TOOL_RESPONSE_API_WITH_REASONING,
    parse_toolcall_actions_response,
)


def _make_call(arguments: str, name: str = "bash") -> dict:
    return {"type": "function_call", "call_id": "call_1", "name": name, "arguments": arguments}


class TestParseToolcallActionsResponse:
    def test_empty_output_raises_format_error(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions_response([], format_error_template="{{ error }}")
        assert "No tool calls found" in exc_info.value.messages[0]["content"][0]["text"]

    def test_missing_reasoning_raises_format_error(self):
        output = [_make_call('{"command": "echo test"}')]
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions_response(output, format_error_template="{{ error }}", require_reasoning=True)
        assert "Missing or empty 'reasoning' argument" in exc_info.value.messages[0]["content"][0]["text"]

    def test_blank_reasoning_raises_format_error(self):
        output = [_make_call('{"command": "echo test", "reasoning": "   "}')]
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions_response(output, format_error_template="{{ error }}", require_reasoning=True)
        assert "Missing or empty 'reasoning' argument" in exc_info.value.messages[0]["content"][0]["text"]

    def test_valid_reasoning_allows_tool_call(self):
        output = [_make_call('{"command": "echo test", "reasoning": "inspect"}')]
        result = parse_toolcall_actions_response(output, format_error_template="{{ error }}", require_reasoning=True)
        assert result == [{"command": "echo test", "tool_call_id": "call_1"}]


class TestBashToolResponseApi:
    def test_bash_tool_response_api_structure(self):
        assert BASH_TOOL_RESPONSE_API["type"] == "function"
        assert BASH_TOOL_RESPONSE_API["name"] == "bash"
        assert "command" in BASH_TOOL_RESPONSE_API["parameters"]["properties"]
        assert "command" in BASH_TOOL_RESPONSE_API["parameters"]["required"]

    def test_bash_tool_response_api_with_reasoning_structure(self):
        assert BASH_TOOL_RESPONSE_API_WITH_REASONING["type"] == "function"
        assert BASH_TOOL_RESPONSE_API_WITH_REASONING["name"] == "bash"
        assert "command" in BASH_TOOL_RESPONSE_API_WITH_REASONING["parameters"]["properties"]
        assert "reasoning" in BASH_TOOL_RESPONSE_API_WITH_REASONING["parameters"]["properties"]
        required = BASH_TOOL_RESPONSE_API_WITH_REASONING["parameters"]["required"]
        assert "command" in required
        assert "reasoning" in required
