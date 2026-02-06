from unittest.mock import Mock, patch

from ralphsweagent.models.litellm_response_model import LitellmResponseModel
from ralphsweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    BASH_TOOL_RESPONSE_API_WITH_REASONING,
)


@patch("ralphsweagent.models.litellm_response_model.litellm.responses")
@patch("ralphsweagent.models.litellm_response_model.litellm.cost_calculator.completion_cost", return_value=0.01)
def test_response_api_tracks_previous_response_id_and_content(mock_cost, mock_responses):
    output_1 = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
        {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": '{"command": "echo test"}'},
    ]
    response_1 = Mock()
    response_1.output = output_1
    response_1.model_dump.return_value = {"id": "resp_1", "output": output_1}

    output_2 = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "World"}]},
        {"type": "function_call", "call_id": "call_2", "name": "bash", "arguments": '{"command": "echo test"}'},
    ]
    response_2 = Mock()
    response_2.output = output_2
    response_2.model_dump.return_value = {"id": "resp_2", "output": output_2}

    mock_responses.side_effect = [response_1, response_2]

    model = LitellmResponseModel(model_name="gpt-4o")
    first = model.query([{"role": "user", "content": "hi"}])
    second = model.query([{"role": "user", "content": "hi again"}])

    assert first["content"] == "Hello"
    assert second["content"] == "World"
    assert mock_responses.call_args_list[0].kwargs["tools"] == [BASH_TOOL_RESPONSE_API]
    assert mock_responses.call_args_list[1].kwargs["previous_response_id"] == "resp_1"


@patch("ralphsweagent.models.litellm_response_model.litellm.responses")
@patch("ralphsweagent.models.litellm_response_model.litellm.cost_calculator.completion_cost", return_value=0.01)
def test_response_api_requires_reasoning_and_tool_choice(mock_cost, mock_responses):
    output = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "bash",
            "arguments": '{"command": "echo test", "reasoning": "inspect"}',
        },
    ]
    response = Mock()
    response.output = output
    response.model_dump.return_value = {"id": "resp_1", "output": output}

    mock_responses.return_value = response

    model = LitellmResponseModel(model_name="gpt-4o", require_reasoning=True, tool_choice="required")
    model.query([{"role": "user", "content": "hi"}])

    assert mock_responses.call_args.kwargs["tools"] == [BASH_TOOL_RESPONSE_API_WITH_REASONING]
    assert mock_responses.call_args.kwargs["tool_choice"] == "required"
