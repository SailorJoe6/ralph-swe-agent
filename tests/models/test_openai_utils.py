from types import SimpleNamespace

from ralphsweagent.models.utils.openai_utils import coerce_responses_text


def test_coerce_responses_text_handles_output_text_items():
    output = [
        {"type": "output_text", "text": "Hello"},
        {"type": "output_text", "text": " world"},
    ]
    assert coerce_responses_text(output) == "Hello world"


def test_coerce_responses_text_handles_message_content():
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi"}],
        }
    ]
    assert coerce_responses_text(output) == "Hi"


def test_coerce_responses_text_handles_response_objects():
    content_obj = SimpleNamespace(type="output_text", text="Object")
    message_obj = SimpleNamespace(type="message", role="assistant", content=[content_obj])
    assert coerce_responses_text([message_obj]) == "Object"


def test_coerce_responses_text_skips_non_text_items():
    output = [
        {"type": "function_call", "name": "bash", "arguments": "{}"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
    ]
    assert coerce_responses_text(output) == "ok"
