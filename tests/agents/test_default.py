"""Tests for context window tracking on DefaultAgent via agent enhancements."""

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment

from ralphsweagent.agents.enhancements import register_agent_enhancements

# Apply enhancements before tests run.
register_agent_enhancements()

# Minimal agent config that satisfies AgentConfig required fields.
_AGENT_CONFIG = {
    "system_template": "You are a test assistant.",
    "instance_template": "Task: {{task}}",
    "step_limit": 5,
    "cost_limit": 5.0,
}


class _UsageModel:
    """Mock model that returns responses with usage data."""

    def __init__(self):
        self.config = type("Config", (), {"model_name": "gpt-4o"})()

    def query(self, messages, **kwargs):
        return {
            "role": "assistant",
            "content": "ok",
            "extra": {
                "actions": [],
                "cost": 0.0,
                "timestamp": 0.0,
                "response": {"usage": {"prompt_tokens": 6400, "completion_tokens": 128, "total_tokens": 6528}},
            },
        }

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return []

    def get_template_vars(self, **kwargs):
        return {"model_name": "gpt-4o"}

    def serialize(self):
        return {}


def test_context_window_tracking_sets_context_left_percent():
    agent = DefaultAgent(model=_UsageModel(), env=LocalEnvironment(), **_AGENT_CONFIG)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "user"})
    message = agent.query()
    assert agent.context_window_max == 128000
    assert agent.context_window_prompt_tokens == 6400
    assert agent.context_left_percent == 95
    assert message["context_left_percent"] == 95


def test_context_window_attributes_initialized():
    """Verify the monkeypatch adds context window attributes to __init__."""
    agent = DefaultAgent(model=_UsageModel(), env=LocalEnvironment(), **_AGENT_CONFIG)
    assert agent.context_window_max is None
    assert agent.context_window_prompt_tokens is None
    assert agent.context_left_percent is None


def test_context_window_in_template_vars():
    """Verify context window values appear in template variables."""
    agent = DefaultAgent(model=_UsageModel(), env=LocalEnvironment(), **_AGENT_CONFIG)
    vars_ = agent.get_template_vars()
    assert "context_window_max" in vars_
    assert "context_window_prompt_tokens" in vars_
    assert "context_left_percent" in vars_
