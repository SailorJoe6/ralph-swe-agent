"""Tests for ralph-swe-agent's custom SWE-bench runner.

Covers the live trajectory streaming lifecycle managed by
``ralphsweagent.run.benchmarks.swebench.process_instance``.
"""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ralphsweagent._bootstrap import ensure_vendor_minisweagent_on_path

ensure_vendor_minisweagent_on_path()

from ralphsweagent.agents.enhancements import register_agent_enhancements

from minisweagent.environments.local import LocalEnvironment


# Ensure enhancements are registered so DefaultAgent has set_live_trajectory_path.
register_agent_enhancements()


class SlowExitModel:
    """Model that sleeps briefly before returning an exit message.

    This gives the test thread time to observe the live JSONL file
    while the agent is still running.
    """

    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.config = SimpleNamespace(model_name="slow_exit")

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        time.sleep(self.delay)
        return {
            "role": "exit",
            "content": "",
            "extra": {
                "exit_status": "Submitted",
                "submission": "",
                "cost": 0.0,
                "timestamp": time.time(),
                "actions": [],
            },
        }

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return []

    def get_template_vars(self, **kwargs) -> dict:
        return {"model_name": self.config.model_name}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": {"model_name": self.config.model_name},
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            },
        }


def test_live_trajectory_streaming_lifecycle(tmp_path):
    """Live JSONL file is created during agent run and cleaned up after.

    Ralph's ``process_instance`` creates a ``.traj.jsonl`` file, passes it
    to the agent via ``set_live_trajectory_path``, and deletes it in the
    ``finally`` block once the final ``.traj.json`` has been saved.
    """
    from ralphsweagent.run.benchmarks.swebench import process_instance

    instance_id = "swe-agent__test-repo-1"
    instance = {"instance_id": instance_id, "problem_statement": "Do the thing."}
    progress_manager = MagicMock()
    config: dict = {
        "agent": {
            "system_template": "You are a test agent.",
            "instance_template": "{{ task }}",
        },
        "model": {},
    }

    with (
        patch("ralphsweagent.run.benchmarks.swebench.get_model", return_value=SlowExitModel()),
        patch("ralphsweagent.run.benchmarks.swebench.base_swebench.get_sb_environment", return_value=LocalEnvironment()),
    ):
        thread = threading.Thread(
            target=process_instance,
            args=(instance, tmp_path, config, progress_manager),
        )
        thread.start()

        # Poll for the live JSONL file to appear.
        live_path = tmp_path / instance_id / f"{instance_id}.traj.jsonl"
        for _ in range(50):
            if live_path.exists():
                break
            time.sleep(0.01)

        assert live_path.exists(), "Live JSONL file was never created"
        # At least system + user messages should have been written.
        assert len(live_path.read_text().splitlines()) >= 2

        thread.join()

    # After the run finishes the live file is cleaned up.
    assert not live_path.exists(), "Live JSONL file should be removed after run"
    # Final trajectory JSON should exist.
    assert (tmp_path / instance_id / f"{instance_id}.traj.json").exists()


def test_live_trajectory_jsonl_content(tmp_path):
    """Each line in the live JSONL file is valid JSON with expected keys."""
    from ralphsweagent.run.benchmarks.swebench import process_instance

    instance_id = "swe-agent__test-repo-1"
    instance = {"instance_id": instance_id, "problem_statement": "Check JSONL."}
    progress_manager = MagicMock()
    config: dict = {
        "agent": {
            "system_template": "You are a test agent.",
            "instance_template": "{{ task }}",
        },
        "model": {},
    }

    # Capture live JSONL content before cleanup.
    captured_lines: list[str] = []

    original_unlink = type(tmp_path / "x").unlink

    def _capture_before_unlink(self, missing_ok=False):
        if self.suffix == ".jsonl" and self.exists():
            captured_lines.extend(self.read_text().splitlines())
        return original_unlink(self, missing_ok=missing_ok)

    with (
        patch("ralphsweagent.run.benchmarks.swebench.get_model", return_value=SlowExitModel(delay=0.0)),
        patch("ralphsweagent.run.benchmarks.swebench.base_swebench.get_sb_environment", return_value=LocalEnvironment()),
        patch.object(type(tmp_path / "x"), "unlink", _capture_before_unlink),
    ):
        process_instance(instance, tmp_path, config, progress_manager)

    assert len(captured_lines) >= 2, "Expected at least system + user JSONL lines"
    for line in captured_lines:
        obj = json.loads(line)
        assert "role" in obj
        assert "content" in obj
