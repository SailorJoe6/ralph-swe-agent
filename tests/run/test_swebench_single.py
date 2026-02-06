"""Tests for ralph-swe-agent's swebench_single runner.

Covers the ``_get_live_trajectory_path`` helper that converts a final
trajectory path into the corresponding live JSONL path.
"""

from pathlib import Path

import pytest

from ralphsweagent._bootstrap import ensure_vendor_minisweagent_on_path

ensure_vendor_minisweagent_on_path()

from ralphsweagent.run.benchmarks.swebench_single import _get_live_trajectory_path


@pytest.mark.parametrize(
    "input_name, expected_name",
    [
        # Standard .traj.json → .traj.jsonl
        ("output.traj.json", "output.traj.jsonl"),
        # Other extension → .jsonl
        ("output.json", "output.jsonl"),
        # No extension → append .traj.jsonl
        ("output", "output.traj.jsonl"),
        # Nested path preserved
        ("deep/nested/run.traj.json", "deep/nested/run.traj.jsonl"),
    ],
)
def test_get_live_trajectory_path(input_name, expected_name, tmp_path):
    """_get_live_trajectory_path converts trajectory paths to JSONL equivalents."""
    input_path = tmp_path / input_name
    expected = tmp_path / expected_name
    assert _get_live_trajectory_path(input_path) == expected
