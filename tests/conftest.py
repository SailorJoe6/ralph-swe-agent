import threading

import pytest

# Bootstrap: ensure vendored mini-swe-agent is importable before any test runs.
from ralphsweagent import ensure_vendor_minisweagent_on_path

ensure_vendor_minisweagent_on_path()

from minisweagent.models import GLOBAL_MODEL_STATS  # noqa: E402


# Global lock for tests that modify global state - this works across threads
_global_stats_lock = threading.Lock()


@pytest.fixture
def reset_global_stats():
    """Reset global model stats and ensure exclusive access for tests that need it."""
    with _global_stats_lock:
        GLOBAL_MODEL_STATS._cost = 0.0
        GLOBAL_MODEL_STATS._n_calls = 0
        yield
        GLOBAL_MODEL_STATS._cost = 0.0
        GLOBAL_MODEL_STATS._n_calls = 0
