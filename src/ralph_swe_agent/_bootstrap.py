"""Bootstrap helpers for loading vendored mini-swe-agent modules."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_vendor_minisweagent_on_path() -> Path:
    """Add vendored mini-swe-agent `src/` to sys.path if needed."""
    repo_root = Path(__file__).resolve().parents[2]
    vendor_src = repo_root / "vendor" / "mini-swe-agent" / "src"
    if not vendor_src.exists():
        raise RuntimeError(f"Vendored mini-swe-agent not found at {vendor_src}")
    vendor_src_str = str(vendor_src)
    if vendor_src_str not in sys.path:
        sys.path.insert(0, vendor_src_str)
    return vendor_src
