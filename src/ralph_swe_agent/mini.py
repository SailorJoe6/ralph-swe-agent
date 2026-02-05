"""Compatibility entrypoint for `mini`/`mini-swe-agent`."""

from __future__ import annotations

from ._bootstrap import ensure_vendor_minisweagent_on_path


def main() -> None:
    ensure_vendor_minisweagent_on_path()
    from minisweagent.run.mini import app

    app()
