"""Custom SWE-bench runner wrapper with configurable agent class."""

from __future__ import annotations

import importlib
import inspect
import traceback
from pathlib import Path

from ._bootstrap import ensure_vendor_minisweagent_on_path

ensure_vendor_minisweagent_on_path()

from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model
from minisweagent.run.benchmarks import swebench as base_swebench
from minisweagent.utils.log import logger

app = base_swebench.app


def _resolve_agent_class(spec: str | None) -> type[DefaultAgent]:
    if not spec:
        return base_swebench.ProgressTrackingAgent
    module_name, class_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    if not issubclass(agent_cls, DefaultAgent):
        raise TypeError(f"run.agent_class must inherit DefaultAgent, got: {spec}")
    return agent_cls


def _supports_kwarg(func, name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    if name in signature.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager,
) -> None:
    """Drop-in replacement for upstream process_instance with configurable agent class."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    base_swebench.remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = base_swebench.get_sb_environment(config, instance)
        agent_cls = _resolve_agent_class(config.get("run", {}).get("agent_class"))
        agent_kwargs = dict(config.get("agent", {}))
        if _supports_kwarg(agent_cls.__init__, "progress_manager"):
            agent_kwargs["progress_manager"] = progress_manager
        if _supports_kwarg(agent_cls.__init__, "instance_id"):
            agent_kwargs["instance_id"] = instance_id
        agent = agent_cls(model, env, **agent_kwargs)
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:  # pragma: no cover - parity with upstream behavior
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        base_swebench.update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


# Monkeypatch upstream runner so `mini-extra swebench` options/behavior stay compatible.
base_swebench.process_instance = process_instance
