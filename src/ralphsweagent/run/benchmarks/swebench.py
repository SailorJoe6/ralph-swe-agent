"""Custom SWE-bench runner wrapper with configurable agent class."""

from __future__ import annotations

import traceback
from pathlib import Path

from ralphsweagent._bootstrap import ensure_vendor_minisweagent_on_path

ensure_vendor_minisweagent_on_path()

from ralphsweagent.agents import resolve_agent_class
from ralphsweagent.agents.enhancements import register_agent_enhancements
from ralphsweagent.models import register_model_overrides

from minisweagent.models import get_model
from minisweagent.run.benchmarks import swebench as base_swebench
from minisweagent.utils.log import logger

app = base_swebench.app


def _wrap_with_progress(agent_class: type) -> type:
    if issubclass(agent_class, base_swebench.ProgressTrackingAgent):
        return agent_class

    class ProgressAgent(agent_class):
        def __init__(self, *args, progress_manager, instance_id: str = "", **kwargs):
            super().__init__(*args, **kwargs)
            self.progress_manager = progress_manager
            self.instance_id = instance_id

        def step(self) -> dict:
            self.progress_manager.update_instance_status(
                self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})"
            )
            return super().step()

    return ProgressAgent


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager,
) -> None:
    """Copy of upstream process_instance, kept here for monkeypatch customization."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    live_traj_path = instance_dir / f"{instance_id}.traj.jsonl"
    base_swebench.remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    live_traj_path.unlink(missing_ok=True)
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
        agent_config = dict(config.get("agent", {}))
        agent_class_spec = agent_config.pop("agent_class", None)
        base_agent_class = resolve_agent_class(agent_class_spec, default=base_swebench.ProgressTrackingAgent)
        agent_class = _wrap_with_progress(base_agent_class)
        agent = agent_class(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **agent_config,
        )
        agent.set_live_trajectory_path(live_traj_path)
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
            live_traj_path.unlink(missing_ok=True)
        base_swebench.update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


# Monkeypatch upstream runner so `mini-extra swebench` options/behavior stay compatible.
register_model_overrides()
register_agent_enhancements()
base_swebench.process_instance = process_instance
