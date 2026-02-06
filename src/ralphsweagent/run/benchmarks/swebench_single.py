"""Run on a single SWE-Bench instance."""

from __future__ import annotations

from pathlib import Path

import typer
from datasets import load_dataset

from ralphsweagent._bootstrap import ensure_vendor_minisweagent_on_path

ensure_vendor_minisweagent_on_path()

from ralphsweagent.agents import resolve_agent_class
from ralphsweagent.agents.enhancements import register_agent_enhancements
from ralphsweagent.models import register_model_overrides

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, get_sb_environment
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import recursive_merge

DEFAULT_OUTPUT_FILE = global_config_dir / "last_swebench_single_run.traj.json"
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

app = typer.Typer(add_completion=False)

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.
"""


def _get_live_trajectory_path(output_path: Path) -> Path:
    filename = output_path.name
    if filename.endswith(".traj.json"):
        filename = filename[: -len(".traj.json")] + ".traj.jsonl"
    elif output_path.suffix:
        filename = output_path.with_suffix(".jsonl").name
    else:
        filename = f"{filename}.traj.jsonl"
    return output_path.with_name(filename)


# fmt: off
@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="SWE-Bench instance ID or index", rich_help_panel="Data selection"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", rich_help_panel="Advanced"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Basic"),
    output: Path = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    register_model_overrides()
    register_agent_enhancements()
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset from {dataset_path}, split {split}...")
    instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=split)}  # type: ignore
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({"agent": {"mode": "yolo"}})
    if environment_class is not None:
        configs.append({"environment": {"environment_class": environment_class}})
    if model_class is not None:
        configs.append({"model": {"model_class": model_class}})
    if model_name is not None:
        configs.append({"model": {"model_name": model_name}})
    if exit_immediately:
        configs.append({"agent": {"confirm_exit": False}})
    configs.append({"agent": {"output_path": output}})
    config = recursive_merge(*configs)

    env = get_sb_environment(config, instance)
    agent_config = dict(config.get("agent", {}))
    agent_class_spec = agent_config.pop("agent_class", None)
    agent_class = resolve_agent_class(agent_class_spec, default=InteractiveAgent)
    agent = agent_class(
        get_model(config=config.get("model", {})),
        env,
        **agent_config,
    )
    agent.set_live_trajectory_path(_get_live_trajectory_path(output))
    agent.run(instance["problem_statement"])


if __name__ == "__main__":
    app()
