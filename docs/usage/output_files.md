# Output Files

ralph-swe-agent produces several output files during SWE-bench evaluation runs.

## Trajectory Files

### Final Trajectory (`.traj.json`)

After each instance completes, the agent saves a final trajectory JSON file containing
the full message history and metadata:

```
<output_dir>/<instance_id>/<instance_id>.traj.json
```

This file is written once when the agent finishes (or encounters an unrecoverable error).

### Live Trajectory JSONL (`.traj.jsonl`)

During an agent run, messages are streamed to a JSONL (JSON Lines) file in real time:

```
<output_dir>/<instance_id>/<instance_id>.traj.jsonl
```

Each line is a JSON object representing one message (system, user, assistant, or exit).
This file is useful for monitoring agent progress while a run is in progress:

```bash
tail -f output/swe-agent__test-repo-1/swe-agent__test-repo-1.traj.jsonl | jq -C '{role, content}'
```

The live JSONL file is **automatically deleted** after the final `.traj.json` is saved.
If the agent crashes before saving, the JSONL file remains as a partial record.

#### Lifecycle

1. **Created** — `process_instance` creates the instance directory and passes the JSONL
   path to `agent.set_live_trajectory_path()`, which clears any pre-existing file.
2. **Appended** — Every call to `agent.add_messages()` appends one JSON line per message
   using `minisweagent.utils.serialize.to_jsonable()` for safe serialization.
3. **Deleted** — After `agent.save()` writes the final trajectory, the JSONL file is
   removed in the `finally` block.

### swebench-single

When running a single instance via `mini-extra swebench-single`, the live JSONL path
is derived from the output path automatically. For example:

| Output path | Live JSONL path |
|---|---|
| `run.traj.json` | `run.traj.jsonl` |
| `run.json` | `run.jsonl` |
| `run` | `run.traj.jsonl` |

## Predictions File (`preds.json`)

A cumulative JSON file at `<output_dir>/preds.json` maps instance IDs to their
model predictions. Updated after each instance completes.
