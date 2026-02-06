# SWE-bench Usage

`ralph-swe-agent` keeps `mini-extra swebench` compatible, but routes it through
`src/ralphsweagent/run/benchmarks/swebench.py` so ralph customizations are applied.

## Reasoning Tool-Call Agent

This custom Agent adds a required `reasoning` field to the `bash` tool call signature.  While the bash tool 
doesn't need this field, we've found 2 use cases for it.  
1) You can set `tool_choice:required` in the `model_kwags:` config, and force the agent to use tool calls on every turn.  The agent will never forget to add a tool call and will often issue 2 or 3 tool calls in the same turn.  However, most LLMs will NOT populate the content field when `tool_choice: required` is set.  In this case, the Agent's reasoning is not available in the context and the agent fails to converge on a solution.  By including a required `reasoning` field in the tool call, you can force the agent's thought processes to be present in every tool call when using `tool_choice: required`.  This seems to work well in most cases, but might be brittle
2) Even when we you do not set `tool_choice:required` we've found that having the extra required `reasoning` field tends to prompt the LLM to put more detailed reasoning into the context anyway.  This seems to lead to better results in solving the coding problem.  

To enable the ReasoningToolCallAgent set:

```yaml
agent:
  agent_class: reasoning_tool_call
```

To enforce tool calling set:
```yaml
model_kwargs:
  tool_choice: required
```

This agent enables `require_reasoning` on the model config.
It does **not** force `tool_choice: required`; set that explicitly in model config if desired.

## Dataset Compatibility

The runner accepts both `image_name` and `docker_image` fields from SWE-bench
instance data.  Standard SWE-bench datasets use `image_name`, while
SWE-bench-Live MultiLang instances provide `docker_image` instead.

When both fields are present, `image_name` takes priority.  If neither is set,
the image name is constructed from the `instance_id` using the upstream
convention (`docker.io/swebench/sweb.eval.x86_64.<id>:latest`).

## Ready-Made Config

You can start from:

- `src/ralphsweagent/config/benchmarks/swebench_toolcall_only.yaml`
