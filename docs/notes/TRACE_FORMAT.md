# Trace format

Every run writes `runs/<name>.jsonl`, one JSON object per line. A span is two lines, a `start` and an
`end`, joined by `span_id`. Events are validated against `agentflow.tracing.events.TraceEvent`.

| field | meaning |
|---|---|
| `trace_id` | one per run |
| `span_id`, `parent_span_id` | span identity; every span is a child of the `invoke_agent` root |
| `event` | `start` or `end` |
| `name` | `invoke_agent`, `chat`, `execute_tool`, `agentflow.plan`, `agentflow.verify`, `agentflow.finish` |
| `ts`, `t` | unix time, and seconds since the run started |
| `status` | `ok`, `error` or `unset` (set on `end`) |
| `attributes` | see below |

Attribute names follow the OpenTelemetry GenAI semantic conventions where one exists
(`gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.request.model`, `gen_ai.tool.name`,
`gen_ai.tool.call.id`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`). Those conventions
are still marked as under development upstream, so this is "aligned with", not "compliant with".
Everything specific to this project lives under `agentflow.*`:

* `agentflow.node`: `planner`, `tool` or `verifier`, the graph node the span belongs to (the UI uses it)
* `agentflow.phase` on `chat`: `plan`, `act` or `finish`
* `agentflow.plan`, `agentflow.replan`, `agentflow.plan_fallback` on `agentflow.plan`
* `agentflow.step_id`, `agentflow.attempt`, `agentflow.arguments`, `agentflow.result`, `agentflow.error_kind` on tools
* `agentflow.verdict` (`pass`, `retry`, `replan`, `fail`), `agentflow.reason`, `agentflow.checks`,
  `agentflow.would_flag` on `agentflow.verify`
* `agentflow.arm` (`off`, `retry` or `full`) on the root and on `agentflow.verify`
* `agentflow.fault` on `execute_tool` when the evaluation injected a fault (`transient` or `silent:<tool>`)
* `agentflow.dropped_steps` on `agentflow.plan`: email steps removed by the send guard
