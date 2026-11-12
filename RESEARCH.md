# AgentFlow: research notes

Research date: 2026-11-12. Items marked (measured) were run on this machine (12 logical cores, CPU-only PyTorch 2.11, transformers 5.7). Items marked (reported) come from web sources I could not verify myself.

## 1. Agent patterns

| Pattern | Idea | What I take from it |
|---|---|---|
| ReAct (Yao et al. 2022, https://arxiv.org/abs/2210.03629) | Interleave reasoning and tool calls, one step at a time | Simple, but every step re-plans from scratch. Small models drift and loop. Good as the *inner* executor of a single step, bad as the whole agent. |
| Plan-and-Solve (Wang et al. 2023, https://arxiv.org/abs/2305.04091), plan-and-execute | Write the whole plan first, then execute | A plan is explicit state the UI can draw and the verifier can check against. Matches the site's "planner breaks a request into steps". |
| ReWOO (Xu et al. 2023, https://arxiv.org/abs/2305.18323) | Plan with variable placeholders (`#E1`), execute without LLM calls between steps, then solve | Cheap on tokens, which matters on CPU. I borrow placeholder passing between steps (`$s1.result`). |
| Reflexion (Shinn et al. 2023, https://arxiv.org/abs/2303.11366) | After failure, write a verbal reflection and retry with it in context | The retry path: on a failed verification the retry prompt includes the verifier's reason. |
| LangGraph plan-and-execute / reflection tutorials (https://blog.langchain.com/reflection-agents, https://docs.langchain.com/oss/python/langgraph) | Planner, executor, replanner as graph nodes | Direct template for the graph. Their replanner is an LLM call; mine is preceded by a deterministic verifier. |
| Verifiers and critics (CRITIC, Gou et al. 2023, https://arxiv.org/abs/2305.11738) | Use tools to check outputs rather than self-assessing | Verifier prefers deterministic checks (schema, row counts, arithmetic re-computation, "file exists", "outbox has one message") over asking the same small model "is this right?". LLM self-verification of a 0.6B model is not trustworthy, so it is optional and off by default. |

Finding: no popular open-source agent framework has a first-class *verify-before-next-step* node with typed outcomes. Most put it inside the replanner prompt. Making it a separate node with outcomes `pass | retry | replan | fail` is both the differentiator and what the site's scene shows (packet goes through the verifier and back).

## 2. Current library APIs (checked here)

Installed: langchain 1.2.17, langchain-core 1.3.2, langgraph 1.1.10, langgraph-prebuilt 1.0.13, openai 2.33.0. `anthropic`, `langchain-openai`, `langchain-anthropic`, `rank_bm25` are NOT installed.

- LangGraph (docs via Context7, https://docs.langchain.com/oss/python/langgraph/graph-api): `StateGraph(State)`, `add_node`, `add_edge`, `add_conditional_edges`, `Command(update=..., goto=...)` for update-and-route in one node, `compile()`.
- Streaming: `graph.stream(inputs, stream_mode=["updates","custom"], version="v2")`, with `get_stream_writer()` inside nodes for custom events. I will use `custom` to emit trace events live while nodes run, and `updates` for state.
- LangChain 1.x moved agent building to `langchain.agents.create_agent`. I do not use it, because it hides the plan and verify steps I need to expose. I use LangChain for tool definitions (`StructuredTool` with pydantic arg schemas, giving JSON schemas for free) and message types, and LangGraph for the loop.
- Decision: do not add `langchain-openai` / `langchain-anthropic`. OpenAI and Anthropic adapters are ~60 lines each over `httpx`, testable against `httpx.MockTransport`. Fewer dependencies, and the stub test hits the real wire format (request body shape, tool-call parsing).

## 3. Tool calling with small local models (measured)

I ran a 10-prompt probe (one prompt per tool, two per tool for some), Qwen chat-template tool format (`<tool_call>{json}</tool_call>`), greedy decoding, bf16 on CPU, five tool schemas in the prompt (~420 prompt tokens):

| Model | Download (safetensors) | Licence | Valid tool-call JSON | Correct tool | Speed |
|---|---|---|---|---|---|
| Qwen3-0.6B (non-thinking) | 1.5 GB | Apache-2.0 | 10/10 | 9/10 | ~9 tok/s |
| Qwen2.5-1.5B-Instruct | 3.1 GB | Apache-2.0 | 9/10 | 9/10 | ~2.5 tok/s (bf16; first-run, likely improvable in fp32) |

Observations:
- Tool *choice* and JSON shape are fine at 0.6B. What fails is argument *content*: the 0.6B model wrote broken Python (a `for` clause stuck on a function def), guessed SQL table and column names (`sales_table`, `order_date`), and once routed an arithmetic question to `search`. The 1.5B model wrote correct code and SQL but still guessed the schema, and once answered in prose instead of calling a tool.
- Consequence for design: inject the real DB schema and doc file list into the prompt, make the verifier catch SQL errors / Python exceptions and feed them back (Reflexion-style retry), and measure how much the verifier+retry loop recovers. That is exactly the result worth reporting.
- Bigger models (Qwen3-1.7B 4.1 GB, Qwen3-4B-Instruct-2507 8 GB, SmolLM3-3B 6 GB) exceed the 3 GB download budget in bf16. Quantised GGUF would fit but needs `llama-cpp-python`, which means compiling on Windows. Not worth the risk. Qwen3-1.7B-FP8 (2.65 GB) needs FP8 kernels that are not CPU-friendly (not tested).
- Third-party leaderboards (reported, unverified): Qwen3 0.6B and 1.7B rank low on BFCL multi-turn, with a capability threshold between 1.7B and 4B for reliable multi-step ReAct (https://d-central.tech/local-llm-agent-capability/, https://benchmarklist.com/benchmarks/bfcl/). Matches my probe: single call fine, multi-step fragile. Re-check the official board (https://gorilla.cs.berkeley.edu/leaderboard.html) before quoting any number. I will quote only my own measurements.
- I deleted the Qwen2.5-1.5B download after probing to stay inside the budget. Qwen3-0.6B stays cached (shared HF cache).

Chosen: Qwen3-0.6B as the default real model; Qwen2.5-0.5B-Instruct (already cached) as an alternative; scripted model for deterministic tests; OpenAI/Anthropic adapters against stubs.

## 4. Sandboxing Python code

Sources agree (https://www.morphllm.com/python-sandbox and others, practitioner blogs, not authoritative): language-level sandboxes (RestrictedPython, stripped `__builtins__`, import filters) get bypassed via introspection; real isolation is at the OS or VM boundary; timeouts plus memory caps alone are denial-of-service protection, not a security boundary.

Platform reality for this build (Windows 11 dev, Linux for CI):
- Windows has no seccomp or rlimit. Job Objects (via `ctypes`, `JOB_OBJECT_LIMIT_PROCESS_MEMORY`, `ACTIVE_PROCESS_LIMIT`, `KILL_ON_JOB_CLOSE`) give memory caps, a child-process cap and reliable kill of the tree. They do not restrict filesystem or network access. Restricted tokens / AppContainer could, but are heavy to do correctly from Python.
- Linux: `resource.setrlimit` (`RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NPROC`, `RLIMIT_FSIZE`) in `preexec_fn`. Network denial would need namespaces/seccomp (not attempted).

Design (defence in depth, stated honestly as NOT a security boundary against a determined attacker):
1. Static AST gate: import allow-list (`math, statistics, json, re, itertools, collections, datetime, decimal, fractions, random, string, textwrap, functools, operator`), reject dunder attribute access, `exec/eval/compile/open/__import__/getattr` on builtins.
2. Child process: `python -I -S`, scrubbed environment, fresh empty temp cwd, stdin closed, output truncated.
3. In-child hardening: stripped builtins, `socket` and `subprocess` replaced with stubs that raise before user code runs.
4. OS limits: Job Object on Windows, rlimits on Linux. Wall-clock timeout with tree kill.
5. Document what it does not protect against: CPython introspection escapes, side-channel reads of files the user can read (Windows), real network denial on Windows. The README recommends running behind a container for anything untrusted. Tests include a red-team list (import os, `().__class__.__bases__`, infinite loop, memory bomb, fork bomb, socket connect, open of a secret file) with the expected outcome per platform.

## 5. Tracing format

OpenTelemetry GenAI semantic conventions (https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans): operations `invoke_agent`, `chat`, `execute_tool`; attributes `gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.request.model`, `gen_ai.usage.input_tokens/output_tokens`. Status is "Development" and the spec moved repos in mid-2026 (https://dev.to/azena-ai/opentelemetrys-genai-semantic-conventions-are-not-stable-yet-heres-what-actually-shipped-in-2026-3mke), so I pin to attribute names and say so.

Decision: one JSONL line per span *event* (start and end as separate lines, so the UI can stream "call started" while a tool is running), each with `trace_id`, `span_id`, `parent_span_id`, `ts`, `name`, `kind` and an `attributes` map using `gen_ai.*` names where one exists, plus an `agentflow.*` namespace for plan steps, verdicts, retries. No OTLP exporter dependency; an optional converter script emits OTLP-JSON shape. Honest claim: "OpenTelemetry GenAI-aligned", not "compliant".

## 6. Retrieval for the search tool

Own BM25 (Okapi, k1=1.5, b=0.75, ~40 lines, no dependency) over paragraph-level chunks, optionally fused with `all-MiniLM-L6-v2` embeddings (already cached, Apache-2.0) via reciprocal rank fusion. Offline either way. A small hand-labelled query set measures recall@3 for BM25 vs hybrid so the choice is evidence-based.

## 7. Data and licences

| Asset | Source | Licence |
|---|---|---|
| Document corpus (~24 short files: handbook, policies, product notes, meeting notes, runbooks) | Written/generated by me, committed (< 200 KB) | MIT (project) |
| Sample SQLite DB (customers, products, orders, order_items, regions; ~2k orders) | Generated by a seeded script, not committed (rebuilt on first run) | MIT |
| Qwen3-0.6B | https://huggingface.co/Qwen/Qwen3-0.6B | Apache-2.0 |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 |
| Fonts (Inter Tight, Instrument Serif, JetBrains Mono), GSAP | Google Fonts / CDN | OFL / GSAP standard licence (free) |

## 8. Prior art worth naming in the README

LangGraph plan-and-execute tutorial, ReWOO and Reflexion reference code, OpenAI Agents SDK tracing (trace/span model), Langfuse/Phoenix (OTel-based trace UIs). AgentFlow is a small, inspectable, offline-first version of the loop with the verifier made explicit, not a framework.
