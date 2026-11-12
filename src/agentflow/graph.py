"""The agent loop as a LangGraph StateGraph: plan -> act -> verify -> (act | plan | finish).

Nodes are methods of AgentRunner so they share the model, the toolbox and the tracer.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .config import AgentConfig
from .llm.base import extract_json
from .planner import make_plan
from .prompts import act_system, act_user, compact, finish_system, finish_user
from .tools.registry import ToolBox
from .tracing.writer import Tracer
from .verifier import preflight, verify


class AgentState(TypedDict, total=False):
    task: str
    plan: list[dict]
    cursor: int
    next_id: int
    obs: dict[str, dict]
    attempt: int
    replans: int
    acts: int
    retry_reason: str | None
    reuse_args: dict | None
    status: str
    answer: str
    failure: str | None


@dataclass
class RunResult:
    status: str
    answer: str
    state: dict
    trace_id: str


def model_id(llm: BaseChatModel) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", None) or llm._llm_type)


class AgentRunner:
    def __init__(
        self, llm: BaseChatModel, toolbox: ToolBox, tracer: Tracer, cfg: AgentConfig | None = None
    ) -> None:
        self.llm, self.tb, self.tracer = llm, toolbox, tracer
        self.cfg = cfg or AgentConfig()
        self.root: str | None = None
        self.files = sorted(p.name for p in toolbox.settings.corpus_dir.glob("*.md"))
        self.model = model_id(llm)

    # ---------------------------------------------------------------- helpers
    def chat(self, messages: list, phase: str, step_id: str | None = None, tools: list | None = None,
             **kw: Any) -> AIMessage:
        attrs = {
            "gen_ai.operation.name": "chat", "gen_ai.request.model": self.model,
            "agentflow.node": "planner", "agentflow.phase": phase,
        }
        if step_id:
            attrs["agentflow.step_id"] = step_id
        with self.tracer.span("chat", self.root, **attrs) as sp:
            model = self.llm.bind_tools(tools) if tools else self.llm
            ai = model.invoke(messages, **kw)
            usage = ai.usage_metadata or {}
            sp.set(**{
                "gen_ai.usage.input_tokens": usage.get("input_tokens", 0),
                "gen_ai.usage.output_tokens": usage.get("output_tokens", 0),
                "agentflow.output": str(ai.content)[:600],
                "agentflow.tool_calls": [{"name": c["name"], "args": c["args"]} for c in ai.tool_calls],
            })
        return ai

    def _prev(self, state: AgentState) -> dict[str, dict]:
        done = {s["id"] for s in state["plan"][: state["cursor"]]}
        return {k: v for k, v in state.get("obs", {}).items() if k in done}

    def _prev_compact(self, state: AgentState) -> dict[str, Any]:
        return {k: compact(v["tool"], v) for k, v in self._prev(state).items()}

    # ---------------------------------------------------------------- nodes
    def plan_node(self, state: AgentState) -> dict:
        replans = state.get("replans", 0)
        cursor = state.get("cursor", 0)
        plan = state.get("plan", [])
        done = [
            {"goal": s["goal"], "summary": str(compact(s["tool"], state["obs"][s["id"]]))[:120]}
            for s in plan[:cursor]
        ]
        failed = plan[cursor] if replans and cursor < len(plan) else None
        steps, fallback = make_plan(
            lambda m, ph, **kw: self.chat(m, ph, **kw), state["task"], self.files,
            self.cfg.max_plan_steps, replan=replans, done=done, failed=failed,
            reason=state.get("retry_reason") or "",
        )
        nid = state.get("next_id", 1)
        new = []
        for st in steps:
            new.append({"id": f"s{nid}", "goal": st.goal, "tool": st.tool})
            nid += 1
        full = plan[:cursor] + new
        self.tracer.end(
            self.tracer.start("agentflow.plan", self.root, **{
                "agentflow.node": "planner", "agentflow.plan": full, "agentflow.replan": replans,
                "agentflow.plan_fallback": fallback, "agentflow.cursor": cursor,
            })
        )
        return {"plan": full, "next_id": nid, "attempt": 0, "retry_reason": None, "reuse_args": None}

    def act_node(self, state: AgentState) -> dict:
        step = state["plan"][state["cursor"]]
        attempt = state.get("attempt", 0) + 1
        tool_name = step["tool"]
        reuse = state.get("reuse_args")
        if reuse is not None:
            args, why = reuse, None
        else:
            args, why = self._fill_args(state, step)
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        with self.tracer.span("execute_tool", self.root, **{
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.id": call_id, "agentflow.node": "tool", "agentflow.step_id": step["id"],
            "agentflow.attempt": attempt, "agentflow.arguments": args or {},
            "agentflow.reused_args": reuse is not None,
        }) as sp:
            blocked = []
            if args is not None and self.cfg.verifier and not self.cfg.retry_only:
                blocked = [c for c in preflight(step, args, self._prev(state), state["task"], self.cfg.send_guard)
                           if not c.ok]
            if args is None:
                out = {"ok": False, "value": None, "error": why, "error_kind": "no_call", "latency_ms": 0}
            elif blocked:
                msg = "rejected before sending: " + "; ".join(c.detail for c in blocked)
                out = {"ok": False, "value": None, "error": msg, "error_kind": "rejected", "latency_ms": 0}
            else:
                o = self.tb.execute(tool_name, args)
                out = {"ok": o.ok, "value": o.value, "error": o.error, "error_kind": o.error_kind,
                       "latency_ms": o.latency_ms, "fault": o.fault}
            obs = {"step_id": step["id"], "tool": tool_name, "args": args or {}, "attempt": attempt, **out}
            if out["ok"]:
                sp.set(**{"agentflow.result": compact(tool_name, obs), "agentflow.latency_ms": out["latency_ms"],
                          **({"agentflow.fault": out["fault"]} if out.get("fault") else {})})
            else:
                sp.fail(**{"agentflow.error": out["error"], "agentflow.error_kind": out["error_kind"],
                           "agentflow.latency_ms": out["latency_ms"],
                           **({"agentflow.fault": out["fault"]} if out.get("fault") else {})})
        return {
            "obs": {**state.get("obs", {}), step["id"]: obs},
            "attempt": attempt, "acts": state.get("acts", 0) + 1, "reuse_args": None,
        }

    def _fill_args(self, state: AgentState, step: dict) -> tuple[dict | None, str]:
        tool = self.tb.tools[step["tool"]]
        msgs = [
            SystemMessage(content=act_system(step["id"], step["tool"], self.files)),
            HumanMessage(content=act_user(
                state["task"], step["goal"], self._prev_compact(state), state.get("retry_reason"),
                state.get("attempt", 0) + 1,
            )),
        ]
        ai = self.chat(msgs, "act", step["id"], tools=[tool], max_new_tokens=220)
        if ai.tool_calls:
            return dict(ai.tool_calls[0]["args"]), ""
        obj = extract_json(str(ai.content))  # some small models print the arguments as plain JSON
        if isinstance(obj, dict):
            return dict(obj.get("arguments", obj)), ""
        return None, "the model did not call the tool"

    def verify_node(self, state: AgentState) -> Command[Literal["act", "plan", "finish"]]:
        step = state["plan"][state["cursor"]]
        obs = state["obs"][step["id"]]
        attempt, replans = state["attempt"], state.get("replans", 0)
        with self.tracer.span("agentflow.verify", self.root, **{
            "agentflow.node": "verifier", "agentflow.step_id": step["id"], "agentflow.attempt": attempt,
            "agentflow.verifier_enabled": self.cfg.verifier, "agentflow.arm": self.cfg.arm,
        }) as sp:
            verdict, would_flag = verify(step, obs, self._prev(state), attempt, replans, self.cfg)
            if state.get("acts", 0) >= self.cfg.max_acts and verdict.outcome in ("retry", "replan"):
                verdict.outcome, verdict.reason = "fail", "step budget used up"
            sp.set(**{
                "agentflow.verdict": verdict.outcome, "agentflow.reason": verdict.reason,
                "agentflow.checks": [c.model_dump() for c in verdict.checks],
                "agentflow.would_flag": would_flag,
            })
        cursor = state["cursor"]
        if verdict.outcome == "pass":
            cursor += 1
            if cursor >= len(state["plan"]):
                return Command(update={"cursor": cursor, "status": "completed"}, goto="finish")
            return Command(update={"cursor": cursor, "attempt": 0, "retry_reason": None}, goto="act")
        if verdict.outcome == "retry":
            reuse = obs["args"] if verdict.reuse_args else None
            return Command(update={"retry_reason": verdict.reason, "reuse_args": reuse}, goto="act")
        if verdict.outcome == "replan":
            return Command(
                update={"replans": replans + 1, "retry_reason": verdict.reason, "attempt": 0}, goto="plan"
            )
        return Command(update={"status": "failed", "failure": f"{step['goal']}: {verdict.reason}"}, goto="finish")

    def finish_node(self, state: AgentState) -> dict:
        status = state.get("status", "completed")
        if status == "failed":
            done = self._prev(state)
            partial = "; ".join(f"{k}: {compact(v['tool'], v)}" for k, v in done.items())
            answer = f"I could not complete this task. {state.get('failure', '')}."
            if partial:
                answer += f" Completed so far: {partial}"
            with self.tracer.span("agentflow.finish", self.root, **{"agentflow.node": "planner"}) as sp:
                sp.set(**{"agentflow.output": answer, "agentflow.status": status})
            return {"answer": answer, "status": status}
        results = {k: compact(v["tool"], v) for k, v in state.get("obs", {}).items()
                   if any(s["id"] == k for s in state["plan"])}
        msgs = [SystemMessage(content=finish_system()), HumanMessage(content=finish_user(state["task"], results))]
        ai = self.chat(msgs, "finish", max_new_tokens=120)
        answer = str(ai.content).strip() or json.dumps(results)
        return {"answer": answer, "status": status}

    # ---------------------------------------------------------------- graph
    def build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self.plan_node)
        g.add_node("act", self.act_node)
        g.add_node("verify", self.verify_node)
        g.add_node("finish", self.finish_node)
        g.add_edge(START, "plan")
        g.add_edge("plan", "act")
        g.add_edge("act", "verify")
        g.add_edge("finish", END)
        return g.compile()

    def run(self, task: str, extra_attrs: dict | None = None) -> RunResult:
        graph = self.build()
        self.root = self.tracer.start("invoke_agent", None, **{
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "agentflow",
            "gen_ai.request.model": self.model, "agentflow.task": task,
            "agentflow.verifier_enabled": self.cfg.verifier, "agentflow.arm": self.cfg.arm, **(extra_attrs or {}),
        })
        try:
            final = graph.invoke(
                {"task": task, "plan": [], "cursor": 0, "next_id": 1, "obs": {}, "attempt": 0,
                 "replans": 0, "acts": 0}, {"recursion_limit": 80},
            )
        except Exception as exc:
            self.tracer.end(self.root, "error", **{"agentflow.error": f"{type(exc).__name__}: {exc}",
                                                    "agentflow.status": "error"})
            raise
        self.tracer.end(self.root, "ok" if final["status"] == "completed" else "error", **{
            "agentflow.status": final["status"], "agentflow.answer": final["answer"],
            "agentflow.acts": final.get("acts", 0), "agentflow.replans": final.get("replans", 0),
        })
        return RunResult(final["status"], final["answer"], dict(final), self.tracer.trace_id)
