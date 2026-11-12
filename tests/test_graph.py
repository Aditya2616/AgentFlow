"""The agent loop, driven by the scripted model so every path is deterministic and offline."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agentflow.config import AgentConfig
from agentflow.evals.harness import run_task
from agentflow.evals.tasks import with_mistake
from agentflow.graph import AgentRunner
from agentflow.llm.scripted import ScriptedChat
from agentflow.tools.registry import FaultInjector, ToolBox
from agentflow.tracing.writer import Tracer


def _run(task, settings, *, cfg=None, faults=None):
    llm = ScriptedChat(scripts={task.prompt: task.script})
    tracer = Tracer()
    res = AgentRunner(llm, ToolBox(settings, faults), tracer, cfg or AgentConfig()).run(task.prompt)
    return res, tracer


def _verdicts(tracer):
    return [e.attributes["agentflow.verdict"] for e in tracer.events
            if e.event == "end" and e.name == "agentflow.verify"]


def test_clean_run_passes_every_step(tasks, settings):
    res, tr = _run(tasks["t09"], settings)
    assert res.status == "completed"
    assert _verdicts(tr) == ["pass", "pass"]
    assert [e.attributes["gen_ai.tool.name"] for e in tr.events if e.event == "start" and e.name == "execute_tool"] == ["sql", "python"]


def test_bad_arguments_trigger_retry_then_pass(tasks, settings):
    task = with_mistake(tasks["t03"], "bad_args")
    res, tr = _run(task, settings)
    assert res.status == "completed"
    assert _verdicts(tr) == ["retry", "pass"]
    retry = [e for e in tr.events if e.event == "end" and e.name == "agentflow.verify"][0]
    assert "no such table" in retry.attributes["agentflow.reason"]


def test_wrong_plan_triggers_replan(tasks, settings):
    task = with_mistake(tasks["t05"], "bad_plan")
    res, tr = _run(task, settings)
    assert res.status == "completed" and res.state["replans"] == 1
    assert _verdicts(tr)[:3] == ["retry", "retry", "replan"]
    plans = [e for e in tr.events if e.name == "agentflow.plan" and e.event == "start"]
    assert len(plans) == 2 and plans[1].attributes["agentflow.replan"] == 1


def test_unrecoverable_task_fails_honestly_within_budget(tasks, settings):
    res, tr = _run(tasks["t22"], settings)
    assert res.status == "failed"
    assert "could not complete" in res.answer
    assert _verdicts(tr)[-1] == "fail"
    assert res.state["replans"] == 2
    assert res.state["acts"] <= AgentConfig().max_acts


def test_email_is_rejected_before_it_is_written(tasks, settings):
    base = tasks["t11"]
    # an email whose body states no fact from the search result must never reach the outbox
    args = {**base.script.args, "email it to lead@example.com": [
        {"to": "lead@example.com", "subject": "P1 SLA", "body": "Hello there, hope you are well."},
        {"to": "lead@example.com", "subject": "P1 SLA", "body": "Priority 1 first response is 4 hours."},
    ]}
    task = replace(base, script=replace(base.script, args=args))
    res, tr = _run(task, settings)
    assert res.status == "completed"
    outbox = (settings.outbox_dir / "index.jsonl").read_text().splitlines()
    assert len(outbox) == 1 and "4 hours" in outbox[0]
    rejected = [e for e in tr.events if e.event == "end" and e.name == "execute_tool" and e.status == "error"]
    assert rejected and rejected[0].attributes["agentflow.error_kind"] == "rejected"


def test_transient_failure_retries_with_the_same_arguments(tasks, settings):
    task = tasks["t03"]
    res, tr = _run(task, settings, faults=FaultInjector(0.5, seed=1))
    assert res.status == "completed"
    reused = [e for e in tr.events if e.event == "start" and e.name == "execute_tool"
              and e.attributes.get("agentflow.reused_args")]
    chats = [e for e in tr.events if e.event == "start" and e.name == "chat"]
    # retries after a transient error reuse the arguments and make no extra model call
    assert len(chats) == 3 and reused


def test_verifier_off_lets_errors_through(tasks, settings):
    task = with_mistake(tasks["t03"], "bad_args")
    res, tr = _run(task, settings, cfg=AgentConfig(verifier=False))
    assert _verdicts(tr) == ["pass"]
    last = [e for e in tr.events if e.event == "end" and e.name == "agentflow.verify"][0]
    assert last.attributes["agentflow.would_flag"] is True


def test_trace_structure(tasks, settings):
    res, tr = _run(tasks["t10"], settings)
    ev = tr.events
    root = ev[0]
    assert root.name == "invoke_agent" and root.parent_span_id is None
    starts = {e.span_id: e for e in ev if e.event == "start"}
    ends = {e.span_id for e in ev if e.event == "end"}
    assert set(starts) == ends  # every span that starts also ends
    assert all(e.parent_span_id == root.span_id for e in ev if e.event == "start" and e is not root)
    assert {e.name for e in ev} >= {"invoke_agent", "chat", "execute_tool", "agentflow.verify", "agentflow.plan"}
    assert all(e.t >= 0 for e in ev) and [e.t for e in ev] == sorted(e.t for e in ev)
    tool = next(e for e in ev if e.event == "start" and e.name == "execute_tool")
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool" and tool.attributes["gen_ai.tool.call.id"]


def test_jsonl_file_roundtrip(tasks, settings, tmp_path):
    from agentflow.tracing.writer import read_trace

    path = tmp_path / "t.jsonl"
    r = run_task(tasks["t01"], ScriptedChat(scripts={tasks["t01"].prompt: tasks["t01"].script}),
                 settings=settings, trace_path=path)
    assert r.success
    events = read_trace(path)  # validates every line against the pydantic model
    assert events[0].name == "invoke_agent" and events[-1].name == "invoke_agent"
    assert events[-1].attributes["agentflow.status"] == "completed"


@pytest.mark.parametrize("verifier", [True, False])
def test_full_scripted_suite_with_oracle_scripts(tasks, settings, verifier):
    ok = 0
    for t in tasks.values():
        llm = ScriptedChat(scripts={t.prompt: t.script})
        ok += run_task(t, llm, settings=settings, cfg=AgentConfig(verifier=verifier)).success
    assert ok == len(tasks)  # oracle scripts make no mistakes, so both modes pass everything
