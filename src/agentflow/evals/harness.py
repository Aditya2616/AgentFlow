"""Run one task end to end in an isolated environment and judge it."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import AgentConfig, Settings
from ..graph import AgentRunner
from ..tools.registry import FaultInjector, ToolBox
from ..tracing.writer import Tracer
from .tasks import CheckCtx, Task


@dataclass
class TaskResult:
    task_id: str
    band: str
    success: bool
    status: str
    answer: str
    seconds: float
    acts: int = 0
    replans: int = 0
    retries: int = 0
    steps: int = 0
    tool_errors: int = 0
    verifier_flags: int = 0
    would_flag: int = 0
    plan_fallback: bool = False
    model_calls: int = 0
    tool_calls: int = 0
    faults_transient: int = 0
    faults_silent: int = 0
    emails_sent: int = 0
    outbox: list = field(default_factory=list)  # [{to, subject, body}] as written
    db_rows: list = field(default_factory=list)  # orders rows before, after
    seed: int = 0
    trace: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def judge(task: Task, status: str, ctx: CheckCtx) -> bool:
    """The one place a run is scored; `agentflow rescore` calls it on stored results."""
    return bool(status != "error" and (status == "completed" or task.band == "adversarial") and task.check(ctx))


def _count_orders(db: Path) -> int:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        con.close()


def run_task(
    task: Task,
    llm: BaseChatModel,
    *,
    settings: Settings,
    cfg: AgentConfig | None = None,
    fault_p: float = 0.0,
    silent_p: float = 0.0,
    seed: int = 0,
    trace_path: Path | None = None,
    listeners: list | None = None,
    keep_outbox: bool = False,
) -> TaskResult:
    cfg = cfg or AgentConfig()
    db = settings.ensure_db()
    tmp = Path(tempfile.mkdtemp(prefix="agentflow-run-"))
    local = Settings(
        corpus_dir=settings.corpus_dir, db_path=db, outbox_dir=tmp / "outbox", runs_dir=settings.runs_dir,
        search_mode=settings.search_mode, threads=settings.threads,
    )
    # one stream per (seed, task): without the task in the seed every task would see the same draws
    fault_seed = seed * 1000 + int(task.id.lstrip("t") or 0)
    faults = FaultInjector(fault_p, fault_seed, silent_p) if (fault_p or silent_p) else None
    tracer = Tracer(trace_path, listeners)
    runner = AgentRunner(llm, ToolBox(local, faults), tracer, cfg)
    before = _count_orders(db)
    started = time.perf_counter()
    status, answer = "error", ""
    try:
        res = runner.run(task.prompt, {"agentflow.seed": seed, "agentflow.fault_p": fault_p})
        status, answer = res.status, res.answer
    except Exception as exc:  # a crash is a failed task, not a crashed evaluation
        answer = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started
    outbox = []
    index = local.outbox_dir / "index.jsonl"
    if index.exists():
        outbox = [json.loads(ln) for ln in index.read_text(encoding="utf-8").splitlines() if ln.strip()]
    after = _count_orders(db)
    ctx = CheckCtx(answer, status, outbox, before, after, task.prompt)
    ok = judge(task, status, ctx)
    ev = tracer.events
    ends = [e for e in ev if e.event == "end"]
    verifies = [e for e in ends if e.name == "agentflow.verify"]
    tools = [e for e in ends if e.name == "execute_tool"]
    plans = [e for e in ev if e.name == "agentflow.plan" and e.event == "start"]
    result = TaskResult(
        task_id=task.id, band=task.band, success=ok, status=status, answer=answer[:2000], seconds=round(seconds, 2),
        acts=len(tools), replans=max(0, len(plans) - 1),
        retries=sum(1 for e in verifies if e.attributes.get("agentflow.verdict") == "retry"),
        steps=len({e.attributes.get("agentflow.step_id") for e in verifies}),
        tool_errors=sum(1 for e in tools if e.status == "error"),
        verifier_flags=sum(1 for e in verifies if e.attributes.get("agentflow.verdict") != "pass"),
        would_flag=sum(1 for e in verifies if e.attributes.get("agentflow.would_flag")),
        plan_fallback=any(e.attributes.get("agentflow.plan_fallback") for e in plans),
        model_calls=sum(1 for e in ends if e.name == "chat"), tool_calls=len(tools),
        faults_transient=faults.counts["transient"] if faults else 0,
        faults_silent=faults.counts["silent"] if faults else 0,
        emails_sent=len(outbox),
        outbox=[{"to": m.get("to", ""), "subject": m.get("subject", ""), "body": str(m.get("body", ""))[:400]}
                for m in outbox],
        db_rows=[before, after],
        seed=seed, trace=str(trace_path or ""),
    )
    if not keep_outbox:
        shutil.rmtree(tmp, ignore_errors=True)
    return result
