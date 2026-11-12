"""In-memory run registry. One worker thread executes runs, so the CPU model is never shared."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import AgentConfig, Settings
from ..graph import AgentRunner
from ..tools.registry import FaultInjector, ToolBox
from ..tracing.events import TraceEvent
from ..tracing.writer import Tracer
from .models import RunInfo, RunRequest


class Run:
    def __init__(self, run_id: str, req: RunRequest, trace_path: Path) -> None:
        self.id, self.req, self.trace_path = run_id, req, trace_path
        self.status = "queued"
        self.answer: str | None = None
        self.events: list[dict] = []
        self.created = time.time()
        self.done = threading.Event()

    def info(self) -> RunInfo:
        return RunInfo(
            run_id=self.id, status=self.status, task=self.req.task, llm=self.req.llm,  # type: ignore[arg-type]
            verifier=self.req.verifier, faults=self.req.faults, answer=self.answer,
            events=len(self.events), created=self.created,
        )


class RunRegistry:
    def __init__(self, settings: Settings, llm_for: Callable[[RunRequest], object]) -> None:
        self.settings = settings
        self.llm_for = llm_for
        self.runs: dict[str, Run] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentflow-run")

    def submit(self, req: RunRequest) -> Run:
        run_id = uuid.uuid4().hex[:10]
        run = Run(run_id, req, self.settings.runs_dir / "server" / f"{run_id}.jsonl")
        self.runs[run_id] = run
        while len(self.runs) > 50:  # keep memory bounded
            self.runs.pop(next(iter(self.runs)))
        self.pool.submit(self._execute, run)
        return run

    def _execute(self, run: Run) -> None:
        run.status = "running"
        try:
            llm = self.llm_for(run.req)
            self.settings.ensure_db()
            faults = FaultInjector(run.req.faults, run.req.seed) if run.req.faults else None
            tracer = Tracer(run.trace_path, [lambda ev: run.events.append(ev.model_dump())])
            runner = AgentRunner(llm, ToolBox(self.settings, faults), tracer, AgentConfig(verifier=run.req.verifier))  # type: ignore[arg-type]
            res = runner.run(run.req.task)
            run.answer, run.status = res.answer, res.status
        except Exception as exc:
            run.answer, run.status = f"{type(exc).__name__}: {exc}", "error"
            run.events.append(TraceEvent(
                trace_id="error", span_id="error", event="end", name="invoke_agent", ts=time.time(), t=0.0,
                attributes={"agentflow.error": run.answer, "agentflow.status": "error"}, status="error",
            ).model_dump())
        finally:
            run.done.set()
