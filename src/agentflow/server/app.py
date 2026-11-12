"""FastAPI service: health, tools, tasks, runs, a Server-Sent Events feed, and the web UI."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Settings
from ..evals.tasks import Task, build_tasks, with_mistake
from ..llm import make_llm
from ..tools.registry import TOOL_HELP, ToolBox
from .models import Health, RunInfo, RunRequest, TaskInfo, ToolInfo
from .runs import RunRegistry

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
LARGE_MODEL = os.environ.get("AGENTFLOW_LARGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def create_app(settings: Settings | None = None, llm_override=None) -> FastAPI:
    settings = settings or Settings()
    allow_remote = os.environ.get("AGENTFLOW_ALLOW_REMOTE_TASKS", "0") == "1"
    app = FastAPI(title="AgentFlow", version=__version__)
    tasks: list[Task] = build_tasks(settings.ensure_db())
    by_prompt = {t.prompt: t for t in tasks}

    def llm_for(req: RunRequest):
        if llm_override is not None:
            return llm_override(req)
        if req.llm == "scripted":
            task = by_prompt.get(req.task)
            if task is None:
                raise ValueError("the scripted model only knows the suite tasks; pick one from the list")
            task = with_mistake(task, req.mistake)
            return make_llm("scripted", scripts={task.prompt: task.script})
        if req.llm == "local-large":
            return make_llm("local", settings=settings, model=LARGE_MODEL)
        return make_llm(req.llm, settings=settings)

    registry = RunRegistry(settings, llm_for)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version=__version__, default_llm=os.environ.get("AGENTFLOW_LLM", "local"),
                      local_model=settings.local_model, large_model=LARGE_MODEL,
                      remote_tasks_allowed=allow_remote)

    @app.get("/api/tools", response_model=list[ToolInfo])
    def tools() -> list[ToolInfo]:
        tb = ToolBox(settings)
        return [ToolInfo(name=n, description=TOOL_HELP[n], schema=t.args_schema.model_json_schema())  # type: ignore[union-attr]
                for n, t in tb.tools.items()]

    @app.get("/api/tasks", response_model=list[TaskInfo])
    def list_tasks() -> list[TaskInfo]:
        return [TaskInfo(id=t.id, band=t.band, title=t.title, prompt=t.prompt) for t in tasks]

    @app.post("/api/runs", response_model=RunInfo, status_code=202)
    def create_run(req: RunRequest, request: Request) -> RunInfo:
        host = request.client.host if request.client else ""
        local = host in LOCAL_HOSTS
        if not local and not allow_remote:
            if req.task not in by_prompt:
                raise HTTPException(403, "free-text tasks are only accepted from localhost; pick a suite task")
            if req.llm in ("openai", "anthropic"):
                raise HTTPException(403, "hosted models can only be used from localhost")
        if req.llm == "scripted" and req.task not in by_prompt:
            raise HTTPException(400, "the scripted model only knows the suite tasks")
        return registry.submit(req).info()

    @app.get("/api/runs", response_model=list[RunInfo])
    def list_runs() -> list[RunInfo]:
        return [r.info() for r in reversed(registry.runs.values())]

    @app.get("/api/runs/{run_id}", response_model=RunInfo)
    def get_run(run_id: str) -> RunInfo:
        run = registry.runs.get(run_id)
        if not run:
            raise HTTPException(404, "unknown run")
        return run.info()

    @app.get("/api/runs/{run_id}/trace", response_class=PlainTextResponse)
    def get_trace(run_id: str) -> str:
        run = registry.runs.get(run_id)
        if not run:
            raise HTTPException(404, "unknown run")
        return "".join(json.dumps(e) + "\n" for e in run.events)

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request) -> StreamingResponse:
        run = registry.runs.get(run_id)
        if not run:
            raise HTTPException(404, "unknown run")
        start = int(request.headers.get("last-event-id", "-1")) + 1

        async def gen() -> AsyncIterator[str]:
            i, idle = start, 0
            while True:
                if i < len(run.events):
                    yield f"id: {i}\nevent: trace\ndata: {json.dumps(run.events[i])}\n\n"
                    i += 1
                    idle = 0
                    continue
                if run.done.is_set():
                    yield f"event: done\ndata: {json.dumps(run.info().model_dump())}\n\n"
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.05)
                idle += 1
                if idle % 300 == 0:
                    yield ": keep-alive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if settings.web_dir.exists():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    return app
