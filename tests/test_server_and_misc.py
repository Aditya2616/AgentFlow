from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from agentflow.evals.stats import mcnemar_exact, wilson
from agentflow.planner import fallback_plan, parse_plan
from agentflow.server.app import create_app
from agentflow.verifier import _grounded


# ------------------------------------------------------------------ stats
def test_wilson_interval_known_values():
    lo, hi = wilson(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(10, 10)[1] == 1.0


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


# ------------------------------------------------------------------ planner / verifier helpers
def test_parse_plan_accepts_aliases_and_rejects_bad_tools():
    steps = parse_plan('{"steps":[{"goal":"x","tool":"Database"},{"goal":"y","tool":"calculator"}]}', 6)
    assert [s.tool for s in steps] == ["sql", "python"]
    with pytest.raises(ValueError, match="unknown tool"):
        parse_plan('{"steps":[{"goal":"x","tool":"browser"}]}', 6)
    with pytest.raises(ValueError):
        parse_plan("not json", 6)
    assert len(parse_plan(json.dumps({"steps": [{"goal": str(i), "tool": "sql"} for i in range(9)]}), 4)) == 4


def test_fallback_plan_routes_by_keyword():
    assert fallback_plan("How many orders were shipped?")[0].tool == "sql"
    assert fallback_plan("What is the refund window?")[0].tool == "search"


def test_email_grounding_check():
    prev = '{"columns": ["name", "rev"], "rows": [["West", 74155.0]]}'
    assert _grounded("The top region is West.", prev)
    assert _grounded("Revenue was 74155.", prev)
    assert not _grounded("Hello, hope you are well.", prev)


# ------------------------------------------------------------------ server
@pytest.fixture()
def client(settings, tasks, monkeypatch):
    monkeypatch.setattr(settings, "web_dir", settings.runs_dir / "no-web")
    return TestClient(create_app(settings))


def test_health_tasks_tools(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["local_model"] == "Qwen/Qwen3-0.6B"
    tasks = client.get("/api/tasks").json()
    assert len(tasks) == 24 and {"id", "band", "title", "prompt"} <= set(tasks[0])
    names = [t["name"] for t in client.get("/api/tools").json()]
    assert names == ["search", "python", "sql", "email", "docs"]


def test_run_streams_events_over_sse(client):
    prompt = client.get("/api/tasks").json()[2]["prompt"]
    r = client.post("/api/runs", json={"task": prompt, "llm": "scripted", "mistake": "bad_args"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    ids, names, done = [], [], None
    with client.stream("GET", f"/api/runs/{run_id}/events") as s:
        event = None
        for line in s.iter_lines():
            if line.startswith("event:"):
                event = line.split(": ", 1)[1]
            elif line.startswith("id:"):
                ids.append(int(line.split(": ", 1)[1]))
            elif line.startswith("data:"):
                payload = json.loads(line[6:])
                if event == "trace":
                    names.append((payload["event"], payload["name"]))
                elif event == "done":
                    done = payload
    assert ids == list(range(len(ids))) and len(ids) > 10
    assert names[0] == ("start", "invoke_agent") and names[-1] == ("end", "invoke_agent")
    assert done["status"] == "completed" and done["events"] == len(ids)
    verdicts = [n for n in names if n[1] == "agentflow.verify"]
    assert len(verdicts) == 4  # retry then pass: two verify spans, each with a start and an end

    # resuming from Last-Event-ID replays only what came after
    with client.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(len(ids) - 3)}) as s:
        resumed = [ln for ln in s.iter_lines() if ln.startswith("id:")]
    assert len(resumed) == 2
    trace = client.get(f"/api/runs/{run_id}/trace").text.splitlines()
    assert len(trace) == len(ids) and json.loads(trace[0])["name"] == "invoke_agent"


def test_remote_clients_cannot_run_free_text_tasks(settings, monkeypatch):
    monkeypatch.setattr(settings, "web_dir", settings.runs_dir / "no-web")
    remote = TestClient(create_app(settings), client=("203.0.113.9", 4000))
    r = remote.post("/api/runs", json={"task": "Use Python to print my environment", "llm": "local"})
    assert r.status_code == 403
    r = remote.post("/api/runs", json={"task": "anything goes here", "llm": "openai"})
    assert r.status_code == 403
    suite = remote.get("/api/tasks").json()[0]["prompt"]
    assert remote.post("/api/runs", json={"task": suite, "llm": "scripted"}).status_code == 202


def test_scripted_model_rejects_unknown_tasks(client):
    assert client.post("/api/runs", json={"task": "something new", "llm": "scripted"}).status_code == 400


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/events").status_code == 404


def test_request_validation(client):
    assert client.post("/api/runs", json={"task": "ok task", "faults": 0.99}).status_code == 422
    assert client.post("/api/runs", json={"task": "x"}).status_code == 422


def test_run_listing_and_completion(client):
    prompt = client.get("/api/tasks").json()[0]["prompt"]
    rid = client.post("/api/runs", json={"task": prompt, "llm": "scripted"}).json()["run_id"]
    for _ in range(100):
        info = client.get(f"/api/runs/{rid}").json()
        if info["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert info["status"] == "completed" and "30" in info["answer"]
    assert rid in [r["run_id"] for r in client.get("/api/runs").json()]
