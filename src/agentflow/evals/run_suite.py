"""Run the task suite for one configuration and save a results file."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import AgentConfig, Settings
from .harness import run_task
from .stats import wilson
from .tasks import Task, build_tasks, with_mistake

RESULTS_DIR = Path(__file__).resolve().parents[3] / "evals" / "results"
MISTAKE_CYCLE = ["none", "bad_args", "bad_plan"]


def scripted_suite(tasks: list[Task]) -> list[Task]:
    """Oracle scripts with a deterministic mix of deliberate mistakes (adversarial tasks untouched)."""
    return [
        t if t.band == "adversarial" else with_mistake(t, MISTAKE_CYCLE[i % 3]) for i, t in enumerate(tasks)
    ]


def run_suite(
    name: str,
    make_llm: Callable[[list[Task]], BaseChatModel],
    *,
    settings: Settings | None = None,
    verifier: bool = True,
    fault_p: float = 0.0,
    seeds: int = 1,
    tasks: list[Task] | None = None,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    trace_dir: Path | None = None,
    scripted: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    settings = settings or Settings()
    tasks = tasks or build_tasks(settings.ensure_db())
    if only:
        tasks = [t for t in tasks if t.id in only]
    if scripted:
        tasks = scripted_suite(tasks)
    llm = make_llm(tasks)
    out_dir = out_dir or RESULTS_DIR
    trace_dir = trace_dir or (settings.runs_dir / "eval" / name)
    cfg = AgentConfig(verifier=verifier)
    rows = []
    started = time.time()
    for seed in range(seeds):
        for t in tasks:
            tp = trace_dir / f"{t.id}_s{seed}.jsonl"
            r = run_task(t, llm, settings=settings, cfg=cfg, fault_p=fault_p, seed=seed, trace_path=tp)
            r.trace = tp.name
            rows.append(r.to_dict())
            progress(f"[{name}] {t.id} seed {seed}: {'ok' if r.success else 'FAIL'} "
                     f"({r.status}, {r.acts} acts, {r.retries} retries, {r.replans} replans, {r.seconds}s)")
    rt = llm.runtime() if hasattr(llm, "runtime") else {}
    k = sum(1 for r in rows if r["success"])
    lo, hi = wilson(k, len(rows))
    result = {
        "name": name,
        "model": getattr(llm, "model_name", "unknown"),
        "device": rt.get("device", "n/a"),
        "dtype": rt.get("dtype", "n/a"),
        "verifier": verifier,
        "fault_p": fault_p,
        "seeds": seeds,
        "n": len(rows),
        "successes": k,
        "rate": round(k / len(rows), 4) if rows else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "wall_seconds": round(time.time() - started, 1),
        "platform": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
