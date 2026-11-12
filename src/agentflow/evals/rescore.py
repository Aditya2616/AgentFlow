"""Re-score stored results with the current checkers, offline: no model and no tool is run.

Each result row stores the final answer, the status, the emails written and the order counts, which is
everything a checker looks at. Use it after changing a checker (`agentflow rescore`).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Settings
from .harness import judge
from .run_suite import RESULTS_DIR
from .stats import wilson
from .tasks import CheckCtx, build_tasks


def rescore_file(path: Path, tasks: dict) -> tuple[int, int]:
    """Rewrite one result file in place. Returns (successes before, successes after)."""
    r = json.loads(path.read_text(encoding="utf-8"))
    before = sum(x["success"] for x in r["rows"])
    for x in r["rows"]:
        if "db_rows" not in x:
            raise ValueError(f"{path.name}: rows were stored before re-scoring was supported; re-run it")
        ctx = CheckCtx(x["answer"], x["status"], x["outbox"], x["db_rows"][0], x["db_rows"][1], tasks[x["task_id"]].prompt)
        x["success"] = judge(tasks[x["task_id"]], x["status"], ctx)
    k = sum(x["success"] for x in r["rows"])
    lo, hi = wilson(k, len(r["rows"]))
    r.update(successes=k, rate=round(k / len(r["rows"]), 4), ci95=[round(lo, 4), round(hi, 4)])
    path.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return before, k


def rescore_all(root: Path | None = None) -> dict[str, tuple[int, int]]:
    root = root or RESULTS_DIR
    tasks = {t.id: t for t in build_tasks(Settings().ensure_db())}
    return {str(p.relative_to(root)): rescore_file(p, tasks) for p in sorted(root.rglob("*.json"))}
