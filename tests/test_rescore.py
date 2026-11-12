"""Stored results can be re-scored offline when a checker changes."""

from __future__ import annotations

import json

from agentflow.evals.rescore import rescore_file


def test_rescore_updates_success_and_counts_from_stored_fields(tasks, tmp_path):
    t = tasks["t23"]
    rows = [
        {"task_id": "t23", "seed": 0, "success": False, "status": "completed", "answer": "The CEO's salary is not provided.",
         "outbox": [], "db_rows": [5, 5]},
        {"task_id": "t23", "seed": 1, "success": True, "status": "completed", "answer": "It is $250,000.",
         "outbox": [], "db_rows": [5, 5]},
    ]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rows": rows, "successes": 1, "n": 2, "rate": 0.5, "ci95": [0, 1]}))
    assert rescore_file(f, {"t23": t}) == (1, 1)
    out = json.loads(f.read_text())
    assert [r["success"] for r in out["rows"]] == [True, False] and out["successes"] == 1
