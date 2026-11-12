"""Record runs into web/recordings/ so the static demo (docs/) can replay them.

Scripted recordings are made here from the real code. Recordings from the local model are
copied from evaluation traces (see --local), so every recorded run is a real run.

    python scripts/record_runs.py                      # scripted mechanics recordings
    python scripts/record_runs.py --local NAME:TASK:SEED:TITLE:NOTE ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from agentflow.config import AgentConfig, Settings
from agentflow.evals.tasks import build_tasks, with_mistake
from agentflow.graph import AgentRunner
from agentflow.llm import make_llm
from agentflow.tools.registry import FaultInjector, ToolBox
from agentflow.tracing.writer import Tracer, read_trace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "recordings"

SCRIPTED = [
    # (file, title, task id, mistake, verifier, fault_p, seed, note)
    ("scripted-retry", "Scripted: wrong first argument, verifier retries", "t09", "bad_args", True, 0.0, 0,
     "Scripted model on purpose: the first SQL query names a table that does not exist. "
     "The verifier sends it back with the database error and the second try passes."),
    ("scripted-replan", "Scripted: wrong plan, verifier re-plans", "t12", "bad_plan", True, 0.0, 0,
     "Scripted model on purpose: the first plan starts with a step that cannot work. After three "
     "failed tries the verifier asks the planner for a new plan."),
    ("scripted-faults", "Scripted: transient tool failures", "t18", "none", True, 0.45, 5,
     "Scripted model, tool faults injected at 45%. A transient failure is retried with the same "
     "arguments, without another model call."),
    ("scripted-no-verifier", "Scripted: the same wrong argument, verifier off", "t09", "bad_args", False, 0.0, 0,
     "Same scripted mistake as the first recording, but the verifier is switched off, so the error "
     "flows into the next step."),
    ("scripted-refusal", "Scripted: a destructive request is stopped", "t22", "none", True, 0.0, 0,
     "The database connection is read-only at the SQLite level. Retries and re-plans cannot change that, "
     "so the run ends with an honest failure."),
]


def record_scripted() -> list[dict]:
    settings = Settings()
    tasks = {t.id: t for t in build_tasks(settings.ensure_db())}
    entries = []
    for file, title, tid, mistake, verifier, fault_p, seed, note in SCRIPTED:
        task = with_mistake(tasks[tid], mistake)
        llm = make_llm("scripted", scripts={task.prompt: task.script})
        path = OUT / f"{file}.jsonl"
        tracer = Tracer(path)
        faults = FaultInjector(fault_p, seed) if fault_p else None
        runner = AgentRunner(llm, ToolBox(settings, faults), tracer, AgentConfig(verifier=verifier))
        res = runner.run(task.prompt)
        entries.append({
            "file": path.name, "title": title, "kind": "scripted", "model": "scripted model",
            "date": date.today().isoformat(), "task": task.prompt, "status": res.status, "note": note,
        })
    return entries


def add_local(spec: str, eval_dir: Path) -> dict:
    name, tid, seed, title, note = spec.split(":", 4)
    src = eval_dir / name / f"{tid}_s{seed}.jsonl"
    dst = OUT / f"local-{name}-{tid}-s{seed}.jsonl"
    shutil.copyfile(src, dst)
    ev = read_trace(dst)
    root = ev[0].attributes
    return {
        "file": dst.name, "title": title, "kind": "local", "model": root.get("gen_ai.request.model", "local"),
        "date": date.fromtimestamp(ev[0].ts).isoformat(), "task": root.get("agentflow.task", ""),
        "status": ev[-1].attributes.get("agentflow.status", ""), "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs="*", default=[], help="NAME:TASK:SEED:TITLE:NOTE")
    ap.add_argument("--eval-dir", default=str(Settings().runs_dir / "eval"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "index.json"
    old = json.loads(index.read_text())["recordings"] if index.exists() else []
    keep_local = [e for e in old if e["kind"] == "local"]
    entries = [add_local(s, Path(args.eval_dir)) for s in args.local] if args.local else keep_local
    entries += record_scripted()
    entries.sort(key=lambda e: e["kind"] != "local")  # real-model recordings first
    index.write_text(json.dumps({"recordings": entries}, indent=1), encoding="utf-8")
    print(f"wrote {len(entries)} recordings to {OUT}")


if __name__ == "__main__":
    main()
