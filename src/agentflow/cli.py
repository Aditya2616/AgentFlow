"""Command line interface: run a task, serve the UI, run the evaluation, print a trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AgentConfig, Settings


def _print_event(ev) -> None:
    a = ev.attributes
    if ev.event == "start" and ev.name == "execute_tool":
        print(f"  [{ev.t:6.1f}s] tool    {a.get('gen_ai.tool.name')} {json.dumps(a.get('agentflow.arguments'))[:110]}")
    elif ev.event == "end" and ev.name == "agentflow.verify":
        print(f"  [{ev.t:6.1f}s] verify  {a.get('agentflow.verdict')}: {a.get('agentflow.reason')}")
    elif ev.name == "agentflow.plan" and ev.event == "start":
        steps = a.get("agentflow.plan", [])
        print(f"  [{ev.t:6.1f}s] plan    " + " -> ".join(f"{s['tool']}" for s in steps))
    elif ev.event == "end" and ev.name == "chat":
        print(f"  [{ev.t:6.1f}s] llm     {a.get('agentflow.phase', '')} ({a.get('gen_ai.usage.output_tokens', 0)} tokens)")


def cmd_run(args: argparse.Namespace) -> int:
    from .graph import AgentRunner
    from .llm import make_llm
    from .tools.registry import FaultInjector, ToolBox
    from .tracing.writer import Tracer

    settings = Settings()
    settings.ensure_db()
    scripts = None
    if args.llm == "scripted":
        from .evals.tasks import build_tasks

        scripts = {t.prompt: t.script for t in build_tasks(settings.ensure_db())}
    llm = make_llm(args.llm, settings=settings, scripts=scripts, model=args.model,
                   cache_dir=args.cache, device=args.device, dtype=args.dtype)
    trace = Path(args.trace) if args.trace else settings.runs_dir / "cli" / "last.jsonl"
    tracer = Tracer(trace, [_print_event] if not args.quiet else [])
    faults = FaultInjector(args.faults, args.seed) if args.faults else None
    runner = AgentRunner(llm, ToolBox(settings, faults), tracer, AgentConfig(verifier=not args.no_verifier))
    res = runner.run(args.task)
    print(f"\nstatus: {res.status}\nanswer: {res.answer}\ntrace:  {trace}")
    return 0 if res.status == "completed" else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals.run_suite import run_suite
    from .llm import make_llm

    settings = Settings()
    out = Path(args.out) if args.out else None
    arms = {"both": ["full", "off"], "on": ["full"], "off": ["off"], "all": ["off", "retry", "full"]}.get(
        args.verifier, args.verifier.split(","))
    short = args.model.split("/")[-1].lower() if args.model else args.llm
    cache = Path(args.cache) if args.cache else None
    for arm in arms:
        name = args.name or f"{short}_{'faults' + str(int(args.faults * 100)) if args.faults else 'clean'}"
        name = f"{name}_{arm}"
        run_suite(
            name,
            lambda tasks: make_llm(
                args.llm, settings=settings, model=args.model, cache_dir=cache,
                device=args.device, dtype=args.dtype, scripts={t.prompt: t.script for t in tasks},
            ),
            settings=settings, arm=arm, fault_p=args.faults, silent_p=args.silent, seeds=args.seeds,
            send_guard=not args.no_send_guard,
            only=args.only.split(",") if args.only else None, out_dir=out,
            scripted=args.llm == "scripted",
        )
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Dose-response: arms x fault kinds x fault rates x seeds. One result file per cell, resumable."""
    import json

    from .evals.run_suite import RESULTS_DIR, run_suite
    from .llm import make_llm

    settings = Settings()
    out = (Path(args.out) if args.out else RESULTS_DIR) / "dose"
    cache = Path(args.cache) if args.cache else None
    arms = args.arms.split(",")
    rates = [float(r) for r in args.rates.split(",")]
    cells = []
    for rate in rates:
        for kind in (["clean"] if rate == 0 else args.kinds.split(",")):
            cells.append((kind, rate))
    for kind, rate in cells:
        for arm in arms:
            name = f"{kind}_p{int(round(rate * 100)):02d}_{arm}"
            if (out / f"{name}.json").exists():
                print(f"skip {name} (done)")
                continue
            run_suite(
                name,
                lambda tasks: make_llm(args.llm, settings=settings, model=args.model, cache_dir=cache,
                                       device=args.device, dtype=args.dtype),
                settings=settings, arm=arm,
                fault_p=rate if kind == "transient" else 0.0, silent_p=rate if kind == "silent" else 0.0,
                seeds=1 if rate == 0 else args.seeds, only=args.only.split(",") if args.only else None,
                out_dir=out, trace_dir=settings.runs_dir / "eval" / "dose" / name,
                progress=lambda m: print(m, flush=True) if m.endswith("s)") and " t24 " in m else None,
            )
            print(f"done {name}", flush=True)
    print(json.dumps({"cells": len(cells) * len(arms)}))
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    from .evals.rescore import rescore_all

    for name, (before, after) in rescore_all().items():
        print(f"{name}: {before} -> {after}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .evals.report import write_report

    print(write_report())
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .tracing.writer import read_trace

    for ev in read_trace(args.trace):
        _print_event(ev)
    return 0


def cmd_build_data(args: argparse.Namespace) -> int:
    print(Settings().ensure_db())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentflow", description="Plan, act, verify agent with five tools.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one task")
    r.add_argument("task")
    r.add_argument("--llm", default=None, choices=["local", "scripted", "openai", "anthropic"])
    r.add_argument("--model", default=None, help="local model id, default Qwen/Qwen3-0.6B")
    r.add_argument("--no-verifier", action="store_true")
    r.add_argument("--faults", type=float, default=0.0, help="probability of an injected transient tool failure")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--trace", default=None)
    r.add_argument("--device", default="auto", help="auto, cpu or cuda (env AGENTFLOW_DEVICE)")
    r.add_argument("--dtype", default="auto", help="auto, float32, float16 or bfloat16 (env AGENTFLOW_DTYPE)")
    r.add_argument("--cache", default=None, help="directory for the local-model response cache")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="start the web UI and API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8102)
    s.set_defaults(fn=cmd_serve)

    e = sub.add_parser("eval", help="run the task suite")
    e.add_argument("--llm", default="scripted", choices=["local", "scripted", "openai", "anthropic"])
    e.add_argument("--model", default=None)
    e.add_argument("--verifier", default="both",
                   help="arms to run: both (full,off), on, off, all (off,retry,full) or a comma list of off,retry,full")
    e.add_argument("--faults", type=float, default=0.0, help="rate of injected transient tool failures")
    e.add_argument("--silent", type=float, default=0.0, help="rate of injected silently wrong tool results")
    e.add_argument("--no-send-guard", action="store_true", help="allow emails to addresses the request does not name")
    e.add_argument("--seeds", type=int, default=1)
    e.add_argument("--only", default=None, help="comma separated task ids")
    e.add_argument("--name", default=None)
    e.add_argument("--out", default=None)
    e.add_argument("--device", default="auto", help="auto, cpu or cuda (env AGENTFLOW_DEVICE)")
    e.add_argument("--dtype", default="auto", help="auto, float32, float16 or bfloat16 (env AGENTFLOW_DTYPE)")
    e.add_argument("--cache", default=None)
    e.set_defaults(fn=cmd_eval)

    d = sub.add_parser("ablate", help="dose-response over fault rates: off / retry-only / full verifier")
    d.add_argument("--llm", default="local", choices=["local", "openai", "anthropic"])
    d.add_argument("--model", default=None)
    d.add_argument("--arms", default="off,retry,full")
    d.add_argument("--kinds", default="transient,silent")
    d.add_argument("--rates", default="0,0.15,0.3,0.5")
    d.add_argument("--seeds", type=int, default=5)
    d.add_argument("--only", default=None)
    d.add_argument("--out", default=None)
    d.add_argument("--device", default="auto")
    d.add_argument("--dtype", default="auto")
    d.add_argument("--cache", default=None)
    d.set_defaults(fn=cmd_ablate)

    sub.add_parser("rescore", help="re-score stored results with the current checkers (offline)").set_defaults(
        fn=cmd_rescore)
    sub.add_parser("eval-report", help="write evals/results/SUMMARY.md").set_defaults(fn=cmd_report)
    t = sub.add_parser("replay", help="print a recorded trace")
    t.add_argument("trace")
    t.set_defaults(fn=cmd_replay)
    sub.add_parser("build-data", help="build the sample database").set_defaults(fn=cmd_build_data)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
