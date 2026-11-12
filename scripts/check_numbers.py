"""Generate, or check, every number that is repeated outside the raw result files.

    python scripts/check_numbers.py --write   # regenerate the derived files and the README results block
    python scripts/check_numbers.py --check   # fail if any of them is out of sync (a fast test runs this)

Inputs: evals/results/dose/*.json, the top-level evals/results/*.json, evals/retrieval.json, the code.
Derived and checked:
  * evals/results/DOSE.md, SUMMARY.md           (tables and paired tests)
  * web/results.json and docs/results.json      (what the UI's Results view reads)
  * docs/assets/dose.svg                        (the README chart)
  * README.md between the RESULTS markers       (tables plus the prose numbers, from scripts/readme_results.md)
  * README.md facts outside the markers         (counts and settings, checked against the code)
  * internal arithmetic of the result files     (counts, totals, intervals)
FAILURES.md needs the git-ignored traces, so --write regenerates it only when they exist; --check verifies that its
pass counts agree with the result files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentflow.config import AgentConfig  # noqa: E402
from agentflow.evals import breakdown, dose_report, report  # noqa: E402
from agentflow.evals.dose_report import ARM_LABEL, ARMS, KIND_LABEL, KINDS, summarise  # noqa: E402
from agentflow.evals.stats import mcnemar_exact, wilson  # noqa: E402
from agentflow.evals.tasks import build_tasks  # noqa: E402

RES = ROOT / "evals" / "results"
DOSE = RES / "dose"
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _tasks():
    from agentflow.config import Settings

    return build_tasks(Settings().ensure_db())


def pc(k: int, n: int) -> str:
    return f"{round(100 * k / n)}%"


def load(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def cell(cells, kind, rate, arm):
    return dose_report._cell_for(cells, kind, rate, arm)


def pair_p(cells, kind, rate, a, b) -> float:
    _, oa, ob, _ = dose_report.paired(cell(cells, kind, rate, a), cell(cells, kind, rate, b))
    return mcnemar_exact(oa, ob)


def fmt_p(p: float) -> str:
    return f"{p:.2f}" if p >= 0.1 else f"{p:.4f}".rstrip("0") if p >= 0.0001 else "< 0.0001"


# ----------------------------------------------------------------------------- facts used in prose
def facts(cells) -> dict[str, str]:
    f: dict[str, str] = {}
    clean = {a: cell(cells, "transient", 0.0, a) for a in ARMS}
    f["n_tasks"] = str(len(clean["full"]["rows"]))
    for a in ARMS:
        k = sum(r["success"] for r in clean[a]["rows"])
        f[f"clean_{a}_k"] = str(k)
        f[f"clean_{a}_pct"] = pc(k, len(clean[a]["rows"]))
    by = {a: {r["task_id"]: r["success"] for r in clean[a]["rows"]} for a in ARMS}
    f["fail_all"] = str(sum(not any(by[a][t] for a in ARMS) for t in by["full"]))
    for kind in KINDS:
        for rate in (0.15, 0.3, 0.5):
            r = f"{int(rate * 100)}"
            for a in ARMS:
                s = summarise(cell(cells, kind, rate, a))
                f[f"{kind}_{r}_{a}"] = pc(s["k"], s["n"])
                f[f"{kind}_{r}_{a}_calls"] = f"{s['model_calls']:.1f}"
            f[f"{kind}_{r}_p_full_retry"] = fmt_p(pair_p(cells, kind, rate, "full", "retry"))
            f[f"{kind}_{r}_p_retry_off"] = fmt_p(pair_p(cells, kind, rate, "retry", "off"))
    f["silent_gap_30"] = str(round(100 * summarise(cell(cells, "silent", 0.3, "full"))["rate"]) -
                              round(100 * summarise(cell(cells, "silent", 0.3, "retry"))["rate"]))
    f["silent_gap_50"] = str(round(100 * summarise(cell(cells, "silent", 0.5, "full"))["rate"]) -
                              round(100 * summarise(cell(cells, "silent", 0.5, "retry"))["rate"]))
    f["seeds"] = str(len(cell(cells, "transient", 0.3, "full")["rows"]) // len(clean["full"]["rows"]))
    f["runs_per_cell"] = str(len(cell(cells, "transient", 0.3, "full")["rows"]))
    # failure breakdown (from FAILURES.md, which is generated from the traces)
    text = (RES / "FAILURES.md").read_text(encoding="utf-8")
    cat = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\| ([a-z][^|]+?) \| (\d+) \| \d+ \|$", text, re.M)}
    f["cat_planner"] = str(cat.get("planner: needed step missing", 0))
    f["cat_args"] = str(cat.get("bad arguments", 0))
    # send guard
    ng, g = load("qwen3-0.6b_clean_noguard_full"), cell(cells, "transient", 0.0, "full")
    tasks = {t.id: t for t in _tasks()}
    f["mail_noguard"] = str(sum(r["emails_sent"] for r in ng["rows"]))
    f["mail_guard"] = str(sum(r["emails_sent"] for r in g["rows"]))
    unasked = [r["task_id"] for r in ng["rows"] if r["emails_sent"] and "@" not in tasks[r["task_id"]].prompt]
    f["mail_unasked_n"] = str(len(unasked))
    f["mail_unasked_ids"] = ", ".join(unasked)
    f["mail_guard_unasked"] = str(sum(1 for r in g["rows"] if r["emails_sent"] and "@" not in tasks[r["task_id"]].prompt))
    f["guard_k"], f["noguard_k"] = str(g["successes"]), str(ng["successes"])
    f["n_addr_tasks"] = str(sum("@" in t.prompt for t in tasks.values()))
    # larger model
    big_off, big_on = load("qwen2.5-1.5b_clean_noverify"), load("qwen2.5-1.5b_clean_verify")
    f["big_off_k"], f["big_on_k"] = str(big_off["successes"]), str(big_on["successes"])
    f["big_seconds"] = str(round(sum(r["seconds"] for r in big_on["rows"]) / len(big_on["rows"])))
    cfg = AgentConfig()
    f["max_attempts"], f["max_replans"] = str(cfg.max_attempts), str(cfg.max_replans)
    f["device"], f["dtype"] = clean["full"]["device"], clean["full"]["dtype"]
    ret = {r["mode"]: r for r in json.loads((ROOT / "evals" / "retrieval.json").read_text(encoding="utf-8"))}
    f["ret_n"] = str(ret["bm25"]["n"])
    for m in ("bm25", "hybrid"):
        for k in ("recall@1", "recall@3"):
            f[f"ret_{m}_{k}".replace("@", "")] = f"{ret[m][k]:.3f}"
    return f


# ----------------------------------------------------------------------------- tables
def dose_tables(cells) -> str:
    out = []
    for kind in KINDS:
        out += [f"**{KIND_LABEL[kind]}** (success with its 95% Wilson interval; mean model calls per task in brackets)", "",
                "| fault rate | runs per arm | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " |", "|---|---|---|---|---|"]
        for rate in dose_report.rates_for(cells):
            sums = [summarise(cell(cells, kind, rate, a)) for a in ARMS]
            row = " | ".join(f"{100 * s['rate']:.0f}% [{100 * s['lo']:.0f}, {100 * s['hi']:.0f}] ({s['model_calls']:.1f})"
                             for s in sums)
            out.append(f"| {rate:.0%} | {sums[0]['n']} | {row} |")
        out.append("")
    return "\n".join(out)


def clean_table() -> str:
    text = (RES / "FAILURES.md").read_text(encoding="utf-8").splitlines()
    return "\n".join(ln for ln in text if ln.startswith("|") or ln.startswith("Passing"))


def extra_table(cells) -> str:
    full = cell(cells, "transient", 0.0, "full")
    rows = [("Qwen3-0.6B, full verifier, send guard on", full), ("Qwen3-0.6B, full verifier, send guard off", load("qwen3-0.6b_clean_noguard_full")),
            ("Qwen2.5-1.5B-Instruct, no verifier", load("qwen2.5-1.5b_clean_noverify")),
            ("Qwen2.5-1.5B-Instruct, full verifier (run before the send guard existed)", load("qwen2.5-1.5b_clean_verify"))]
    body = "\n".join(f"| {label} | {r['successes']}/{r['n']} | {sum(x['emails_sent'] for x in r['rows']) if 'emails_sent' in r['rows'][0] else 'not recorded'} | {r['device']}, {r['dtype']} |"
                     for label, r in rows)
    return "| configuration (clean run) | passed | emails written | device, dtype |\n|---|---|---|---|\n" + body


def render_block(cells) -> str:
    text = (ROOT / "scripts" / "readme_results.md").read_text(encoding="utf-8")
    f = facts(cells)
    text = (text.replace("{{dose}}", dose_tables(cells)).replace("{{clean}}", clean_table())
            .replace("{{extra}}", extra_table(cells)))
    for key, val in f.items():
        text = text.replace("{{" + key + "}}", val)
    left = re.findall(r"\{\{(\w+)\}\}", text)
    if left:
        raise SystemExit(f"unfilled slots in readme_results.md: {sorted(set(left))}")
    return text.rstrip("\n")


def readme_with(block: str) -> str:
    text = README.read_text(encoding="utf-8")
    new, n = re.subn(re.escape(START) + ".*?" + re.escape(END), lambda m: f"{START}\n{block}\n{END}", text, flags=re.S)
    if n != 1:
        raise SystemExit("RESULTS markers not found in README.md")
    return new


# ----------------------------------------------------------------------------- facts outside the block
def readme_facts(cells) -> list[tuple[str, str, str]]:
    """(description, regex with one group, expected) checked against README.md text outside the block."""
    cfg = AgentConfig()
    tasks = _tasks()
    blocked = len(re.findall(r'^    "', (ROOT / "tests" / "test_sandbox.py").read_text(encoding="utf-8").split("BLOCKED = [")[1]
                            .split("]\n")[0], re.M))
    corpus = len(list((ROOT / "src" / "agentflow" / "data" / "corpus").glob("*.md")))
    views = re.search(r"VIEWS = \{(.*?)\n\}", (ROOT / "scripts" / "screenshots.py").read_text(encoding="utf-8"), re.S).group(1)
    desk = re.search(r'"width": (\d+), "height": (\d+)', views)
    phone = re.findall(r'"width": (\d+), "height": (\d+)', views)[1]
    port = re.search(r'"--port", type=int, default=(\d+)', (ROOT / "src" / "agentflow" / "cli.py").read_text(encoding="utf-8")).group(1)
    return [
        ("red-team samples", r"(\d+) code samples the static gate", str(blocked)),
        ("suite size in quickstart", r"prompts of the (\d+) suite tasks", str(len(tasks))),
        ("suite size in limitations", r"\((\d+) tasks over a small synthetic database", str(len(tasks))),
        ("corpus size", r"and (\d+) documents I wrote", str(corpus)),
        ("tries per step", r"(\d+) tries per step", str(cfg.max_attempts)),
        ("re-plans", r"tries per step, (\d+) re-plans", str(cfg.max_replans)),
        ("port", r"UI and API on http://127\.0\.0\.1:(\d+)", port),
        ("desktop size", r"at (\d+x\d+) and", f"{desk.group(1)}x{desk.group(2)}"),
        ("phone size", r"and (\d+x\d+) in both", f"{phone[0]}x{phone[1]}"),
    ]


def check_arithmetic(cells) -> list[str]:
    bad = []
    for key, c in cells.items():
        rows, name = c["rows"], "/".join(map(str, key))
        k = sum(r["success"] for r in rows)
        if (k, len(rows)) != (c["successes"], c["n"]):
            bad.append(f"{name}: successes/n do not match the rows")
        lo, hi = wilson(k, len(rows))
        if [round(lo, 4), round(hi, 4)] != c["ci95"]:
            bad.append(f"{name}: stored interval differs from the one recomputed from the counts")
        seeds = 1 if key[1] == 0 else 5
        if len(rows) != len({r["task_id"] for r in rows}) * seeds:
            bad.append(f"{name}: expected tasks x {seeds} seeds rows")
    for p in sorted(RES.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if sum(x["success"] for x in r["rows"]) != r["successes"] or len(r["rows"]) != r["n"]:
            bad.append(f"{p.name}: successes/n do not match the rows")
    return bad


def failures_agree(cells) -> list[str]:
    text = (RES / "FAILURES.md").read_text(encoding="utf-8")
    m = re.search(r"Passing: off (\d+)/(\d+), retry only (\d+)/\d+, full (\d+)/\d+\.", text)
    exp = [str(sum(r["success"] for r in cell(cells, "transient", 0.0, a)["rows"])) for a in ARMS]
    got = [m.group(1), m.group(3), m.group(4)] if m else []
    return [] if got == exp else [f"FAILURES.md pass counts {got} differ from the result files {exp}"]


# ----------------------------------------------------------------------------- driver
def derived(cells) -> dict[Path, str]:
    out = {
        RES / "DOSE.md": dose_report.build_markdown(cells),
        RES / "SUMMARY.md": report.build_report(RES),
        ROOT / "web" / "results.json": json.dumps(dose_report.results_json(cells), indent=1),
        ROOT / "docs" / "results.json": json.dumps(dose_report.results_json(cells), indent=1),
        ROOT / "docs" / "assets" / "dose.svg": dose_report.chart_svg(cells),
        README: readme_with(render_block(cells)),
    }
    return out


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n") if p.exists() else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    cells = dose_report.load_cells(DOSE)
    problems = check_arithmetic(cells)
    if args.write:
        if (ROOT / "runs" / "eval" / "dose" / "clean_p00_full").exists():
            breakdown.write()
        for path, text in derived(cells).items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        print("derived files and README results written")
    problems += failures_agree(cells)
    if args.check:
        for path, text in derived(cells).items():
            if read(path).rstrip("\n") != text.replace("\r\n", "\n").rstrip("\n"):
                problems.append(f"{path.relative_to(ROOT)} is out of sync; run scripts/check_numbers.py --write")
    readme = read(README)
    outside = re.sub(re.escape(START) + ".*?" + re.escape(END), "", readme, flags=re.S)
    for what, rx, expected in readme_facts(cells):
        m = re.search(rx, outside)
        if not m:
            problems.append(f"README: could not find the sentence for '{what}'")
        elif m.group(1) != expected:
            problems.append(f"README: {what} says {m.group(1)}, the code or results say {expected}")
    for p in problems:
        print("MISMATCH:", p)
    if not problems:
        print("numbers in sync")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
