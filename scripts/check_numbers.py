"""Generate, or check, every number that is repeated outside the raw result files.

    python scripts/check_numbers.py --write   # regenerate the derived files and the README results block
    python scripts/check_numbers.py --check   # fail if any of them is out of sync (a fast test runs this)

Inputs: evals/results/dose/*.json, evals/results/larger/dose/*.json, the top-level evals/results/*.json,
evals/first_scoring.json, evals/retrieval.json, and the code.
Derived and checked:
  * evals/results/DOSE.md, SUMMARY.md           (tables and paired tests)
  * web/results.json and docs/results.json      (what the UI's Results view reads)
  * docs/assets/dose.svg                        (the README chart, drawn from the same cells)
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
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentflow.config import AgentConfig  # noqa: E402
from agentflow.evals import breakdown, dose_report, report  # noqa: E402
from agentflow.evals.dose_report import ARM_LABEL, ARMS, KIND_LABEL, KINDS, summarise  # noqa: E402
from agentflow.evals.stats import mcnemar_exact, pct_int, round_half_up, wilson  # noqa: E402
from agentflow.evals.tasks import EMAIL, build_tasks  # noqa: E402

RES = ROOT / "evals" / "results"
DOSE = RES / "dose"
BIG = RES / "larger" / "dose"
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
RATES = (0.0, 0.15, 0.3, 0.5)


def _tasks():
    from agentflow.config import Settings

    return build_tasks(Settings().ensure_db())


def pc(k: int, n: int) -> str:
    """A rate as a whole percentage, halves rounded up (the UI rounds the same way)."""
    return f"{pct_int(k, n)}%"


def pts(k: int, n: int) -> int:
    return pct_int(k, n)


def diff_pts(a: dict, b: dict) -> int:
    """a minus b in percentage points, computed from the unrounded rates and then rounded half up."""
    return round_half_up(float(Fraction(100 * a["k"], a["n"]) - Fraction(100 * b["k"], b["n"])))


def load(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def cell(cells, kind, rate, arm):
    return dose_report._cell_for(cells, kind, rate, arm)


def kn(c) -> tuple[int, int]:
    return sum(r["success"] for r in c["rows"]), len(c["rows"])


def p_of(cells, kind, rate, a, b) -> float:
    _, oa, ob, _ = dose_report.paired(cell(cells, kind, rate, a), cell(cells, kind, rate, b))
    return mcnemar_exact(oa, ob)


def fmt_p(p: float) -> str:
    """The p-value with its operator: '= 0.12' or '< 0.0001'."""
    if p >= 0.1:
        return f"= {p:.2f}"
    return f"= {p:.4f}".rstrip("0") if p >= 0.0001 else "< 0.0001"


def first_scoring() -> dict[str, dict[tuple[str, int], int]]:
    raw = json.loads((ROOT / "evals" / "first_scoring.json").read_text(encoding="utf-8"))
    return {k: {(t, s): ok for t, s, ok in rows} for k, rows in raw.items()}


# ----------------------------------------------------------------------------- facts used in prose
def facts(cells, big) -> dict[str, str]:
    f: dict[str, str] = {}
    tasks = {t.id: t for t in _tasks()}
    clean = {a: cell(cells, "transient", 0.0, a) for a in ARMS}
    f["n_tasks"] = str(len(clean["full"]["rows"]))
    f["n_tasks_less_three"] = str(len(clean["full"]["rows"]) - 3)
    by = {a: {r["task_id"]: r["success"] for r in clean[a]["rows"]} for a in ARMS}
    for a in ARMS:
        k, n = kn(clean[a])
        f[f"clean_{a}_k"], f[f"clean_{a}_pct"] = str(k), pc(k, n)
    f["fail_all"] = str(sum(not any(by[a][t] for a in ARMS) for t in by["full"]))
    only_full = [t for t in by["full"] if by["full"][t] and not by["off"][t] and not by["retry"][t]]
    f["only_full_ids"] = ", ".join(only_full[:-1]) + (" and " if len(only_full) > 1 else "") + only_full[-1] if only_full else "none"
    f["only_full_n"] = str(len(only_full))
    f["p_clean_full_retry"] = fmt_p(p_of(cells, "transient", 0.0, "full", "retry"))
    for kind in KINDS:
        for rate in RATES:
            r = f"{int(rate * 100)}"
            sums = {a: summarise(cell(cells, kind, rate, a)) for a in ARMS}
            for a in ARMS:
                f[f"{kind}_{r}_{a}"] = pc(sums[a]["k"], sums[a]["n"])
                f[f"{kind}_{r}_{a}_calls"] = f"{sums[a]['model_calls']:.1f}"
            f[f"{kind}_{r}_gap"] = str(diff_pts(sums["full"], sums["retry"]))
            f[f"{kind}_{r}_retry_gain"] = str(diff_pts(sums["retry"], sums["off"]))
            if rate:
                f[f"{kind}_{r}_p_full_retry"] = fmt_p(p_of(cells, kind, rate, "full", "retry"))
                f[f"{kind}_{r}_p_retry_off"] = fmt_p(p_of(cells, kind, rate, "retry", "off"))
        for a in ARMS:
            c0, c5 = summarise(cell(cells, kind, 0.0, a)), summarise(cell(cells, kind, 0.5, a))
            f[f"{kind}_drop_{a}"] = str(diff_pts(c0, c5))
    f["seeds"] = str(len(cell(cells, "transient", 0.3, "full")["rows"]) // len(clean["full"]["rows"]))
    f["runs_per_cell"] = str(len(cell(cells, "transient", 0.3, "full")["rows"]))
    # failure breakdown (from FAILURES.md, which is generated from the traces)
    text = (RES / "FAILURES.md").read_text(encoding="utf-8")
    cat = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in
           re.finditer(r"^\| ([a-z][^|]+?) \| (\d+) \| (\d+) \|$", text, re.M)}
    f["cat_planner"] = str(cat.get("planner: needed step missing", (0, 0))[0])
    f["cat_args"] = str(cat.get("bad arguments", (0, 0))[0])
    # send guard
    ng, g = load("qwen3-0.6b_clean_noguard_full"), clean["full"]
    f["mail_noguard"] = str(sum(r["emails_sent"] for r in ng["rows"]))
    f["mail_guard"] = str(sum(r["emails_sent"] for r in g["rows"]))
    stray_ng = [r for r in ng["rows"] if any(m["to"].lower() not in {a.lower() for a in EMAIL.findall(tasks[r["task_id"]].prompt)}
                                             for m in r["outbox"])]
    stray_g = [r for r in g["rows"] if any(m["to"].lower() not in {a.lower() for a in EMAIL.findall(tasks[r["task_id"]].prompt)}
                                           for m in r["outbox"])]
    f["stray_noguard_n"], f["stray_noguard_ids"] = str(len(stray_ng)), ", ".join(r["task_id"] for r in stray_ng)
    f["stray_guard_n"] = str(len(stray_g))
    f["guard_k"], f["noguard_k"] = str(g["successes"]), str(ng["successes"])
    f["n_addr_tasks"] = str(sum("@" in t.prompt for t in tasks.values()))
    # first scoring versus the corrected checkers
    old = first_scoring()
    changed = []
    for a in ARMS:
        was = old[f"dose/clean_p00_{a}"]
        for r in clean[a]["rows"]:
            before = was[(r["task_id"], r["seed"])]
            if bool(before) != bool(r["success"]):
                changed.append(f"{r['task_id']} {'fail' if before and not r['success'] else 'pass'} in the {ARM_LABEL[a]} arm")
    f["chg_clean_tasks"] = "; ".join(changed) if changed else "none"
    for a in ARMS:
        was = sum(old[f"dose/clean_p00_{a}"].values())
        f[f"chg_clean_{a}"] = f"{was} to {clean[a]['successes']}"
    # larger model
    bigclean = {a: load(f"qwen2.5-1.5b_clean_{a}") for a in ARMS}
    for a in ARMS:
        f[f"big_clean_{a}_k"] = str(bigclean[a]["successes"])
        s = summarise(cell(big, "silent", 0.3, a))
        f[f"big_silent_{a}"] = pc(s["k"], s["n"])
        f[f"big_silent_{a}_calls"] = f"{s['model_calls']:.1f}"
    f["big_silent_p_full_retry"] = fmt_p(p_of(big, "silent", 0.3, "full", "retry"))
    f["big_silent_n"] = str(len(cell(big, "silent", 0.3, "full")["rows"]))
    f["big_device"], f["big_dtype"] = bigclean["full"]["device"], bigclean["full"]["dtype"]
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
            row = " | ".join(f"{pct_int(s['k'], s['n'])}% [{round_half_up(100 * s['lo'])}, {round_half_up(100 * s['hi'])}] ({s['model_calls']:.1f})"
                             for s in sums)
            out.append(f"| {rate:.0%} | {sums[0]['n']} | {row} |")
        out.append("")
    return "\n".join(out)


def clean_table() -> str:
    """The two tables and the pass-count line from FAILURES.md, with a blank line wherever the kind of line changes
    (Markdown needs one before and after a table, or the next table renders as rows of the first)."""
    out: list[str] = []
    prev = None
    for ln in (RES / "FAILURES.md").read_text(encoding="utf-8").splitlines():
        kind = "table" if ln.startswith("|") else "pass" if ln.startswith("Passing") else None
        if kind is None:
            continue
        if prev is not None and kind != prev:
            out.append("")
        out.append(ln)
        prev = kind
    return "\n".join(out)


def extra_table(cells) -> str:
    rows = [("Qwen3-0.6B, full verifier, send guard on", cell(cells, "transient", 0.0, "full")),
            ("Qwen3-0.6B, full verifier, send guard off", load("qwen3-0.6b_clean_noguard_full"))]
    body = "\n".join(f"| {label} | {r['successes']}/{r['n']} | {sum(x['emails_sent'] for x in r['rows'])} |" for label, r in rows)
    return "| clean run | passed | emails written |\n|---|---|---|\n" + body


def big_table(cells, big) -> str:
    bigclean = {a: load(f"qwen2.5-1.5b_clean_{a}") for a in ARMS}
    head = "| model | " + " | ".join(f"clean, {ARM_LABEL[a]}" for a in ARMS) + " | " + " | ".join(
        f"30% silent, {ARM_LABEL[a]}" for a in ARMS) + " |"
    sep = "|---|" + "---|" * 6

    def row(label, clean_by, silent_cells):
        cs = [f"{clean_by[a]['successes']}/{clean_by[a]['n']}" for a in ARMS]
        ss = []
        for a in ARMS:
            s = summarise(cell(silent_cells, "silent", 0.3, a))
            ss.append(f"{pc(s['k'], s['n'])} ({s['k']}/{s['n']})")
        return f"| {label} | " + " | ".join(cs + ss) + " |"

    small = {a: cell(cells, "transient", 0.0, a) for a in ARMS}
    return "\n".join([head, sep, row("Qwen3-0.6B (default)", small, cells),
                      row("Qwen2.5-1.5B-Instruct", bigclean, big)])


def scoring_table(cells) -> str:
    old = first_scoring()
    rows = ["| fault rate | kind | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " |", "|---|---|---|---|---|"]
    for kind in ("clean", *KINDS):
        for rate in ((0.0,) if kind == "clean" else RATES[1:]):
            cs = []
            for a in ARMS:
                new = cell(cells, "transient" if kind == "clean" else kind, rate, a)
                k, n = kn(new)
                key = f"dose/{kind}_p{int(round(rate * 100)):02d}_{a}"
                was = sum(old[key].values())
                cs.append(f"{pts(was, n)}% to {pts(k, n)}%")
            rows.append(f"| {rate:.0%} | {'none' if kind == 'clean' else kind} | " + " | ".join(cs) + " |")
    return "\n".join(rows)


def render_block(cells, big) -> str:
    text = (ROOT / "scripts" / "readme_results.md").read_text(encoding="utf-8")
    f = facts(cells, big)
    text = (text.replace("{{dose}}", dose_tables(cells)).replace("{{clean}}", clean_table())
            .replace("{{extra}}", extra_table(cells)).replace("{{big}}", big_table(cells, big))
            .replace("{{scoring}}", scoring_table(cells)))
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
def readme_facts() -> list[tuple[str, str, str]]:
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


def check_arithmetic(cells, big) -> list[str]:
    bad = []
    for group, seeds_nonzero in ((cells, 5), (big, 5)):
        for key, c in group.items():
            rows, name = c["rows"], "/".join(map(str, key))
            k = sum(r["success"] for r in rows)
            if (k, len(rows)) != (c["successes"], c["n"]):
                bad.append(f"{name}: successes/n do not match the rows")
            lo, hi = wilson(k, len(rows))
            if [round(lo, 4), round(hi, 4)] != c["ci95"]:
                bad.append(f"{name}: stored interval differs from the one recomputed from the counts")
            seeds = 1 if key[1] == 0 else seeds_nonzero
            if len(rows) != len({r["task_id"] for r in rows}) * seeds:
                bad.append(f"{name}: expected tasks x {seeds} seeds rows")
    for p in sorted(RES.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if sum(x["success"] for x in r["rows"]) != r["successes"] or len(r["rows"]) != r["n"]:
            bad.append(f"{p.name}: successes/n do not match the rows")
    return bad


def rescoring_agrees() -> list[str]:
    """Stored successes must equal what the current checkers give for the stored answers and side effects."""
    from agentflow.evals.harness import judge
    from agentflow.evals.tasks import CheckCtx

    tasks = {t.id: t for t in _tasks()}
    bad = []
    for p in sorted(RES.rglob("*.json")):
        for x in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            ctx = CheckCtx(x["answer"], x["status"], x["outbox"], x["db_rows"][0], x["db_rows"][1], tasks[x["task_id"]].prompt)
            if judge(tasks[x["task_id"]], x["status"], ctx) != x["success"]:
                bad.append(f"{p.relative_to(RES)}: {x['task_id']} seed {x['seed']} is scored differently by the current checkers; "
                           "run agentflow rescore")
                break
    return bad


def failures_agree(cells) -> list[str]:
    text = (RES / "FAILURES.md").read_text(encoding="utf-8")
    m = re.search(r"Passing: off (\d+)/(\d+), retry only (\d+)/\d+, full (\d+)/\d+\.", text)
    exp = [str(kn(cell(cells, "transient", 0.0, a))[0]) for a in ARMS]
    got = [m.group(1), m.group(3), m.group(4)] if m else []
    return [] if got == exp else [f"FAILURES.md pass counts {got} differ from the result files {exp}"]


# ----------------------------------------------------------------------------- driver
def derived(cells, big) -> dict[Path, str]:
    return {
        RES / "DOSE.md": dose_report.build_markdown(cells),
        RES / "SUMMARY.md": report.build_report(RES),
        ROOT / "web" / "results.json": json.dumps(dose_report.results_json(cells), indent=1),
        ROOT / "docs" / "results.json": json.dumps(dose_report.results_json(cells), indent=1),
        ROOT / "docs" / "assets" / "dose.svg": dose_report.chart_svg(cells),
        README: readme_with(render_block(cells, big)),
    }


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n") if p.exists() else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    cells, big = dose_report.load_cells(DOSE), dose_report.load_cells(BIG)
    problems = check_arithmetic(cells, big)
    if args.write:
        if (ROOT / "runs" / "eval" / "dose" / "clean_p00_full").exists():
            breakdown.write()
        for path, text in derived(cells, big).items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        print("derived files and README results written")
    problems += failures_agree(cells) + rescoring_agrees()
    if args.check:
        for path, text in derived(cells, big).items():
            if read(path).rstrip("\n") != text.replace("\r\n", "\n").rstrip("\n"):
                problems.append(f"{path.relative_to(ROOT)} is out of sync; run scripts/check_numbers.py --write")
    readme = read(README)
    outside = re.sub(re.escape(START) + ".*?" + re.escape(END), "", readme, flags=re.S)
    for what, rx, expected in readme_facts():
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
