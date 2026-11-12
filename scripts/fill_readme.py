"""Put the generated result tables into README.md between the RESULTS markers.

    python scripts/fill_readme.py

The tables come from evals/results/ (run `agentflow eval-report` first). The prose around them is in
scripts/readme_results.md, which has {{dose}}, {{clean}} and {{extra}} slots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agentflow.evals.dose_report import ARM_LABEL, ARMS, KIND_LABEL, KINDS, _cell_for, load_cells, rates_for, summarise

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"


def dose_tables() -> str:
    cells = load_cells(RES / "dose")
    out = []
    for kind in KINDS:
        out += [f"**{KIND_LABEL[kind]}** (success, 95% Wilson interval; mean model calls per task in brackets)", "",
                "| fault rate | n per arm | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " |", "|---|---|---|---|---|"]
        for rate in rates_for(cells):
            sums = [summarise(_cell_for(cells, kind, rate, a)) for a in ARMS]
            row = " | ".join(f"{100 * s['rate']:.0f}% [{100 * s['lo']:.0f}, {100 * s['hi']:.0f}] ({s['model_calls']:.1f})"
                             for s in sums)
            out.append(f"| {rate:.0%} | {sums[0]['n']} | {row} |")
        out.append("")
    return "\n".join(out)


def clean_table() -> str:
    text = (RES / "FAILURES.md").read_text(encoding="utf-8").splitlines()
    keep = [ln for ln in text if ln.startswith("|") or ln.startswith("Passing")]
    return "\n".join(keep)


def extra_table() -> str:
    rows = []
    for name, label in (("qwen3-0.6b_clean_noguard_full", "Qwen3-0.6B, full verifier, send guard off"),
                        ("qwen2.5-1.5b_clean_noverify", "Qwen2.5-1.5B-Instruct, no verifier"),
                        ("qwen2.5-1.5b_clean_verify", "Qwen2.5-1.5B-Instruct, full verifier (before the send guard existed)")):
        p = RES / f"{name}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            rows.append(f"| {label} | {r['successes']}/{r['n']} | {r['device']}, {r['dtype']} |")
    full = json.loads((RES / "dose" / "clean_p00_full.json").read_text(encoding="utf-8"))
    rows.insert(0, f"| Qwen3-0.6B, full verifier, send guard on | {full['successes']}/{full['n']} | {full['device']}, {full['dtype']} |")
    return "| configuration (clean run, 24 tasks) | passed | device, dtype |\n|---|---|---|\n" + "\n".join(rows)


def main() -> None:
    prose = (ROOT / "scripts" / "readme_results.md").read_text(encoding="utf-8")
    block = (prose.replace("{{dose}}", dose_tables()).replace("{{clean}}", clean_table())
             .replace("{{extra}}", extra_table()))
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    new, n = re.subn(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
                     lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", text, flags=re.S)
    if n != 1:
        raise SystemExit("RESULTS markers not found in README.md")
    readme.write_text(new, encoding="utf-8")
    print("README results updated")


if __name__ == "__main__":
    main()
