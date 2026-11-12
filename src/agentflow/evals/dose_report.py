"""Aggregate the dose-response results (evals/results/dose/*.json) into tables, a chart and web/results.json.

Everything printed in the README and drawn in the UI comes from the result files through this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from .run_suite import RESULTS_DIR
from .stats import mcnemar_exact, wilson

ARMS = ("off", "retry", "full")
ARM_LABEL = {"off": "no verifier", "retry": "retry only", "full": "full verifier"}
KINDS = ("transient", "silent")
KIND_LABEL = {"transient": "transient tool failures", "silent": "silently wrong results"}


def load_cells(dose_dir: Path | None = None) -> dict[tuple[str, float, str], dict]:
    dose_dir = dose_dir or RESULTS_DIR / "dose"
    cells = {}
    for p in sorted(dose_dir.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        kind = "clean" if r["fault_p"] == 0 and r.get("silent_p", 0) == 0 else (
            "transient" if r["fault_p"] > 0 else "silent")
        rate = r["fault_p"] or r.get("silent_p", 0.0)
        cells[(kind, round(rate, 2), r["arm"])] = r
    return cells


def _mean(rows: list[dict], key: str) -> float:
    return sum(x.get(key, 0) for x in rows) / max(1, len(rows))


def summarise(cell: dict) -> dict:
    rows = cell["rows"]
    k, n = sum(x["success"] for x in rows), len(rows)
    lo, hi = wilson(k, n)
    return {
        "k": k, "n": n, "rate": k / n if n else 0.0, "lo": lo, "hi": hi,
        "model_calls": _mean(rows, "model_calls"), "tool_calls": _mean(rows, "tool_calls"),
        "faults": _mean(rows, "faults_transient") + _mean(rows, "faults_silent"),
    }


def _cell_for(cells, kind: str, rate: float, arm: str):
    return cells.get(("clean", 0.0, arm)) if rate == 0 else cells.get((kind, round(rate, 2), arm))


def rates_for(cells) -> list[float]:
    return sorted({r for (_, r, _) in cells} | {0.0})


def paired(a: dict, b: dict) -> tuple[int, int, int, int]:
    """(both, only a, only b, neither) over the same (task, seed) pairs. At rate 0 the seeds repeat one run."""
    da = {(x["task_id"], x["seed"]): x["success"] for x in a["rows"]}
    db = {(x["task_id"], x["seed"]): x["success"] for x in b["rows"]}
    keys = da.keys() & db.keys()
    return (sum(da[k] and db[k] for k in keys), sum(da[k] and not db[k] for k in keys),
            sum(db[k] and not da[k] for k in keys), sum(not da[k] and not db[k] for k in keys))


def pct(s: dict) -> str:
    return f"{100 * s['rate']:.0f}% ({s['k']}/{s['n']}) [{100 * s['lo']:.0f}, {100 * s['hi']:.0f}]"


def build_markdown(cells) -> str:
    if not cells:
        return ""
    out = ["## Fault dose-response: no verifier, retry only, full verifier", "",
           "Success = task passes its checker. Intervals are 95% Wilson. Rate 0 is one clean run per arm (greedy "
           "decoding makes seeds identical); other rates use the seeds shown in `n`. Model calls and tool calls "
           "are means per task run.", ""]
    for kind in KINDS:
        out += [f"### {KIND_LABEL[kind]}", "",
                "| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |",
                "|---|---|---|---|---|---|"]
        for rate in rates_for(cells):
            for arm in ARMS:
                c = _cell_for(cells, kind, rate, arm)
                if c is None:
                    continue
                s = summarise(c)
                out.append(f"| {rate:.0%} | {ARM_LABEL[arm]} | {pct(s)} | {s['model_calls']:.1f} | "
                           f"{s['tool_calls']:.1f} | {s['faults']:.1f} |")
        out += ["", "Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). "
                "Seeds of one task are not independent, so treat the p-values as optimistic.", "",
                "| fault rate | comparison | only first passes | only second passes | exact p |", "|---|---|---|---|---|"]
        for rate in rates_for(cells):
            for a, b in (("retry", "off"), ("full", "retry"), ("full", "off")):
                ca, cb = _cell_for(cells, kind, rate, a), _cell_for(cells, kind, rate, b)
                if ca is None or cb is None:
                    continue
                _, oa, ob, _ = paired(ca, cb)
                out.append(f"| {rate:.0%} | {ARM_LABEL[a]} vs {ARM_LABEL[b]} | {oa} | {ob} | {mcnemar_exact(oa, ob):.4f} |")
        out.append("")
    return "\n".join(out) + "\n"


def results_json(cells) -> dict:
    data = []
    for (kind, rate, arm), c in sorted(cells.items()):
        s = summarise(c)
        data.append({"kind": kind, "fault_rate": rate, "arm": arm, **{k: round(v, 4) for k, v in s.items()}})
    meta = next(iter(cells.values())) if cells else {}
    return {"model": meta.get("model", ""), "device": meta.get("device", ""), "dtype": meta.get("dtype", ""),
            "cells": data}


COLORS = {"off": "#8a8d94", "retry": "#e9e9e4", "full": "#d7ff3f"}
DASH = {"off": "5 4", "retry": "", "full": ""}


def chart_svg(cells) -> str:
    """Two panels (transient, silent): success vs fault rate, one line per arm with Wilson bands."""
    W, H, PAD_L, PAD_T, PW, PH, GAP = 880, 340, 56, 44, 340, 230, 70
    rates = rates_for(cells)
    xmax = max(rates) or 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             'font-family="JetBrains Mono, monospace" font-size="11">',
             f'<rect width="{W}" height="{H}" fill="#0b0c0f" rx="10"/>']
    for i, kind in enumerate(KINDS):
        x0 = PAD_L + i * (PW + GAP)

        def X(r, x0=x0):
            return x0 + PW * r / xmax

        def Y(v):
            return PAD_T + PH * (1 - v)

        parts.append(f'<text x="{x0}" y="26" fill="#e9e9e4" font-size="13">{KIND_LABEL[kind]}</text>')
        for g in (0, 0.25, 0.5, 0.75, 1.0):
            parts.append(f'<line x1="{x0}" x2="{x0 + PW}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" stroke="#23252b"/>'
                         f'<text x="{x0 - 8}" y="{Y(g) + 4:.1f}" fill="#8a8d94" text-anchor="end">{g:.0%}</text>')
        for r in rates:
            parts.append(f'<text x="{X(r):.1f}" y="{PAD_T + PH + 18}" fill="#8a8d94" text-anchor="middle">{r:.0%}</text>')
        parts.append(f'<text x="{x0 + PW / 2}" y="{PAD_T + PH + 36}" fill="#8a8d94" text-anchor="middle">'
                     f'injected fault rate per tool call</text>')
        for arm in ARMS:
            pts = []
            for r in rates:
                c = _cell_for(cells, kind, r, arm)
                if c:
                    pts.append((r, summarise(c)))
            if len(pts) < 2:
                continue
            band = [(X(r), Y(s["hi"])) for r, s in pts] + [(X(r), Y(s["lo"])) for r, s in reversed(pts)]
            parts.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in band) +
                         f'" fill="{COLORS[arm]}" opacity="0.10"/>')
            parts.append('<polyline fill="none" stroke-width="2" stroke="' + COLORS[arm] + '"'
                         + (f' stroke-dasharray="{DASH[arm]}"' if DASH[arm] else "") + ' points="'
                         + " ".join(f"{X(r):.1f},{Y(s['rate']):.1f}" for r, s in pts) + '"/>')
            for r, s in pts:
                parts.append(f'<circle cx="{X(r):.1f}" cy="{Y(s["rate"]):.1f}" r="3.2" fill="{COLORS[arm]}"/>')
    lx = PAD_L
    for arm in ARMS:
        parts.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{H - 14}" y2="{H - 14}" stroke="{COLORS[arm]}" stroke-width="2"'
                     + (f' stroke-dasharray="{DASH[arm]}"' if DASH[arm] else "") + '/>'
                     f'<text x="{lx + 28}" y="{H - 10}" fill="#e9e9e4">{ARM_LABEL[arm]}</text>')
        lx += 170
    parts.append("</svg>")
    return "\n".join(parts)


def write_all(root: Path | None = None) -> list[Path]:
    root = root or Path(__file__).resolve().parents[3]
    cells = load_cells(root / "evals" / "results" / "dose")
    written = []
    if not cells:
        return written
    (root / "evals" / "results" / "DOSE.md").write_text(build_markdown(cells), encoding="utf-8")
    (root / "web" / "results.json").write_text(json.dumps(results_json(cells), indent=1), encoding="utf-8")
    assets = root / "docs" / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "dose.svg").write_text(chart_svg(cells), encoding="utf-8")
    return written
