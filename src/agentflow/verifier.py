"""The verifier: checks each tool result against the step before the next step runs.

Checks are deterministic (no second model call). The verdict says what to do next:
  pass    the result looks right, advance
  retry   try the same step again (with the reason, or with the same arguments if the failure
          was a transient tool error)
  replan  retries are used up, ask the planner for a different approach
  fail    budgets are exhausted, stop and report honestly
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import AgentConfig
from .prompts import compact

Outcome = Literal["pass", "retry", "replan", "fail"]
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


class Check(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class Verdict(BaseModel):
    outcome: Outcome
    reason: str = ""
    checks: list[Check] = Field(default_factory=list)
    reuse_args: bool = False


def _numbers(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


COMMON = {"the", "this", "that", "with", "from", "for", "and", "all", "any", "one", "new", "our",
          "are", "not", "can", "has", "have", "will", "its", "your", "you", "per", "after", "before"}


def _grounded(body: str, prev_text: str) -> bool:
    """Does the email body mention at least one fact (a number or a name) from earlier results?"""
    prev_nums, body_nums = _numbers(prev_text), _numbers(body)
    if any(abs(a - b) <= 0.005 * max(1.0, abs(a)) for a in prev_nums for b in body_nums):
        return True
    names = {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", prev_text)} - COMMON
    return bool(names & {w.lower() for w in re.findall(r"[A-Za-z]{3,}", body)})


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def preflight(
    step: dict, args: dict, prev: dict[str, dict], task: str = "", send_guard: bool = True
) -> list[Check]:
    """Checks on the arguments of a side-effecting tool, run BEFORE it executes, so a rejected
    email is never written and a retry cannot create duplicates.

    Send guard: an email may only go to an address the user's request names. A model that
    invents a recipient (or emails something nobody asked for) is stopped before anything is written.
    """
    if step["tool"] != "email":
        return []
    guard: list[Check] = []
    if send_guard and task:
        allowed = {a.lower() for a in EMAIL.findall(task)}
        to = str(args.get("to", "")).strip().lower()
        guard = [Check(name="recipient_requested", ok=to in allowed,
                       detail="the request does not ask for an email to this address; only email addresses named in the request")]
        if not guard[0].ok:
            return guard
    done = [p for p in prev.values() if p.get("ok")]
    if not done:
        return guard
    prev_text = " ".join(json.dumps(compact(p["tool"], p), default=str) for p in done)
    ok = _grounded(args.get("body", ""), prev_text)
    return guard + [Check(name="grounded", ok=ok, detail="the email body does not mention any fact from the earlier results")]


def run_checks(step: dict, obs: dict, prev: dict[str, dict]) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name=name, ok=ok, detail="" if ok else detail))

    add("tool_ran", bool(obs["ok"]), str(obs.get("error") or ""))
    if not obs["ok"]:
        return checks
    v, tool = obs["value"], step["tool"]
    if tool == "search":
        hits = v["hits"]
        add("has_hits", bool(hits), "no document matched; try different keywords")
        if hits:
            add("relevant", hits[0]["score"] >= 0.8, f"best match is weak (score {hits[0]['score']})")
    elif tool == "sql":
        rows = v["rows"]
        add("has_rows", bool(rows), "the query returned no rows; check table names, columns and filter values")
        if rows:
            add("no_nulls", all(c is not None for r in rows[:3] for c in r),
                "the result contains NULL; the filter probably matched nothing")
    elif tool == "python":
        add("has_output", bool(v["stdout"].strip()), "the code printed nothing; print the final result")
    elif tool == "docs":
        add("has_text", bool((v.get("text") or v.get("summary") or "").strip()), "the document is empty")
    return checks


def decide(
    checks: list[Check], obs: dict, attempt: int, replans: int, cfg: AgentConfig
) -> Verdict:
    bad = [c for c in checks if not c.ok]
    if not bad:
        return Verdict(outcome="pass", reason="all checks passed", checks=checks)
    reason = "; ".join(c.detail or c.name for c in bad)
    if attempt < cfg.max_attempts:
        return Verdict(
            outcome="retry", reason=reason, checks=checks, reuse_args=obs.get("error_kind") == "transient"
        )
    if replans < cfg.max_replans:
        return Verdict(outcome="replan", reason=f"{reason} (after {attempt} attempts)", checks=checks)
    return Verdict(outcome="fail", reason=f"{reason} (retries and replans used up)", checks=checks)


def verify(
    step: dict, obs: dict, prev: dict[str, dict], attempt: int, replans: int, cfg: AgentConfig
) -> tuple[Verdict, Any]:
    """Returns (verdict, would_flag). With the verifier disabled the verdict is always pass,
    but the checks still run so traces show what the verifier would have caught."""
    if cfg.verifier and cfg.retry_only:
        # the retry-only arm: a tool that raised is retried with the error text, nothing else is checked,
        # and there is no re-plan, so once the retries are used up the run fails
        checks = [Check(name="tool_ran", ok=bool(obs["ok"]), detail=str(obs.get("error") or ""))]
        return decide(checks, obs, attempt, cfg.max_replans, cfg), None
    checks = run_checks(step, obs, prev)
    real = decide(checks, obs, attempt, replans, cfg)
    if cfg.verifier:
        return real, None
    forced = Verdict(outcome="pass", reason="verifier disabled", checks=checks)
    return forced, real.outcome != "pass"
