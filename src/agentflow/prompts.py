"""Prompt construction. Prompts are short on purpose: a 0.6B model on CPU pays for every token.

Each system prompt starts with a [phase:...] marker. Real models ignore it; the scripted
model uses it to know which part of the loop is calling it.
"""

from __future__ import annotations

import json
from typing import Any

from .data.build_db import SCHEMA_HINT
from .tools.registry import TOOL_HELP

PLAN_EXAMPLES = (
    'Request: "What is the battery life of the Zeta watch?"\n'
    '{"steps":[{"goal":"find the Zeta watch battery life","tool":"search"}]}\n'
    'Request: "How many customers are in the East region? Email the number to a@b.com"\n'
    '{"steps":[{"goal":"count customers in the East region","tool":"sql"},'
    '{"goal":"email the count to a@b.com","tool":"email"}]}'
)


def plan_system(files: list[str], replan: int = 0) -> str:
    tools = "\n".join(f"- {n}: {d}" for n, d in TOOL_HELP.items())
    return (
        f"[phase:plan replan={replan}]\n"
        "You are a planner. Split the request into the fewest steps that answer it (1 to 4). "
        "Each step uses exactly one tool. Only add an email step if the request asks for an email.\n"
        f"Tools:\n{tools}\n"
        f"Document files: {', '.join(files)}\n"
        "Use sql for questions about orders, customers, products or regions. "
        "Use search for facts from policies and notes. Use docs only when a file name is given. "
        "Use python for arithmetic.\n"
        f"Reply with JSON only. Examples:\n{PLAN_EXAMPLES}"
    )


def plan_user(task: str, done: list[dict] | None = None, failed: dict | None = None, reason: str = "") -> str:
    out = f"Task: {task}"
    if done:
        out += "\nAlready done: " + "; ".join(f"{d['goal']} -> {d['summary']}" for d in done)
    if failed:
        out += (
            f"\nThis step failed: {failed['goal']} (tool {failed['tool']}). Reason: {reason}"
            "\nPlan the remaining steps, using a different approach."
        )
    return out


ACT_HINTS = {
    "sql": SCHEMA_HINT + " Use SQLite syntax and write one SELECT statement.",
    "python": "Write short Python. Print the final answer with print().",
    "search": "Use 2 to 5 keywords that would appear in the document.",
    "email": "Write a clear subject and a body that states the concrete facts from the previous results.",
    "docs": "Pass the exact file name. Action is read or summarise.",
}


def act_system(step_id: str, tool: str, files: list[str]) -> str:
    extra = f" Files: {', '.join(files)}." if tool == "docs" else ""
    return (
        f"[phase:act step={step_id}]\n"
        f"You must respond by calling the {tool} tool to do the current step. Use facts and numbers "
        f"from the previous results. Do not invent values. {ACT_HINTS[tool]}{extra}"
    )


def act_user(
    task: str, goal: str, prev: dict[str, Any], retry_reason: str | None, attempt: int = 1
) -> str:
    out = f"Task: {task}\nStep: {goal}\nPrevious results: {json.dumps(prev, ensure_ascii=False)}"
    if retry_reason:
        out += (
            f"\nAttempt {attempt}. The last attempt failed: {retry_reason}. "
            "Call the tool again with corrected arguments."
        )
    return out


def finish_system() -> str:
    return (
        "[phase:finish]\n"
        "Answer the request in one or two plain sentences, using only the results below. "
        "If a result is an error, say what failed instead of guessing."
    )


def finish_user(task: str, results: dict[str, Any]) -> str:
    return f"Task: {task}\nResults: {json.dumps(results, ensure_ascii=False)}"


def compact(tool: str, obs: dict) -> Any:
    """A short, JSON-friendly view of an observation for prompts and the trace."""
    if not obs.get("ok"):
        return f"ERROR: {obs.get('error')}"
    v = obs["value"]
    if tool == "search":
        return [{"doc": h["doc"], "text": h["text"][:220]} for h in v["hits"]]
    if tool == "sql":
        return {"columns": v["columns"], "rows": v["rows"][:8]}
    if tool == "python":
        return v["stdout"].strip()[:300]
    if tool == "docs":
        return (v.get("text") or v.get("summary") or "")[:600]
    if tool == "email":
        return f"email to {v['to']} with subject {v['subject']!r} written to the outbox"
    return v
