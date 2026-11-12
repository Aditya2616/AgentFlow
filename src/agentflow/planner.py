"""The planner: turns a request into a short list of single-tool steps."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from .llm.base import extract_json
from .prompts import plan_system, plan_user
from .tools.registry import TOOL_NAMES

ALIASES = {
    "sql": "sql", "database": "sql", "db": "sql", "python": "python", "calculator": "python",
    "code": "python", "search": "search", "lookup": "search", "email": "email", "mail": "email",
    "docs": "docs", "document": "docs", "documents": "docs", "read": "docs", "read_doc": "docs",
}

ChatFn = Callable[..., AIMessage]


class PlanStep(BaseModel):
    goal: str
    tool: str


def parse_plan(text: str, max_steps: int) -> list[PlanStep]:
    obj: Any = extract_json(text)
    raw = obj.get("steps") if isinstance(obj, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected a JSON object with a non-empty 'steps' list")
    steps = []
    for item in raw[:max_steps]:
        try:
            step = PlanStep.model_validate(item)
        except ValidationError as exc:
            raise ValueError(f"each step needs a 'goal' and a 'tool': {exc.errors()[0]['msg']}") from exc
        tool = ALIASES.get(step.tool.strip().lower())
        if tool not in TOOL_NAMES:
            raise ValueError(f"unknown tool {step.tool!r}; use one of {', '.join(TOOL_NAMES)}")
        steps.append(PlanStep(goal=step.goal.strip(), tool=tool))
    return steps


def fallback_plan(task: str) -> list[PlanStep]:
    """Keyword router used only when the model cannot produce a valid plan twice."""
    t = task.lower()
    if re.search(r"\b(orders?|customers?|revenue|regions?|products?|sold|sales|database)\b", t):
        tool = "sql"
    elif re.search(r"\b\d+\s*[-+*/^]\s*\d+|compute|calculate|prime|fibonacci\b", t):
        tool = "python"
    elif re.search(r"\.md\b|\bread\b", t):
        tool = "docs"
    elif "email" in t:
        tool = "email"
    else:
        tool = "search"
    return [PlanStep(goal=task, tool=tool)]


def make_plan(
    chat: ChatFn,
    task: str,
    files: list[str],
    max_steps: int,
    *,
    replan: int = 0,
    done: list[dict] | None = None,
    failed: dict | None = None,
    reason: str = "",
) -> tuple[list[PlanStep], bool]:
    """Returns (steps, used_fallback). One repair attempt before falling back."""
    messages: list = [
        SystemMessage(content=plan_system(files, replan)),
        HumanMessage(content=plan_user(task, done, failed, reason)),
    ]
    error = ""
    for _ in range(2):
        ai = chat(messages, "plan", max_new_tokens=140)
        try:
            return parse_plan(str(ai.content), max_steps), False
        except ValueError as exc:
            error = str(exc)
            messages = messages + [
                AIMessage(content=str(ai.content)),
                HumanMessage(content=f"That was not valid: {error}. Reply with the JSON object only."),
            ]
    return fallback_plan(task), True
