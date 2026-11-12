"""A deterministic chat model for tests and for the mechanics demo.

It replays a Script per task. It never calls a network or a model, and it can be told to make
deliberate mistakes (a wrong first argument, a wrong first plan) so the verifier's retry and
re-plan paths can be exercised repeatably. Results from it describe the loop, not a model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr

from .base import AgentChatModel, make_tool_call, text_of

ArgsSpec = dict | Callable[[dict], dict]


@dataclass
class Script:
    plan: list[tuple[str, str]]  # (tool, goal)
    args: dict[str, list[ArgsSpec]] = field(default_factory=dict)  # goal -> args per attempt
    replan: list[tuple[str, str]] | None = None
    answer: Callable[[dict], str] | None = None  # results -> final answer; default: last result


class ScriptError(RuntimeError):
    pass


def _grab(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.M)
    return m.group(1).strip() if m else ""


class ScriptedChat(AgentChatModel):
    model_name: str = "scripted"
    scripts: dict[str, Any] = {}
    _attempts: dict = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "agentflow-scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = text_of(messages[0])
        user = text_of(messages[-1]) if len(messages) == 2 else text_of(messages[1])
        task = _grab(r"^Task: (.*)$", user)
        script: Script | None = self.scripts.get(task)
        if script is None:
            raise ScriptError(f"no script for task {task!r}")
        phase = _grab(r"\[phase:(\w+)", system)
        if phase == "plan":
            replan = int(_grab(r"replan=(\d+)", system) or 0)
            if not replan:  # a fresh run of this task: start its attempt counters over
                self._attempts = {k: v for k, v in self._attempts.items() if k[0] != task}
            steps = script.replan if (replan and script.replan) else script.plan
            body = {"steps": [{"goal": g, "tool": t} for t, g in steps]}
            return self.result(json.dumps(body), [], 40, 40)
        if phase == "act":
            goal = _grab(r"^Step: (.*)$", user)
            prev = json.loads(_grab(r"^Previous results: (.*)$", user) or "{}")
            specs = script.args.get(goal)
            if not specs:
                raise ScriptError(f"no scripted arguments for step {goal!r}")
            n = self._attempts.get((task, goal), 0)
            self._attempts[(task, goal)] = n + 1
            spec = specs[min(n, len(specs) - 1)]
            try:
                args = spec(prev) if callable(spec) else dict(spec)
            except (TypeError, KeyError, IndexError, ValueError):
                args = {}  # the script could not build arguments from a bad earlier result
            tools = kwargs.get("tools") or []
            name = tools[0]["function"]["name"] if tools else "unknown"
            return self.result("", [make_tool_call(name, args, f"scripted_{n}")], 30, 20)
        if phase == "finish":
            results = json.loads(_grab(r"^Results: (.*)$", user) or "{}")
            if script.answer:
                text = script.answer(results)
            else:
                last = list(results.values())[-1] if results else ""
                text = last if isinstance(last, str) else json.dumps(last)
            return self.result(f"Answer: {text}", [], 30, 20)
        raise ScriptError(f"unknown phase {phase!r}")
