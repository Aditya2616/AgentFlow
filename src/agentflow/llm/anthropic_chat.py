"""Anthropic Messages API adapter over httpx.

Exercised in this repository only against a stubbed HTTP transport. It has not been run
against the live API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import ConfigDict

from .base import AgentChatModel, make_tool_call, require_model, role_of, text_of
from .openai_chat import LLMHTTPError


def to_anthropic(messages: list[BaseMessage]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = role_of(m)
        if role == "system":
            system_parts.append(text_of(m))
        elif role == "assistant" and isinstance(m, AIMessage) and m.tool_calls:
            blocks: list[dict] = [{"type": "text", "text": text_of(m)}] if text_of(m) else []
            blocks += [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]} for c in m.tool_calls
            ]
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": getattr(m, "tool_call_id", ""), "content": text_of(m)}
            ]})
        else:
            out.append({"role": "assistant" if role == "assistant" else "user", "content": text_of(m)})
    return "\n\n".join(system_parts), out


class AnthropicChat(AgentChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = ""  # no default: set ANTHROPIC_MODEL
    api_key: str = ""
    base_url: str = "https://api.anthropic.com"
    max_tokens: int = 400
    timeout: float = 60.0
    transport: Any = None

    @classmethod
    def from_env(cls, **kw: Any) -> AnthropicChat:
        return cls(
            model_name=kw.pop("model_name", None) or os.environ.get("ANTHROPIC_MODEL", ""),
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            **kw,
        )

    @property
    def _llm_type(self) -> str:
        return "agentflow-anthropic"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.api_key:
            raise LLMHTTPError("ANTHROPIC_API_KEY is not set")
        system, msgs = to_anthropic(messages)
        body: dict[str, Any] = {
            "model": require_model(self.model_name, "ANTHROPIC_MODEL"),
            "max_tokens": int(kwargs.get("max_new_tokens", self.max_tokens)),
            "messages": msgs,
        }
        if system:
            body["system"] = system
        if kwargs.get("tools"):  # OpenAI-style schemas from bind_tools -> Anthropic tool format
            body["tools"] = [
                {"name": t["function"]["name"], "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})}
                for t in kwargs["tools"]
            ]
        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            r = client.post(
                f"{self.base_url.rstrip('/')}/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json=body,
            )
        if r.status_code != 200:
            raise LLMHTTPError(f"Anthropic request failed with HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        text, calls = "", []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                calls.append(make_tool_call(block["name"], block.get("input", {}), block["id"]))
        usage = data.get("usage", {})
        return self.result(text, calls, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
