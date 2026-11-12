"""OpenAI Chat Completions adapter over httpx.

Exercised in this repository only against a stubbed HTTP transport. It has not been run
against the live API.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import ConfigDict

from .base import AgentChatModel, extract_json, make_tool_call, role_of, text_of


class LLMHTTPError(RuntimeError):
    pass


def to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = role_of(m)
        if role == "assistant" and isinstance(m, AIMessage) and m.tool_calls:
            out.append({
                "role": "assistant", "content": text_of(m) or None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in m.tool_calls
                ],
            })
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": getattr(m, "tool_call_id", ""), "content": text_of(m)})
        else:
            out.append({"role": role, "content": text_of(m)})
    return out


class OpenAIChat(AgentChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "gpt-4.1-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 400
    timeout: float = 60.0
    transport: Any = None  # httpx transport, injected by tests

    @classmethod
    def from_env(cls, **kw: Any) -> OpenAIChat:
        return cls(
            model_name=os.environ.get("AGENTFLOW_OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            **kw,
        )

    @property
    def _llm_type(self) -> str:
        return "agentflow-openai"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.api_key:
            raise LLMHTTPError("OPENAI_API_KEY is not set")
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": to_openai_messages(messages),
            "max_tokens": int(kwargs.get("max_new_tokens", self.max_tokens)),
            "temperature": 0,
        }
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            body["tool_choice"] = "auto"
        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            r = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        if r.status_code != 200:
            raise LLMHTTPError(f"OpenAI request failed with HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            raw = c["function"].get("arguments") or "{}"
            args = extract_json(raw) if isinstance(raw, str) else raw
            calls.append(make_tool_call(c["function"]["name"], args or {}, c.get("id", f"call_{i}")))
        usage = data.get("usage", {})
        return self.result(
            msg.get("content") or "", calls, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
