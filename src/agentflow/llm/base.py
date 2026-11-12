"""Shared helpers for the chat model backends. Every backend is a LangChain BaseChatModel."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


class AgentChatModel(BaseChatModel):
    """BaseChatModel with the standard bind_tools contract (OpenAI-style tool schemas)."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    @staticmethod
    def result(
        text: str, tool_calls: list[dict] | None = None, in_tok: int = 0, out_tok: int = 0
    ) -> ChatResult:
        msg = AIMessage(
            content=text,
            tool_calls=tool_calls or [],
            usage_metadata={"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def extract_json(text: str) -> Any | None:
    """Pull the first JSON object out of text, tolerating code fences and light damage."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    for loader in (json.loads, ast.literal_eval):
                        try:
                            return loader(chunk)
                        except (ValueError, SyntaxError):
                            pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse Qwen/Hermes style <tool_call>{...}</tool_call> blocks. Returns (remaining text, calls)."""
    calls: list[dict] = []
    for i, m in enumerate(re.finditer(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)):
        obj = extract_json(m.group(1))
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = extract_json(args) or {}
            if isinstance(args, dict):
                calls.append(make_tool_call(str(obj["name"]), args, f"call_{i}"))
    rest = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.S)
    rest = re.sub(r"<think>.*?</think>", "", rest, flags=re.S).strip()
    return rest, calls


def role_of(m: BaseMessage) -> str:
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(m.type, "user")


def text_of(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
