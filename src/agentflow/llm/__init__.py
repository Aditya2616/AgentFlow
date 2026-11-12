"""Chat model backends. Every one is a LangChain BaseChatModel."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings

KINDS = ("local", "scripted", "openai", "anthropic")


def make_llm(
    kind: str | None = None,
    *,
    settings: Settings | None = None,
    scripts: dict | None = None,
    model: str | None = None,
    cache_dir: str | Path | None = None,
    dtype: str = "bfloat16",
) -> BaseChatModel:
    settings = settings or Settings()
    kind = kind or os.environ.get("AGENTFLOW_LLM", "local")
    if kind == "scripted":
        from .scripted import ScriptedChat

        return ScriptedChat(scripts=scripts or {})
    if kind == "openai":
        from .openai_chat import OpenAIChat

        return OpenAIChat.from_env()
    if kind == "anthropic":
        from .anthropic_chat import AnthropicChat

        return AnthropicChat.from_env()
    if kind == "local":
        from .local import LocalChat

        return LocalChat(
            model_name=model or settings.local_model, threads=settings.threads, dtype=dtype,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    raise ValueError(f"unknown LLM kind {kind!r}; choose one of {', '.join(KINDS)}")
