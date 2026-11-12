"""Local Hugging Face chat model (CPU). Qwen-style tool calling through the chat template."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr

from .base import AgentChatModel, parse_tool_calls, role_of, text_of

_LOAD_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}


def _load(name: str, dtype: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = (name, dtype)
    with _LOAD_LOCK:
        if key not in _MODELS:
            torch.set_num_threads(threads)
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
            model.eval()
            _MODELS[key] = (tok, model)
        return _MODELS[key]


class LocalChat(AgentChatModel):
    """Greedy, deterministic generation. An optional on-disk cache makes repeated runs cheap."""

    model_name: str = "Qwen/Qwen3-0.6B"
    max_new_tokens: int = 160
    threads: int = 3
    dtype: str = "bfloat16"
    cache_dir: str | None = None
    _hits: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "agentflow-local-hf"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        tok, model = _load(self.model_name, self.dtype, self.threads)
        tools = kwargs.get("tools") or None
        max_new = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        chat = [{"role": role_of(m), "content": text_of(m)} for m in messages]
        extra = {"enable_thinking": False} if "Qwen3" in self.model_name else {}
        prompt = tok.apply_chat_template(chat, tools=tools, add_generation_prompt=True, tokenize=False, **extra)

        key = hashlib.sha256(json.dumps([self.model_name, self.dtype, prompt, max_new]).encode()).hexdigest()
        cache_file = Path(self.cache_dir) / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            hit = json.loads(cache_file.read_text(encoding="utf-8"))
            text, n_in, n_out = hit["text"], hit["in"], hit["out"]
            self._hits += 1
        else:
            ids = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
            n_in = int(ids["input_ids"].shape[1])
            new = out[0, n_in:]
            n_out = int(new.shape[0])
            text = tok.decode(new, skip_special_tokens=True)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({"text": text, "in": n_in, "out": n_out}), encoding="utf-8")
        content, calls = parse_tool_calls(text)
        return self.result(content, calls, n_in, n_out)
