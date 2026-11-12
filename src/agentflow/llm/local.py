"""Local Hugging Face chat model (CUDA if available, else CPU). Qwen-style tool calling through the chat template."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr

from .base import AgentChatModel, parse_tool_calls, role_of, text_of

_LOAD_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str, str], tuple[Any, Any]] = {}


def _torch_cuda_available() -> bool:
    """True if torch is installed and sees a CUDA device. Torch is imported here, not at module import."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_runtime(
    device: str = "auto", dtype: str = "auto", cuda_available: Callable[[], bool] | None = None
) -> tuple[str, str]:
    """Pick a device and dtype. "auto" means cuda in float16 if CUDA is available, else cpu in bfloat16.

    The AGENTFLOW_DEVICE and AGENTFLOW_DTYPE environment variables override "auto". `cuda_available` is a probe
    that tests can inject, so this function needs no torch installed.
    """
    device = os.environ.get("AGENTFLOW_DEVICE", "") if device == "auto" else device
    dtype = os.environ.get("AGENTFLOW_DTYPE", "") if dtype == "auto" else dtype
    if not device or device == "auto":
        device = "cuda" if (cuda_available or _torch_cuda_available)() else "cpu"
    if not dtype or dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "bfloat16"
    return device, dtype


def _load(name: str, device: str, dtype: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = (name, device, dtype)
    with _LOAD_LOCK:
        if key not in _MODELS:
            torch.set_num_threads(threads)
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
            model.to(device)
            model.eval()
            _MODELS[key] = (tok, model)
        return _MODELS[key]


class LocalChat(AgentChatModel):
    """Greedy, deterministic generation. An optional on-disk cache makes repeated runs cheap."""

    model_name: str = "Qwen/Qwen3-0.6B"
    max_new_tokens: int = 160
    threads: int = 3
    device: str = "auto"
    dtype: str = "auto"
    cache_dir: str | None = None
    _hits: int = PrivateAttr(default=0)

    def runtime(self) -> dict[str, str]:
        """Device and dtype this model actually runs with (recorded with every evaluation result)."""
        device, dtype = resolve_runtime(self.device, self.dtype)
        return {"device": device, "dtype": dtype}

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

        device, dtype = resolve_runtime(self.device, self.dtype)
        tok, model = _load(self.model_name, device, dtype, self.threads)
        tools = kwargs.get("tools") or None
        max_new = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        chat = [{"role": role_of(m), "content": text_of(m)} for m in messages]
        extra = {"enable_thinking": False} if "Qwen3" in self.model_name else {}
        prompt = tok.apply_chat_template(chat, tools=tools, add_generation_prompt=True, tokenize=False, **extra)

        key = hashlib.sha256(json.dumps([self.model_name, device, dtype, prompt, max_new]).encode()).hexdigest()
        cache_file = Path(self.cache_dir) / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            hit = json.loads(cache_file.read_text(encoding="utf-8"))
            text, n_in, n_out = hit["text"], hit["in"], hit["out"]
            self._hits += 1
        else:
            ids = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                try:
                    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
                except torch.cuda.OutOfMemoryError:  # another process may hold the GPU: free ours and try once more
                    torch.cuda.empty_cache()
                    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
            n_in = int(ids["input_ids"].shape[1])
            new = out[0, n_in:].cpu()
            n_out = int(new.shape[0])
            text = tok.decode(new, skip_special_tokens=True)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({"text": text, "in": n_in, "out": n_out}), encoding="utf-8")
        content, calls = parse_tool_calls(text)
        return self.result(content, calls, n_in, n_out)
