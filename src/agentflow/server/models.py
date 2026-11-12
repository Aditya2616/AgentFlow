"""Typed request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(min_length=3, max_length=500)
    llm: Literal["local", "local-large", "scripted", "openai", "anthropic"] = "local"
    verifier: bool = True
    faults: float = Field(default=0.0, ge=0.0, le=0.6, description="probability of an injected tool failure")
    seed: int = 0
    mistake: Literal["none", "bad_args", "bad_plan"] = Field(
        default="none", description="scripted model only: make one deliberate first mistake"
    )


class RunInfo(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed", "error"]
    task: str
    llm: str
    verifier: bool
    faults: float
    answer: str | None = None
    events: int = 0
    created: float


class TaskInfo(BaseModel):
    id: str
    band: str
    title: str
    prompt: str


class ToolInfo(BaseModel):
    name: str
    description: str
    schema_: dict = Field(alias="schema")


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    default_llm: str
    local_model: str
    large_model: str
    remote_tasks_allowed: bool
