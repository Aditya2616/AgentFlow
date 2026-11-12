"""Trace event model. One JSON line per event; spans have a start and an end line.

Attribute names follow the OpenTelemetry GenAI conventions where one exists
(gen_ai.operation.name, gen_ai.tool.name, ...). Everything else lives under agentflow.*
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    event: Literal["start", "end"]
    name: str
    ts: float = Field(description="Unix time in seconds")
    t: float = Field(description="Seconds since the run started")
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "unset"] = "unset"
