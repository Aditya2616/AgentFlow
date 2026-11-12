"""Tracer: writes JSONL and notifies listeners (the server's SSE feed, tests)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .events import TraceEvent

MAX_ATTR_CHARS = 1500


def clip(value: Any, limit: int = MAX_ATTR_CHARS) -> Any:
    """Make a value JSON-safe and bounded so trace lines stay small."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"... [{len(value) - limit} more chars]"
    if isinstance(value, dict):
        return {str(k): clip(v, limit) for k, v in value.items()}
    if isinstance(value, list | tuple):
        out = [clip(v, limit) for v in value[:60]]
        if len(value) > 60:
            out.append(f"... [{len(value) - 60} more items]")
        return out
    if isinstance(value, int | float | bool) or value is None:
        return value
    return clip(str(value), limit)


class _Span:
    def __init__(self, span_id: str) -> None:
        self.id = span_id
        self.attrs: dict[str, Any] = {}
        self.status = "ok"

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def fail(self, **attrs: Any) -> None:
        self.status = "error"
        self.attrs.update(attrs)


class Tracer:
    def __init__(
        self,
        path: str | Path | None = None,
        listeners: list[Callable[[TraceEvent], None]] | None = None,
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self.path = Path(path) if path else None
        self.listeners = list(listeners or [])
        self.events: list[TraceEvent] = []
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._names: dict[str, str] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def _emit(self, ev: TraceEvent) -> None:
        with self._lock:
            self.events.append(ev)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(ev.model_dump_json() + "\n")
        for cb in self.listeners:
            cb(ev)

    def _now(self) -> float:
        return round(time.perf_counter() - self._t0, 4)

    def start(self, name: str, parent: str | None = None, **attributes: Any) -> str:
        span_id = uuid.uuid4().hex[:16]
        self._names[span_id] = name
        self._emit(
            TraceEvent(
                trace_id=self.trace_id, span_id=span_id, parent_span_id=parent, event="start",
                name=name, ts=time.time(), t=self._now(), attributes=clip(attributes),
            )
        )
        return span_id

    def end(self, span_id: str, status: str = "ok", **attributes: Any) -> None:
        self._emit(
            TraceEvent(
                trace_id=self.trace_id, span_id=span_id, event="end",
                name=self._names.get(span_id, ""), ts=time.time(), t=self._now(),
                attributes=clip(attributes), status=status,  # type: ignore[arg-type]
            )
        )

    @contextmanager
    def span(self, name: str, parent: str | None = None, **attributes: Any) -> Iterator[_Span]:
        sid = self.start(name, parent, **attributes)
        sp = _Span(sid)
        try:
            yield sp
        except Exception as exc:
            self.end(sid, "error", **{**sp.attrs, "error": f"{type(exc).__name__}: {exc}"})
            raise
        else:
            self.end(sid, sp.status, **sp.attrs)


def read_trace(path: str | Path) -> list[TraceEvent]:
    return [
        TraceEvent.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
