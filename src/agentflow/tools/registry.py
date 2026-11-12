"""The five tools as LangChain StructuredTools, plus one uniform way to execute them."""

from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from .docs import DocError, run_docs
from .email_tool import EmailError, run_email
from .python_sandbox import SandboxError, run_python
from .search import CorpusIndex
from .sql import SqlError, run_sql

TOOL_NAMES = ("search", "python", "sql", "email", "docs")


class SearchArgs(BaseModel):
    query: str = Field(description="Keywords to look for in the company documents")


class PythonArgs(BaseModel):
    code: str = Field(description="Python code. Use print() to show the final result")


class SqlArgs(BaseModel):
    query: str = Field(description="One read-only SQLite SELECT statement")


class EmailArgs(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Short subject line")
    body: str = Field(description="Plain text body")


class DocsArgs(BaseModel):
    filename: str = Field(description="File name of a document, for example refund_policy.md")
    action: Literal["read", "summarise"] = Field(default="read", description="read or summarise")


TOOL_HELP = {
    "search": "look up facts in the company documents (policies, products, reports, meeting notes)",
    "python": "calculate or process data with Python code",
    "sql": "query the sales database (customers, orders, products, regions)",
    "email": "write an email (goes to a local outbox)",
    "docs": "read or summarise one named document file",
}


@dataclass
class ToolOutcome:
    ok: bool
    value: Any = None
    error: str | None = None
    error_kind: str | None = None  # bad_args | tool_error | blocked | timeout | transient | no_call
    latency_ms: int = 0
    fault: str | None = None  # "transient" or "silent:<what>" when the evaluation injected a fault


class FaultInjector:
    """Seeded faults for the evaluation.

    transient: the tool call raises a timeout (a plain retry fixes it).
    silent: the tool succeeds but returns a well-formed, wrong result: no search hits, a SQL result
        with zero rows, empty Python output, or a document read cut to its first 40 characters.
        Email is never perturbed. Only checking the result can notice a silent fault. The docs tool states the
        document's length (`chars`), so the verifier can see that a read was cut short.
    """

    def __init__(self, probability: float = 0.0, seed: int = 0, silent: float = 0.0) -> None:
        self.p, self.silent_p = probability, silent
        self.rng = random.Random(seed)
        self.srng = random.Random(seed + 7919)
        self.counts = {"transient": 0, "silent": 0}

    def hit(self) -> bool:
        hit = self.p > 0 and self.rng.random() < self.p
        self.counts["transient"] += hit
        return hit

    def perturb(self, name: str, value: Any) -> tuple[Any, str | None]:
        if name == "email" or not (self.silent_p > 0 and self.srng.random() < self.silent_p):
            return value, None
        v = dict(value) if isinstance(value, dict) else value
        if name == "search":
            v["hits"] = []
        elif name == "sql":
            v["rows"] = []
        elif name == "python":
            v["stdout"] = ""
        elif name == "docs" and isinstance(v.get("text"), str):
            v["text"] = v["text"][:40]  # "chars" still states the real length, as the tool's contract says
        else:
            return value, None
        self.counts["silent"] += 1
        return v, f"silent:{name}"


class ToolBox:
    def __init__(self, settings: Settings | None = None, faults: FaultInjector | None = None) -> None:
        self.settings = settings or Settings()
        self.faults = faults
        self._index: CorpusIndex | None = None
        self.tools: dict[str, StructuredTool] = {
            "search": self._tool("search", self._search, SearchArgs),
            "python": self._tool("python", self._python, PythonArgs),
            "sql": self._tool("sql", self._sql, SqlArgs),
            "email": self._tool("email", self._email, EmailArgs),
            "docs": self._tool("docs", self._docs, DocsArgs),
        }

    @staticmethod
    def _tool(name: str, fn, schema) -> StructuredTool:
        return StructuredTool.from_function(func=fn, name=name, description=TOOL_HELP[name], args_schema=schema)

    # tool bodies -----------------------------------------------------------------
    def _search(self, query: str) -> dict:
        if self._index is None:
            self._index = CorpusIndex(self.settings.corpus_dir, self.settings.search_mode)
        return {"hits": self._index.search(query, k=3)}

    def _python(self, code: str) -> dict:
        return run_python(code)

    def _sql(self, query: str) -> dict:
        return run_sql(self.settings.ensure_db(), query)

    def _email(self, to: str, subject: str, body: str) -> dict:
        s = self.settings
        return run_email(
            s.outbox_dir, to, subject, body,
            smtp_url=s.smtp_url, allow_send=s.allow_send, allowed_domains=s.email_domains,
        )

    def _docs(self, filename: str, action: str = "read") -> dict:
        return run_docs(self.settings.corpus_dir, filename, action)

    # uniform execution -------------------------------------------------------------
    def execute(self, name: str, args: dict) -> ToolOutcome:
        started = time.perf_counter()

        def done(**kw) -> ToolOutcome:
            return ToolOutcome(latency_ms=int((time.perf_counter() - started) * 1000), **kw)

        tool = self.tools.get(name)
        if tool is None:
            return done(ok=False, error=f"unknown tool {name!r}", error_kind="bad_args")
        if self.faults and self.faults.hit():
            return done(ok=False, error="TransientError: tool backend timed out (injected fault)",
                        error_kind="transient", fault="transient")
        try:
            value = tool.invoke(args)
            fault = None
            if self.faults:
                value, fault = self.faults.perturb(name, value)
            return done(ok=True, value=value, fault=fault)
        except ValidationError as exc:
            fields = ", ".join(str(e["loc"][0]) for e in exc.errors() if e.get("loc"))
            return done(ok=False, error=f"invalid arguments for {name}: check {fields}", error_kind="bad_args")
        except SandboxError as exc:
            return done(ok=False, error=str(exc), error_kind=exc.kind if exc.kind in ("blocked", "timeout") else "tool_error")
        except (SqlError, DocError, EmailError) as exc:
            return done(ok=False, error=str(exc), error_kind="tool_error")
        except (sqlite3.Error, OSError) as exc:
            return done(ok=False, error=f"{type(exc).__name__}: {exc}", error_kind="tool_error")
