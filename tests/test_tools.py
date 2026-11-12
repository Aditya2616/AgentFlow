from __future__ import annotations

import json
import smtplib
import socket
import sqlite3

import pytest

from agentflow.tools.docs import DocError, run_docs, summarise
from agentflow.tools.email_tool import EmailError, run_email
from agentflow.tools.registry import FaultInjector, ToolBox
from agentflow.tools.search import BM25, CorpusIndex, tokenize
from agentflow.tools.sql import SqlError, run_sql


# ------------------------------------------------------------------ SQL
def test_sql_select_works(db_path):
    out = run_sql(db_path, "SELECT COUNT(*) FROM orders")
    assert out["rows"][0][0] == 640


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM orders",
        "UPDATE orders SET status='x'",
        "DROP TABLE orders",
        "INSERT INTO regions VALUES (9, 'Moon')",
        "CREATE TABLE evil (a)",
        "ATTACH DATABASE ':memory:' AS m",
        "PRAGMA writable_schema = ON",
        "WITH x AS (SELECT 1) DELETE FROM orders",
    ],
)
def test_sql_refuses_writes_at_sqlite_level(db_path, query):
    with pytest.raises(SqlError):
        run_sql(db_path, query)
    assert run_sql(db_path, "SELECT COUNT(*) FROM orders")["rows"][0][0] == 640


def test_sql_rejects_multiple_statements(db_path):
    with pytest.raises(SqlError):
        run_sql(db_path, "SELECT 1; DELETE FROM orders")


def test_sql_file_handle_is_read_only(db_path):
    # even if the authorizer were bypassed, the connection itself is mode=ro
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM orders")
    con.close()


def test_sql_row_cap_and_timeout(db_path):
    out = run_sql(db_path, "SELECT * FROM order_items")
    assert len(out["rows"]) == 200 and out["truncated"]
    with pytest.raises(SqlError, match="too long"):
        run_sql(
            db_path,
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c",
        )


# ------------------------------------------------------------------ search
def test_bm25_prefers_matching_document():
    docs = [tokenize("refund window thirty days"), tokenize("password length policy"), tokenize("lunch menu")]
    scores = BM25(docs).scores(tokenize("how long is the refund window"))
    assert scores.index(max(scores)) == 0 and scores[2] == 0


def test_corpus_search_finds_facts(settings):
    idx = CorpusIndex(settings.corpus_dir)
    assert idx.search("refund window physical goods")[0]["doc"] == "refund_policy.md"
    assert idx.search("minimum password length")[0]["doc"] == "security_policy.md"
    assert "4 hours" in idx.search("priority 1 first response time")[0]["text"]
    assert idx.search("zzzqq") == []


# ------------------------------------------------------------------ docs
def test_docs_read_and_summarise(settings):
    out = run_docs(settings.corpus_dir, "onboarding.md")
    assert "7 days" in out["text"]
    assert len(run_docs(settings.corpus_dir, "onboarding.md", "summarise")["summary"]) > 20


@pytest.mark.parametrize("name", ["../pyproject.toml", "..\\x", "/etc/passwd", ".hidden", "nope.md", "a/b.md"])
def test_docs_path_jail(settings, name):
    with pytest.raises(DocError):
        run_docs(settings.corpus_dir, name)


def test_summarise_short_text_unchanged():
    assert "a short sentence that is long enough." in summarise("# T\na short sentence that is long enough.")


# ------------------------------------------------------------------ email
def test_email_writes_outbox_only_by_default(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network was touched")

    monkeypatch.setattr(smtplib.SMTP, "__init__", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    out = run_email(tmp_path, "a@example.com", "Hello", "Body text")
    assert out["sent_over_smtp"] is False
    assert list(tmp_path.glob("*.eml"))
    entry = json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])
    assert entry["to"] == "a@example.com" and entry["sent"] is False


def test_email_smtp_url_alone_does_not_send(tmp_path, monkeypatch):
    monkeypatch.setattr(smtplib.SMTP, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sent")))
    out = run_email(tmp_path, "a@example.com", "Hi", "Body", smtp_url="smtp://u:p@mail.example.com:587")
    assert out["sent_over_smtp"] is False


def test_email_sends_only_with_url_and_flag(tmp_path, monkeypatch):
    calls = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            calls.append((host, port))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            calls.append("tls")

        def login(self, u, p):
            calls.append(("login", u))

        def send_message(self, msg):
            calls.append(("send", msg["To"]))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    out = run_email(tmp_path, "a@example.com", "Hi", "Body", smtp_url="smtp://u:p@mail.example.com:2525", allow_send=True)
    assert out["sent_over_smtp"] is True
    assert ("send", "a@example.com") in calls and ("mail.example.com", 2525) in calls


@pytest.mark.parametrize(
    "to,subject,body",
    [("nope", "s", "b"), ("a@b", "s", "b"), ("a@example.com", "", "b"), ("a@example.com", "s", "  ")],
)
def test_email_validation(tmp_path, to, subject, body):
    with pytest.raises(EmailError):
        run_email(tmp_path, to, subject, body)
    assert not list(tmp_path.glob("*.eml"))


def test_email_domain_allow_list(tmp_path):
    with pytest.raises(EmailError, match="allow-list"):
        run_email(tmp_path, "a@evil.com", "s", "b", allowed_domains=["example.com"])


# ------------------------------------------------------------------ toolbox
def test_toolbox_uniform_errors(settings):
    tb = ToolBox(settings)
    assert tb.execute("sql", {"query": "SELECT 1"}).ok
    bad = tb.execute("sql", {})
    assert not bad.ok and bad.error_kind == "bad_args"
    assert tb.execute("nope", {}).error_kind == "bad_args"
    assert tb.execute("docs", {"filename": "missing.md"}).error_kind == "tool_error"


def test_toolbox_fault_injection_is_deterministic(settings):
    a = ToolBox(settings, FaultInjector(0.5, seed=3))
    b = ToolBox(settings, FaultInjector(0.5, seed=3))
    ra = [a.execute("sql", {"query": "SELECT 1"}).error_kind for _ in range(12)]
    rb = [b.execute("sql", {"query": "SELECT 1"}).error_kind for _ in range(12)]
    assert ra == rb and "transient" in ra and None in ra


def test_cut_off_document_read_is_caught_by_the_completeness_check(settings):
    from agentflow.tools.registry import FaultInjector, ToolBox
    from agentflow.verifier import run_checks

    step = {"id": "s1", "tool": "docs", "goal": "read"}
    clean = ToolBox(settings).execute("docs", {"filename": "onboarding.md", "action": "read"})
    assert all(c.ok for c in run_checks(step, {"ok": True, "value": clean.value}, {}))
    cut = ToolBox(settings, FaultInjector(0.0, 0, silent=1.0)).execute("docs", {"filename": "onboarding.md", "action": "read"})
    assert cut.fault == "silent:docs" and len(cut.value["text"]) == 40 and cut.value["chars"] == clean.value["chars"]
    bad = [c for c in run_checks(step, {"ok": True, "value": cut.value}, {}) if not c.ok]
    assert [c.name for c in bad] == ["complete"] and "cut short" in bad[0].detail
    # a summary has no length contract and is left alone
    s = ToolBox(settings, FaultInjector(0.0, 0, silent=1.0)).execute("docs", {"filename": "onboarding.md", "action": "summarise"})
    assert s.fault is None
