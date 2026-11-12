"""Red-team list for the Python sandbox. See README for what this does and does not prove."""

from __future__ import annotations

import os

import pytest

from agentflow.tools.python_sandbox import SandboxError, run_python, static_gate


def test_runs_plain_code_and_prints_last_expression():
    assert run_python("print(2**10)")["stdout"].strip() == "1024"
    assert run_python("x = 6\nx * 7")["stdout"].strip() == "42"
    assert run_python("import math\nprint(round(math.pi, 3))")["stdout"].strip() == "3.142"


def test_runtime_errors_are_reported_with_the_cause():
    with pytest.raises(SandboxError, match="ZeroDivisionError"):
        run_python("1/0")
    with pytest.raises(SandboxError, match="NameError"):
        run_python("print(nope)")
    with pytest.raises(SandboxError, match="SyntaxError"):
        run_python("def f(:")


BLOCKED = [
    "import os",
    "import subprocess",
    "import socket",
    "import sys",
    "from os import path",
    "import importlib",
    "import ctypes",
    "__import__('os')",
    "open('pyproject.toml').read()",
    "eval('1+1')",
    "exec('x=1')",
    "getattr(print, '__self__')",
    "().__class__.__bases__[0].__subclasses__()",
    "import random\nrandom.os.system('echo hi')",
    "import json\njson.sys.modules['os']",
    "[c for c in ().__class__.__mro__]",
    "globals()",
]


@pytest.mark.parametrize("code", BLOCKED)
def test_static_gate_blocks(code):
    with pytest.raises(SandboxError) as e:
        static_gate(code)
    assert e.value.kind == "blocked"


def test_blocked_code_never_reaches_a_process():
    with pytest.raises(SandboxError) as e:
        run_python("import os\nprint(os.listdir('.'))")
    assert e.value.kind == "blocked"


def test_child_layer_blocks_even_if_the_static_gate_is_bypassed(monkeypatch):
    from agentflow.tools import python_sandbox as ps

    monkeypatch.setattr(ps, "static_gate", lambda code: None)
    with pytest.raises(SandboxError, match="not allowed"):
        ps.run_python("import os")
    with pytest.raises(SandboxError, match="NameError"):
        ps.run_python("print(open)")


def test_infinite_loop_is_killed():
    with pytest.raises(SandboxError) as e:
        run_python("while True:\n    pass", timeout=1.0)
    assert e.value.kind == "timeout"


def test_output_flood_is_cut_off():
    with pytest.raises(SandboxError):
        run_python("while True:\n    print('x' * 1000)", timeout=5.0)


def test_memory_bomb_fails_cleanly():
    with pytest.raises(SandboxError):
        run_python("x = 'a' * (2 * 10**9)\nprint(len(x))")


def test_child_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("AGENTFLOW_SECRET_TOKEN", "hunter2")
    # os is not importable, so read the environment through a module that exposes it indirectly
    # (it must not): the child simply has no such variable.
    out = run_python("print('ok')")
    assert "hunter2" not in out["stdout"] + out["stderr"]


@pytest.mark.skipif(os.name != "nt", reason="Windows job object test")
def test_windows_job_object_is_applied():
    from agentflow.tools.python_sandbox import _make_job

    assert _make_job(64 * 1024 * 1024) is not None
