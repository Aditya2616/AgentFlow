"""Python execution tool with layered, best-effort isolation.

THIS IS NOT A SECURITY BOUNDARY. See README "Sandbox: what it does and does not protect".

Layers
  1. Static gate (AST): import allow-list, no dunder or underscore attribute access, no
     exec/eval/open/getattr and friends, no access to attributes like .sys / .os / .modules.
  2. Separate interpreter process: `python -I -S`, scrubbed environment, empty temp working
     directory, stdin closed, builtins narrowed, imports from user code filtered.
  3. Resource limits: wall-clock timeout with whole-tree kill, output cap, and
       Windows: a Job Object (process memory cap, one active process, kill-on-close)
       Linux:  setrlimit for address space, CPU seconds and file size.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

CHILD = Path(__file__).with_name("_sandbox_child.py")
ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "itertools", "collections", "datetime", "decimal",
    "fractions", "random", "string", "textwrap", "functools", "operator", "heapq", "bisect",
    "typing", "dataclasses", "enum", "time", "calendar", "cmath", "numbers",
}
BANNED_NAMES = {
    "exec", "eval", "compile", "open", "input", "breakpoint", "exit", "quit", "help",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "__import__",
}
BANNED_ATTRS = {
    "sys", "os", "modules", "system", "popen", "spawn", "fork", "exec", "subprocess", "builtins",
    "importlib", "ctypes", "socket", "loader", "gi_frame", "f_back", "f_globals", "tb_frame",
}
MEMORY_LIMIT = 256 * 1024 * 1024
OUTPUT_CAP = 64 * 1024


class SandboxError(Exception):
    def __init__(self, message: str, kind: str = "runtime") -> None:
        super().__init__(message)
        self.kind = kind


def static_gate(code: str) -> None:
    """Raise SandboxError(kind='blocked') if the code uses anything outside the allowed subset."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"SyntaxError: {exc.msg} (line {exc.lineno})", "syntax") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_MODULES:
                    raise SandboxError(f"import of {a.name!r} is not allowed in the sandbox", "blocked")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise SandboxError(f"import from {node.module!r} is not allowed in the sandbox", "blocked")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BANNED_ATTRS:
                raise SandboxError(f"access to attribute {node.attr!r} is not allowed", "blocked")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                raise SandboxError(f"use of {node.id!r} is not allowed in the sandbox", "blocked")


# ---------------------------------------------------------------- Windows job object
def _make_job(memory_limit: int):
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x8 | 0x100 | 0x2000  # active proc, proc memory, kill on close
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = memory_limit
    ok = k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    if not ok:
        k32.CloseHandle(wintypes.HANDLE(job))
        return None

    def assign(pid: int) -> bool:
        h = k32.OpenProcess(0x1F0FFF, False, pid)
        if not h:
            return False
        try:
            return bool(k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(h)))
        finally:
            k32.CloseHandle(wintypes.HANDLE(h))

    def close() -> None:
        k32.TerminateJobObject(wintypes.HANDLE(job), 1)
        k32.CloseHandle(wintypes.HANDLE(job))

    return assign, close


def _linux_limits(timeout: float):
    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT * 2, MEMORY_LIMIT * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _drain(stream, buf: list[bytes], flag: threading.Event) -> None:
    total = 0
    while chunk := stream.read(4096):
        total += len(chunk)
        if total <= OUTPUT_CAP:
            buf.append(chunk)
        else:
            flag.set()
            break


def run_python(code: str, timeout: float = 5.0) -> dict:
    static_gate(code)
    interp = getattr(sys, "_base_executable", sys.executable) or sys.executable
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="agentflow-sbx-") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
        kwargs: dict = {}
        job = None
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            job = _make_job(MEMORY_LIMIT)
        else:
            kwargs["preexec_fn"] = _linux_limits(timeout)
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [interp, "-I", "-S", str(CHILD), str(script)], cwd=tmp, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs,
        )
        if job:
            job[0](proc.pid)
        out: list[bytes] = []
        err: list[bytes] = []
        flood = threading.Event()
        threads = [
            threading.Thread(target=_drain, args=(proc.stdout, out, flood), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err, flood), daemon=True),
        ]
        for t in threads:
            t.start()

        def kill() -> None:
            if job:
                job[1]()
            elif os.name != "nt":
                import signal

                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.kill()

        timed_out = False
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if flood.is_set():
                kill()
                break
            if time.monotonic() > deadline:
                timed_out = True
                kill()
                break
            time.sleep(0.01)
        proc.wait()
        for t in threads:
            t.join(1)
        if job and not timed_out:
            job[1]()
    stdout = b"".join(out).decode("utf-8", "replace")
    stderr = b"".join(err).decode("utf-8", "replace")
    ms = int((time.perf_counter() - started) * 1000)
    if timed_out:
        raise SandboxError(f"execution exceeded the {timeout:.0f}s time limit and was killed", "timeout")
    if flood.is_set():
        raise SandboxError("output exceeded the size limit and the process was killed", "output")
    if proc.returncode != 0:
        lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
        final = [ln for ln in lines if not ln.startswith((" ", "\t", "Traceback"))][-1:]
        where = [ln.strip() for ln in lines if '"<agent-code>"' in ln][-1:]
        msg = " ".join(final + [f"({where[0].split(', ', 1)[1].split(',')[0]})" if where else ""]).strip()
        msg = msg or f"process exited with code {proc.returncode} (memory limit?)"
        kind = "blocked" if "not allowed in the sandbox" in msg else "runtime"
        raise SandboxError(msg, kind)
    return {"stdout": stdout, "stderr": stderr.strip(), "duration_ms": ms}
