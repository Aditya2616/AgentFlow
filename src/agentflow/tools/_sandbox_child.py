"""Child side of the Python sandbox. Run as:  python -I -S _sandbox_child.py <code-file>

This file is executed by a separate interpreter process, never imported by the agent.
It narrows what user code can reach. It is a best-effort filter, not a security boundary.
"""

import ast
import builtins
import sys

ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "itertools", "collections", "datetime", "decimal",
    "fractions", "random", "string", "textwrap", "functools", "operator", "heapq", "bisect",
    "typing", "dataclasses", "enum", "time", "calendar", "cmath", "numbers",
}
REMOVED_BUILTINS = {
    "open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit", "help",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}


def main() -> None:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source, filename="<agent-code>")
    last = tree.body[-1] if tree.body else None
    is_print = (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Name)
        and last.value.func.id == "print"
    )
    if isinstance(last, ast.Expr) and not is_print:  # behave like a REPL: show the final value
        wrapped = ast.parse(f"_r = ({ast.unparse(last.value)})\nif _r is not None:\n    print(_r)")
        tree.body[-1:] = wrapped.body
        ast.fix_missing_locations(tree)
    code = compile(tree, "<agent-code>", "exec")

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Imports made by the standard library itself pass; imports from user code are filtered.
        if globals is not None and globals.get("__name__") == "__sandbox__":
            if level != 0 or name.split(".")[0] not in ALLOWED_MODULES:
                raise ImportError(f"import of {name!r} is not allowed in the sandbox")
        return real_import(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in vars(builtins).items() if k not in REMOVED_BUILTINS}
    safe["__import__"] = guarded_import
    env = {"__name__": "__sandbox__", "__builtins__": safe}
    sys.setrecursionlimit(2000)
    exec(code, env)  # noqa: S102


main()
