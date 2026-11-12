"""Copy the web UI into docs/ so GitHub Pages can serve it with no backend.

docs/assets/ (README screenshots) is left alone.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB, DOCS = ROOT / "web", ROOT / "docs"


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for item in WEB.iterdir():
        dst = DOCS / item.name
        if item.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(item, dst)
        else:
            shutil.copyfile(item, dst)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"copied {WEB} -> {DOCS}")


if __name__ == "__main__":
    main()
