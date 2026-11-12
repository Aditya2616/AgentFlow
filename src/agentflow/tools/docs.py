"""Read or summarise a local document. Paths are jailed to the corpus directory."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .search import tokenize

MAX_CHARS = 4000


class DocError(Exception):
    pass


def resolve(root: Path, filename: str) -> Path:
    name = filename.strip().strip("\"'")
    if not name or "/" in name or "\\" in name or name.startswith(".") or ".." in name:
        raise DocError(f"invalid file name {filename!r}; use a bare file name from the corpus")
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        avail = ", ".join(sorted(p.name for p in root.glob("*.md")))
        raise DocError(f"file {name!r} not found. Available files: {avail}")
    return path


def summarise(text: str, max_sentences: int = 3) -> str:
    """Extractive summary: highest scoring sentences by word frequency, in original order."""
    body = re.sub(r"^#+ .*$", "", text, flags=re.M)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if len(s.strip()) > 20]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    freq = Counter(t for s in sentences for t in tokenize(s))
    scored = [
        (sum(freq[t] for t in tokenize(s)) / (len(tokenize(s)) or 1), i) for i, s in enumerate(sentences)
    ]
    keep = sorted(i for _, i in sorted(scored, reverse=True)[:max_sentences])
    return " ".join(sentences[i] for i in keep)


def run_docs(root: Path, filename: str, action: str = "read") -> dict:
    path = resolve(root, filename)
    text = path.read_text(encoding="utf-8")
    if action == "summarise":
        return {"file": path.name, "summary": summarise(text), "chars": len(text)}
    return {"file": path.name, "text": text[:MAX_CHARS], "chars": len(text)}
