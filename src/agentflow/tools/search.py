"""Local document search: Okapi BM25 over section-level chunks, optional embedding fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOP = frozenset(
    "a an the of to in on for and or is are was were be been it its this that with as by at from "
    "what which who how do does did can i we you our my me about tell please find me than then "
    "there their they them these those if so not no".split()
)


def _norm(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [_norm(t) for t in toks if t not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = docs
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tfs, self.lens, strict=True):
            s = 0.0
            for q in set(query):
                f = tf.get(q, 0)
                if not f:
                    continue
                s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


@dataclass
class Chunk:
    doc: str
    section: str
    text: str


def load_chunks(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        title, section, buf = path.stem, "", []

        def flush(path=path, title=title) -> None:
            body = " ".join(buf).strip()
            if body:
                chunks.append(Chunk(path.name, section, f"{title} / {section}: {body}" if section else body))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                buf, section = [], line[3:].strip()
            elif line.strip():
                buf.append(line.strip())
        flush()
    return chunks


class CorpusIndex:
    def __init__(self, corpus_dir: Path, mode: str = "bm25") -> None:
        self.chunks = load_chunks(corpus_dir)
        self.bm25 = BM25([tokenize(c.doc.replace("_", " ") + " " + c.text) for c in self.chunks])
        self.mode = mode
        self._emb = None
        self._model = None

    def _embed(self):
        if self._emb is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._emb = self._model.encode([c.text for c in self.chunks], normalize_embeddings=True)
        return self._emb

    def search(self, query: str, k: int = 3) -> list[dict]:
        sc = self.bm25.scores(tokenize(query))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        if self.mode == "hybrid":
            emb = self._embed()
            qv = self._model.encode([query], normalize_embeddings=True)[0]  # type: ignore[union-attr]
            dense = emb @ qv
            d_order = sorted(range(len(dense)), key=lambda i: -dense[i])
            rrf: dict[int, float] = {}
            for ranking in (order, d_order):
                for r, i in enumerate(ranking):
                    rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
            order = sorted(rrf, key=lambda i: -rrf[i])
            scores = {i: rrf[i] * 60 for i in order}
            # a hit needs lexical support or a decent dense score
            order = [i for i in order if sc[i] > 0 or dense[i] > 0.3]
        else:
            scores = {i: sc[i] for i in order}
            order = [i for i in order if sc[i] > 0]
        return [
            {"doc": self.chunks[i].doc, "section": self.chunks[i].section,
             "text": self.chunks[i].text, "score": round(float(scores[i]), 3)}
            for i in order[:k]
        ]
