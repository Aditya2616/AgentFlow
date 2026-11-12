"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
CORPUS_DIR = PKG_DIR / "data" / "corpus"
REPO_ROOT = PKG_DIR.parents[1]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


@dataclass
class AgentConfig:
    """Budgets and switches for one agent run."""

    max_attempts: int = 3  # tries per step (the first try counts)
    max_replans: int = 2
    max_acts: int = 14  # hard cap on tool-filling attempts per run
    max_plan_steps: int = 6
    verifier: bool = True


@dataclass
class Settings:
    corpus_dir: Path = CORPUS_DIR
    db_path: Path = field(default_factory=lambda: _env_path("AGENTFLOW_DB", REPO_ROOT / "data" / "sales.db"))
    outbox_dir: Path = field(default_factory=lambda: _env_path("AGENTFLOW_OUTBOX", REPO_ROOT / "outbox"))
    runs_dir: Path = field(default_factory=lambda: _env_path("AGENTFLOW_RUNS", REPO_ROOT / "runs"))
    web_dir: Path = field(default_factory=lambda: _env_path("AGENTFLOW_WEB", REPO_ROOT / "web"))
    local_model: str = field(default_factory=lambda: os.environ.get("AGENTFLOW_LOCAL_MODEL", "Qwen/Qwen3-0.6B"))
    threads: int = field(default_factory=lambda: int(os.environ.get("AGENTFLOW_THREADS", "3")))
    search_mode: str = field(default_factory=lambda: os.environ.get("AGENTFLOW_SEARCH", "bm25"))
    smtp_url: str = field(default_factory=lambda: os.environ.get("AGENTFLOW_SMTP_URL", ""))
    allow_send: bool = field(default_factory=lambda: os.environ.get("AGENTFLOW_ALLOW_SEND", "0") == "1")
    email_domains: list[str] = field(
        default_factory=lambda: [d for d in os.environ.get("AGENTFLOW_EMAIL_ALLOW", "").split(",") if d]
    )

    def ensure_db(self) -> Path:
        from .data.build_db import build_db

        if not self.db_path.exists():
            build_db(self.db_path)
        return self.db_path
