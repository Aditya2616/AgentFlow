from __future__ import annotations

import pytest

from agentflow.config import Settings
from agentflow.data.build_db import build_db
from agentflow.evals.tasks import build_tasks


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    return build_db(tmp_path_factory.mktemp("db") / "sales.db")


@pytest.fixture()
def settings(tmp_path, db_path):
    s = Settings()
    s.db_path = db_path
    s.outbox_dir = tmp_path / "outbox"
    s.runs_dir = tmp_path / "runs"
    return s


@pytest.fixture(scope="session")
def tasks(db_path):
    return {t.id: t for t in build_tasks(db_path)}
