"""The README, the chart and the UI's results file are generated from the result files; this fails if they drift."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run():
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_numbers.py"), "--check"],
                          capture_output=True, text=True, cwd=ROOT)


def test_readme_tables_prose_numbers_and_ui_data_match_the_result_files():
    r = _run()
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_check_notices_a_hand_edited_number():
    readme = ROOT / "README.md"
    original = readme.read_bytes()
    try:
        text = original.decode("utf-8")
        assert "17 code samples" in text
        readme.write_bytes(text.replace("17 code samples", "18 code samples").encode("utf-8"))
        r = _run()
        assert r.returncode == 1 and "red-team samples" in r.stdout
    finally:
        readme.write_bytes(original)
