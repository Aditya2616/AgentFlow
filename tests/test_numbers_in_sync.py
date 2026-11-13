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


def test_percentages_round_halves_up_like_the_ui():
    from agentflow.evals.stats import pct_int, round_half_up

    assert pct_int(15, 24) == 63 and pct_int(75, 120) == 63  # 62.5, not banker's 62
    assert pct_int(69, 120) == 58 and pct_int(63, 120) == 53  # 57.5 and 52.5 (floating-point division gets these wrong)
    assert pct_int(17, 24) == 71 and pct_int(0, 24) == 0 and pct_int(24, 24) == 100
    assert round_half_up(62.5) == 63 and round_half_up(62.49) == 62
