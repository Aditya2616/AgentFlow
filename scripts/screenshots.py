"""Drive the UI in headless Chromium and capture screenshots, console errors and overflow checks.

    python scripts/screenshots.py            # server mode (uses port 8102) and static docs mode
    python scripts/screenshots.py --assets   # also copy the README screenshots into docs/assets/

Screenshots go to .shots/ (ignored by git). Exit code 1 if any console error or horizontal overflow.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / ".shots"
PORT = 8102
BASE = f"http://127.0.0.1:{PORT}/"
VIEWS = {
    "desktop": dict(viewport={"width": 1440, "height": 900}),
    "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, has_touch=True, is_mobile=True),
}


def wait_up(timeout: float = 40) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(BASE, timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("server did not come up")


def start(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_page(page, name: str, problems: list[str], phone: bool = False) -> None:
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    if overflow > 0:
        problems.append(f"{name}: horizontal overflow of {overflow}px")
    if not phone:
        return
    small = page.evaluate(
        """() => [...document.querySelectorAll('button, select, input[type=range], textarea, a.dl')]
            .filter(e => e.offsetParent !== null && !e.closest('[hidden]'))
            .map(e => [e, e.getBoundingClientRect()])
            .filter(([e, r]) => r.height < 43.5)
            .map(([e, r]) => (e.id || e.className || e.tagName) + ':' + Math.round(r.height))"""
    )
    if small:
        problems.append(f"{name}: small tap targets {small[:6]}")


def run_mode(label: str, live: bool, problems: list[str]) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for vname, opts in VIEWS.items():
            ctx = browser.new_context(**opts)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("console", lambda m, e=errors: e.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda exc, e=errors: e.append(str(exc)))
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_selector("#badge:not(:empty)")
            time.sleep(1.6)
            page.screenshot(path=SHOTS / f"{label}-{vname}-01-playing.png")
            page.wait_for_function("document.querySelector('#answer') && !document.querySelector('#answer').hidden", timeout=60000)
            time.sleep(0.8)
            page.screenshot(path=SHOTS / f"{label}-{vname}-02-done.png", full_page=True)
            check_page(page, f"{label}/{vname}/recorded", problems, vname == "phone")
            if vname == "phone":
                page.click("#sheet-handle")
                time.sleep(0.7)
                page.screenshot(path=SHOTS / f"{label}-{vname}-03-trace-sheet.png")
                page.click("#sheet-handle")
            # open a row to check the detail view
            page.click("#rows .row:nth-child(4)", force=True)
            # scrub back to the middle
            box = page.locator("#scrubber").bounding_box()
            page.mouse.click(box["x"] + box["width"] * 0.45, box["y"] + box["height"] / 2)
            time.sleep(0.5)
            page.screenshot(path=SHOTS / f"{label}-{vname}-04-scrubbed.png")
            if live:
                page.click("#tab-live")
                page.select_option("#llm-select", "scripted")
                page.select_option("#task-select", index=8)
                page.select_option("#mistake-select", "bad_args")
                page.click("#run-btn")
                page.wait_for_function("document.querySelector('#answer') && !document.querySelector('#answer').hidden", timeout=60000)
                time.sleep(1.0)
                page.screenshot(path=SHOTS / f"{label}-{vname}-05-live.png", full_page=True)
                check_page(page, f"{label}/{vname}/live", problems, vname == "phone")
            for e in errors:
                problems.append(f"{label}/{vname}: console error: {e}")
            ctx.close()
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()
    SHOTS.mkdir(exist_ok=True)
    problems: list[str] = []
    py = sys.executable
    srv = start([py, "-m", "agentflow.cli", "serve", "--port", str(PORT)])
    try:
        wait_up()
        run_mode("server", True, problems)
    finally:
        srv.terminate()
        srv.wait(10)
    if not args.skip_static:
        subprocess.run([py, "scripts/build_docs.py"], cwd=ROOT, check=True)
        srv = start([py, "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "-d", "docs"])
        try:
            wait_up()
            run_mode("static", False, problems)
        finally:
            srv.terminate()
            srv.wait(10)
    if args.assets:
        assets = ROOT / "docs" / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        for src, dst in [("server-desktop-02-done.png", "ui-desktop.png"), ("server-phone-01-playing.png", "ui-phone.png"),
                         ("server-phone-03-trace-sheet.png", "ui-phone-trace.png")]:
            if (SHOTS / src).exists():
                shutil.copyfile(SHOTS / src, assets / dst)
    print("\n".join(problems) if problems else "no console errors, no overflow, tap targets ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
