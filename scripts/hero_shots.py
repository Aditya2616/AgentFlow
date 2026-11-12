"""Capture the README hero frames: a packet in flight mid-run, and a retry in motion.

    python scripts/hero_shots.py        # writes docs/assets/hero-*.png and retry-*.png

Plays a recorded run at half speed in headless Chromium (static site, port 8102) and takes the frame when the
state asked for is on screen: a lime packet on an edge with the trace part filled, then an orange packet
(an error, retry or re-plan verdict) returning from the verifier.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8102
BASE = f"http://127.0.0.1:{PORT}/"
VIEWS = {
    "desktop": dict(viewport={"width": 1440, "height": 900}),
    "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, has_touch=True, is_mobile=True),
}
PROBE = """() => {
  const p = [...document.querySelectorAll('#graph .packet')];
  const col = p.map(n => n.querySelector('.trail').getAttribute('stroke'));
  const len = Math.max(0, ...p.map(n => { const t = n.querySelector('.trail'); const g = n => +t.getAttribute(n);
    return Math.hypot(g('x2') - g('x1'), g('y2') - g('y1')); }));
  return {n: p.length, ember: col.some(c => c.includes('ember')), lime: col.some(c => c.includes('lime')),
          len, rows: document.querySelectorAll('#rows .row').length, done: !document.querySelector('#answer').hidden};
}"""


def grab(page, want_ember: bool, path: Path, min_rows: int, timeout: float = 40.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        s = page.evaluate(PROBE)
        if s["n"] and (s["ember"] if want_ember else s["lime"] and not s["ember"]) and s["len"] >= 70 and s["rows"] >= min_rows and not s["done"]:
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(path))
            return True
        if s["done"]:
            return False
        time.sleep(0.02)
    return False


def main() -> int:
    subprocess.run([sys.executable, "scripts/build_docs.py"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "-d", "docs"],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = ROOT / "docs" / "assets"
    ok = True
    try:
        time.sleep(1.5)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for vname, opts in VIEWS.items():
                for shot, want_ember, rec_index, min_rows in (("hero", False, 0, 6), ("retry", True, 0, 6)):
                    ctx = browser.new_context(**opts)
                    page = ctx.new_page()
                    page.goto(BASE, wait_until="networkidle")
                    page.wait_for_selector("#badge:not(:empty)")
                    page.select_option("#rec-select", index=rec_index)
                    page.click('#speed button[data-speed="0.5"]')
                    time.sleep(0.4)
                    page.evaluate("document.querySelector('#scrubber').value = 0; document.querySelector('#scrubber').dispatchEvent(new Event('input'))")
                    page.click("#play-btn") if not page.evaluate("document.querySelector('#play-btn').getAttribute('aria-label') === 'Pause'") else None
                    got = grab(page, want_ember, out / f"{shot}-{vname}.png", min_rows)
                    print(f"{shot}-{vname}: {'captured' if got else 'NOT captured'}")
                    ok &= got
                    ctx.close()
            browser.close()
    finally:
        srv.terminate()
        srv.wait(10)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
