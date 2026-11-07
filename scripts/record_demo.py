"""Record a silent, captioned demo video of the live app (for a voice-over), following docs/demo.md.

Usage (separate environment with Playwright; not part of requirements.lock):
  python -m venv recvenv && recvenv/Scripts/pip install playwright && recvenv/Scripts/python -m playwright install chromium
  recvenv/Scripts/python scripts/record_demo.py --url https://<live-url> --out <folder>

Drives Chromium at 1280x720 with Playwright's video recorder. Inputs: two Stanford Cars test images (illustrative
only; the dataset is for non-commercial research and is not redistributed in the repository) and two CC-licensed
negative-suite photos (attribution shown in the caption). Captions are drawn on top of the page; nothing in the
app is modified.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

DATA = Path(os.environ.get("SEMEVIA_DATA_ROOT", r"C:\semevia\StanfordCars\stanford_cars"))
NEG = Path(os.environ.get("SEMEVIA_NEG_ROOT", r"C:\semevia\negatives"))
W, H = 1280, 720
PACE = 1.0   # set by --pace; stretches every hold


def hold(secs: float) -> None:
    time.sleep(secs * PACE)

CAPTION_JS = """(t) => {
  let d = document.getElementById('__cap');
  if (!d) { d = document.createElement('div'); d.id = '__cap';
    d.style.cssText = 'position:fixed;left:50%;bottom:22px;transform:translateX(-50%);max-width:1100px;z-index:99999;' +
      'background:rgba(11,11,11,.82);color:#fff;font:600 19px/1.35 system-ui,Segoe UI,sans-serif;padding:10px 18px;' +
      'border-radius:10px;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,.25);pointer-events:none';
    document.body.appendChild(d); }
  d.textContent = t; d.style.display = t ? 'block' : 'none';
}"""

CARD = """<html><body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
background:#0f172a;color:#f8fafc;font-family:system-ui,Segoe UI,sans-serif"><div style="text-align:center;max-width:1000px">
<div style="font-size:54px;font-weight:800;letter-spacing:.5px">{title}</div>
<div style="font-size:24px;margin-top:18px;color:#cbd5e1">{sub}</div>
<div style="font-size:18px;margin-top:28px;color:#94a3b8">{foot}</div></div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pace", type=float, default=1.45, help="multiply every hold (1.45 gives about 2:45)")
    ap.add_argument("--repo", default="https://github.com/DuhItzAniket/SEMEVIA/blob/semevia/implementation")
    a = ap.parse_args()
    global PACE
    PACE = a.pace
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clear_car = DATA / "cars_test" / "00957.jpg"      # Audi R8 Coupe 2012: one category in the candidate set
    family_car = DATA / "cars_test" / "06207.jpg"     # Dodge Caliber Wagon: two model years remain (r3)
    bus = NEG / "tune" / "uv_tune_01.jpg"             # CC BY 4.0, ALC, Wikimedia Commons
    moto = NEG / "tune" / "tw_tune_03.jpg"            # CC BY-SA 4.0, Terragio67, Wikimedia Commons
    kia = NEG / "tune" / "uc_tune_01.jpg"             # CC BY-SA 4.0, Ethan Llamas, Wikimedia Commons

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(out / "raw"),
                                  record_video_size={"width": W, "height": H}, device_scale_factor=1)
        pg = ctx.new_page()
        cap = lambda t: pg.evaluate(CAPTION_JS, t)  # noqa: E731

        def card(title, sub, foot="", secs=4.0):
            pg.set_content(CARD.format(title=title, sub=sub, foot=foot))
            hold(secs)

        def scroll_to(sel, secs=0.0):
            pg.locator(sel).scroll_into_view_if_needed()
            pg.evaluate("(s) => document.querySelector(s).scrollIntoView({behavior:'smooth', block:'start'})", sel)
            hold(1.0 + secs)

        def analyze(path, mode="hybrid"):
            cap("")
            pg.locator("#reset").click()
            pg.set_input_files("#file", str(path))
            pg.locator(f"input[name=mode][value={mode}]").check()
            pg.locator("#analyze").click()
            pg.wait_for_selector("#result:not([hidden])", timeout=60000)
            hold(1.2)

        card("SEMEVIA", "Vehicle identity through evidence",
             "Team Hypermetropia (T35) - Computer Vision Hackathon 2026 - Problem statement PS5", 5)

        pg.goto(a.url, wait_until="networkidle")
        pg.wait_for_function("document.getElementById('ready').classList.contains('on')", timeout=60000)
        cap("Live public URL. Upload a car photo, choose a mode (Hybrid = classical + deep), press Analyze.")
        hold(4)

        analyze(clear_car)
        cap("YOLO11n finds the car (thick box). Illustrative Stanford Cars test image.")
        scroll_to("#result", 3)
        cap("Best guess and Top-5: make, model and year among 196 categories. Detector and classifier scores are separate.")
        scroll_to("#pred-card", 4)
        cap("Identity Evidence Passport, a calibrated candidate set: " + pg.locator("#pp-state").inner_text().strip())
        scroll_to("#passport-card", 5)

        analyze(family_car)
        cap("A harder photo. Passport: " + pg.locator("#pp-state").inner_text().strip())
        scroll_to("#passport-card", 5)
        if pg.locator("#show-all").is_visible():
            pg.locator("#show-all").click()
            hold(1.5)

        cap("Evidence Challenge: blur one of six regions at a time and re-score with the same deployed model.")
        scroll_to("#challenge-card", 1)
        pg.locator("#run-challenge").click()
        pg.wait_for_selector("#ch-result:not([hidden])", timeout=90000)
        hold(2)
        btns = pg.locator("#ch-buttons button")
        if btns.count() > 2:
            btns.nth(2).click()
            hold(2.5)
            cap("The number is how much the lead of A over B shrinks. It measures sensitivity, not which car part decides.")
            hold(3)
            pg.locator("#ch-restore").click()
            hold(1.5)

        cap("Classical computer vision, shown for every upload: crop, grayscale, Canny edges, contours, geometry.")
        scroll_to("#proc-card", 5)

        def outcome() -> str:
            """Caption text for what the app actually decided (never assumed)."""
            if pg.locator("#passport-card").is_visible():
                state = pg.locator("#pp-state").inner_text().strip()
                if state.lower().startswith("one category"):
                    return f"Accepted as a car identity: '{state}'"
                return f"No identity accepted: '{state}'"
            return "No supported vehicle detected: no identity claimed."

        analyze(moto)
        cap("Unsupported input: a motorcycle (Wikimedia Commons, CC BY-SA 4.0, Terragio67). " + outcome())
        scroll_to("#result", 5)

        analyze(kia)
        cap("Unsupported input: a 2021 Kia EV6, newer than every Stanford Cars class (Wikimedia Commons, CC BY-SA 4.0, Ethan Llamas).")
        scroll_to("#passport-card", 3)
        cap(outcome())
        hold(4)

        analyze(bus)
        cap("Known failure, shown on purpose: a city bus (Wikimedia Commons, CC BY 4.0, ALC). " + outcome())
        scroll_to("#passport-card", 4)
        cap("Softmax rejection is not an open-set detector; false acceptance on 40 held-out negatives is reported in the eval file.")
        hold(4)

        cap("")
        pg.goto(f"{a.repo}/eval_P5_T35.md", wait_until="domcontentloaded")
        hold(3)
        cap("Every number in eval_P5_T35.md is printed by a script from saved reports: 8,041 test images, end to end.")
        hold(6)
        pg.evaluate('window.scrollBy({top: 1400, behavior: \"smooth\"})')
        hold(4)
        cap("Classical vs deep vs hybrid, IoU, mAP, size and latency, robustness, calibration, ablations, failure cases.")
        pg.evaluate('window.scrollBy({top: 1600, behavior: \"smooth\"})')
        hold(5)

        pg.goto(f"{a.repo}/docs/architecture.md", wait_until="domcontentloaded")
        cap("Architecture, design, timeline and the full reproduction runbook are in the repository.")
        hold(4)
        pg.evaluate('window.scrollBy({top: 900, behavior: \"smooth\"})')
        hold(5)
        cap("")
        card("SEMEVIA", "github.com/DuhItzAniket/SEMEVIA", "Team Hypermetropia - Aniket Kumar - C Lakshanaa", 5)

        video = pg.video.path()
        ctx.close()
        browser.close()
    dst = out / "semevia_demo_silent.webm"
    shutil.move(video, dst)
    shutil.rmtree(out / "raw", ignore_errors=True)
    print("wrote", dst)


if __name__ == "__main__":
    main()
