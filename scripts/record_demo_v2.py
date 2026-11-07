"""Screen recorder for minutes 2 and 3 of the SEMEVIA demo video (v2), driven by the cue sheet.

What it makes (per minute, exactly 60.0 s, 1920x1080, 30 fps, H.264 yuv420p, no audio track):
  <out>/minute2_clean.mp4       app + on-brand cards/figures, no captions (for the voice-over edit)
  <out>/minute2_captioned.mp4   the same frames with captions_minute2.srt burned in
  <out>/captions_minute2.srt    copy of the caption file (and the .ass actually burned)
  ... and the same for minute 3, plus <out>/minuteN_timing.json (when each cue really ran) and
  <out>/check/ (stills every 5 s and at the end of every cue) when --check is given.

How it works:
  * Playwright Chromium, viewport 1920x1080, device_scale_factor 1, record_video at 1920x1080, one fresh browser
    context per minute. The app itself is shown at 130 % (CSS zoom on its masthead, main column and footer;
    --zoom), so its column fills about 77 % of the frame width and its text is about 30 % larger. Playwright's
    video ignores device_scale_factor, so CSS zoom is the way to get larger, still sharp, text.
  * Loads the live app, waits for the footer badge #ready to get class "on", does the off-camera preparation
    (minute 2: analyses the Audi R8 example and runs the six-region challenge, preloading every blurred variant,
    so no spinner or half-loaded image is ever on camera; minute 3: one warm-up upload of the judge photo, then
    Reset), then shows a magenta sync frame and removes it at t = 0. Post-processing finds that frame in the raw
    recording and cuts exactly 1800 frames (60.0 s) from there, so the warm-up is trimmed away frame-accurately.
  * App targets are framed in the upper part of the frame and captions are small and low (Segoe UI 36, 30 px
    from the bottom), so captions do not cover the part of the app being talked about.
  * Every cue runs at its start time and holds until its end (actions are timed so they fit inside the window;
    real durations are logged and a warning is printed when one overruns).
  * Card and figure screens are full-screen overlays drawn inside the same page (paper background, ink text,
    signal-orange accents, small SEMEVIA wordmark), so the app state underneath is kept.
  * A drawn mouse cursor glides to every click. Minute 3's upload is a visible drag and drop: a small file card
    with the photo travels in from the left edge onto #drop, then real dragenter/dragover/drop events carrying a
    File built from the image bytes are dispatched on #drop, so the app shows its own thumbnail.
  * The judge photo is uploaded as "unseen_photo.<ext>" so the file name does not give the answer away.
  * Photo credit (author, licence, Wikimedia Commons) is shown in a small chip on the app screens.

Usage (Windows; Playwright lives in its own venv, not in requirements.lock):
  set PLAYWRIGHT_BROWSERS_PATH=C:\\semevia\\playwright
  C:\\semevia\\recvenv\\Scripts\\python.exe scripts\\record_demo_v2.py --only both --check

  --url     live app (default https://semevia.web.app)
  --cues    cue sheet JSON (default ...\\SEMEVIA_demo\\v2\\script\\cues.json); captions_minuteN.srt are read
            from the same folder unless --srt-dir is given
  --images  folder with the judge's unseen photos (default ...\\SEMEVIA_demo\\v2\\judge_inputs). Minute 3 uses the
            file named by the drop cue's screen.image in cues.json (a path relative to ...\\SEMEVIA_demo\\v2, e.g.
            "judge_inputs/1_bugatti_veyron.jpg"); if that is missing, the first image file alphabetically in this
            folder; if the folder has no image, app/static/examples/audi_r8.jpg (the log says so).
            credits.json there: {"file.jpg": {"author", "license"}}; the credit is shown on the app screens and
            replaces "{judge_credit}" in the end card's bullets.
  --zoom    CSS zoom for the app (default 1.3)
  --out     output folder (default ...\\SEMEVIA_demo\\v2\\recordings)
  --only    minute2 | minute3 | both (default both)
  --ffmpeg  path to ffmpeg.exe (ffprobe.exe must sit next to it); default: the WinGet Gyan build, else PATH
  --check   also write verification stills to <out>\\check

Never uses Stanford Cars images (non-commercial licence). Numbers on the cards come from the cue sheet, which
was checked against eval_P5_T35.md.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(r"C:\Users\Luikz\Downloads\SEMEVIA_demo\v2")
FFMPEG_DEFAULT = Path(r"C:\Users\Luikz\AppData\Local\Microsoft\WinGet\Packages"
                      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
W, H, FPS, SECONDS = 1920, 1080, 30, 60.0
FRAMES = int(round(FPS * SECONDS))
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPEAKER_LANE = {"Aniket Kumar": "Lane A · model, data, evaluation",
                "C Lakshanaa": "Lane B · application, deployment"}

# How each scroll target is framed: (mode, offset px). "start": element top sits `offset` px below the viewport
# top. "center": element centred (but never closer than `offset` to the top). Everything is framed from the top so
# the part being talked about stays above the caption band (about y 950-1050).
FRAMING = {
    "#result": ("start", 30), "#verdict": ("start", 40), "#pred-card": ("start", 40),
    "#proc-card": ("start", 50), "#cmp-card": ("start", 40), "#passport-card": ("start", 40),
    "#challenge-card": ("start", 40), "#ch-pair": ("start", 50), "#ref-card": ("start", 40),
}
PARK_X = 1815          # where the cursor rests: in the right margin, outside the zoomed app column (x 219-1701)
APP_ZOOM_CSS = "body > .masthead, body > main, body > footer {{ zoom: {z}; }}"

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


# ----------------------------------------------------------------------------------------------- page-side helpers
DEMO_JS = r"""(markSvg) => {
  if (window.__demo) return;
  const css = `
  .dm-scr{position:fixed;inset:0;z-index:2147483000;background:#f4f1ea;color:#15171a;opacity:0;visibility:hidden;
    pointer-events:none;font-family:"Segoe UI",system-ui,-apple-system,Arial,sans-serif;
    transition:opacity .42s ease,visibility 0s linear .42s}
  .dm-scr.in{opacity:1;visibility:visible;transition:opacity .42s ease}
  .dm-scr.instant{transition:none!important}
  .dm-head{position:absolute;left:96px;right:96px;top:52px;display:flex;align-items:center;justify-content:space-between;
    border-bottom:1px solid #dcd6c8;padding-bottom:22px}
  .dm-brand{display:flex;align-items:center;gap:14px;font-weight:800;letter-spacing:.22em;font-size:24px;color:#15171a}
  .dm-brand svg{width:40px;height:40px}
  .dm-brand i{font-style:normal;color:#d9480f}
  .dm-meta{font:600 16px/1.2 Consolas,"Cascadia Mono",monospace;letter-spacing:.14em;text-transform:uppercase;color:#74777c}
  .dm-body{position:absolute;left:190px;right:190px;top:150px;bottom:165px;display:flex;flex-direction:column;justify-content:center}
  .dm-c{position:absolute;width:64px;height:64px;border:0 solid #d9480f;opacity:0}
  .dm-c.tl{left:128px;top:168px;border-left-width:6px;border-top-width:6px}
  .dm-c.br{right:128px;bottom:150px;border-right-width:6px;border-bottom-width:6px}
  .play .dm-c{animation:dmFade .5s ease forwards .1s}
  .dm-eyebrow{font:600 20px/1.2 Consolas,"Cascadia Mono",monospace;letter-spacing:.16em;text-transform:uppercase;color:#d9480f;margin:0 0 18px}
  .dm-title{font-size:76px;line-height:1.06;font-weight:700;letter-spacing:-.02em;margin:0 0 12px}
  .dm-title.big{font-size:100px}
  .dm-title b{color:#d9480f;font-weight:700}
  .dm-sub{font:600 26px/1.3 Consolas,"Cascadia Mono",monospace;color:#4a4d52;margin:6px 0 34px;letter-spacing:.02em}
  .dm-list{list-style:none;margin:28px 0 0;padding:0;display:grid;gap:16px;max-width:1460px}
  .dm-list li{display:grid;grid-template-columns:46px 1fr;gap:22px;align-items:start;font-size:34px;line-height:1.3;
    opacity:0;transform:translateY(14px)}
  .play .dm-list li{animation:dmIn .55s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:calc(.35s + var(--i) * .22s)}
  .dm-tick{width:40px;height:40px;margin-top:3px;border:2px solid #d9480f;border-radius:8px;display:grid;place-items:center;background:#fbfaf6}
  .dm-tick svg{width:26px;height:26px}
  .dm-tick path{stroke:#d9480f;stroke-width:3.4;fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:30;stroke-dashoffset:30}
  .play .dm-tick path{animation:dmTick .38s ease forwards;animation-delay:calc(.6s + var(--i) * .22s)}
  .dm-fine{font-size:21px;color:#74777c;margin:40px 0 0;opacity:0}
  .play .dm-fine{animation:dmFade .5s ease forwards 1.2s}
  .dm-fig{position:absolute;left:0;right:0;top:132px;bottom:150px;display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:20px}
  .dm-fig img{max-width:1480px;max-height:690px;background:#fff;border:1px solid #dcd6c8;border-radius:10px;padding:12px;
    box-shadow:0 10px 30px rgba(21,23,26,.08);opacity:0;transform:scale(.985)}
  .play .dm-fig img{animation:dmIn .6s cubic-bezier(.2,.7,.2,1) forwards .1s}
  .dm-figcap{font-size:30px;line-height:1.3;color:#15171a;max-width:1480px;text-align:center;margin:0;opacity:0}
  .play .dm-figcap{animation:dmFade .5s ease forwards .45s}
  @keyframes dmIn{to{opacity:1;transform:none}}
  @keyframes dmFade{to{opacity:1}}
  @keyframes dmTick{to{stroke-dashoffset:0}}
  #dm-cur{position:fixed;left:0;top:0;z-index:2147483646;pointer-events:none;opacity:0;transition:opacity .25s ease;
    filter:drop-shadow(0 3px 4px rgba(0,0,0,.28));will-change:transform}
  #dm-cur svg{display:block;transform-origin:3px 3px;transition:transform .12s ease}
  #dm-cur.down svg{transform:scale(.86)}
  .dm-rip{position:fixed;width:18px;height:18px;margin:-9px 0 0 -9px;border:3px solid #d9480f;border-radius:50%;
    z-index:2147483645;pointer-events:none;animation:dmRip .55s ease-out forwards}
  @keyframes dmRip{from{transform:scale(.3);opacity:1}to{transform:scale(2.8);opacity:0}}
  #dm-ring{position:fixed;z-index:2147482000;pointer-events:none;border:3px solid #d9480f;border-radius:14px;opacity:0;
    box-shadow:0 0 0 8px rgba(217,72,15,.10);transition:opacity .35s ease}
  #dm-ring.on{opacity:1}
  #dm-credit{position:fixed;left:14px;bottom:14px;max-width:178px;z-index:2147481000;pointer-events:none;font:13px/1.35 Consolas,"Cascadia Mono",monospace;
    color:#4a4d52;background:rgba(244,241,234,.94);border:1px solid #dcd6c8;border-radius:6px;padding:5px 10px;display:none}
  #dm-mark{position:fixed;inset:0;z-index:2147483647;background:#ff00ff;display:none}
  .dm-chip{position:fixed;left:0;top:0;width:330px;display:flex;gap:14px;align-items:center;padding:10px 14px 10px 10px;
    background:#fbfaf6;border:1px solid #dcd6c8;border-radius:12px;box-shadow:0 18px 40px rgba(21,23,26,.24);
    z-index:2147483600;pointer-events:none;font-family:"Segoe UI",system-ui,sans-serif;will-change:transform}
  .dm-chip img{width:76px;height:76px;object-fit:cover;border-radius:7px;border:1px solid #dcd6c8;flex:none}
  .dm-chip b{display:block;font-size:17px;color:#15171a;font-weight:600}
  .dm-chip span{display:block;font:12.5px/1.4 Consolas,monospace;color:#74777c;margin-top:4px}
  `;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  const mk = (tag, id, cls) => { const e = document.createElement(tag); if (id) e.id = id; if (cls) e.className = cls; document.body.appendChild(e); return e; };
  const root = mk('div', 'dm-root');
  const ring = mk('div', 'dm-ring');
  const credit = mk('div', 'dm-credit');
  const cur = mk('div', 'dm-cur');
  const mark = mk('div', 'dm-mark');
  cur.innerHTML = '<svg width="32" height="40" viewBox="0 0 32 40"><path d="M3 3 L3 31.5 L10.4 24.6 L15.6 36.6 L21 34.2 L15.8 22.6 L26 22.6 Z" fill="#15171a" stroke="#ffffff" stroke-width="2.4" stroke-linejoin="round"/></svg>';
  const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const anim = (ms, f) => new Promise(res => {
    const t0 = performance.now();
    const step = now => { const k = Math.min(1, (now - t0) / ms); f(ease(k), k); if (k < 1) requestAnimationFrame(step); else res(); };
    requestAnimationFrame(step);
  });
  window.PARKX = 1815;
  let cx = window.PARKX, cy = 760, ringEl = null, ringLoop = 0;
  const place = () => { cur.style.transform = `translate(${cx - 3}px, ${cy - 3}px)`; };
  place();
  const screens = {};
  const D = window.__demo = {
    markSvg,
    prepare(id, htmlText) { const s = document.createElement('div'); s.className = 'dm-scr'; s.innerHTML = htmlText; root.appendChild(s); screens[id] = s; },
    show(id, instant) {
      const s = screens[id]; const old = [...root.children].filter(x => x !== s && x.classList.contains('in'));
      root.appendChild(s);
      if (instant) { s.classList.add('instant'); s.classList.add('in'); s.offsetHeight; s.classList.remove('instant'); }
      else { s.offsetHeight; s.classList.add('in'); }
      s.classList.add('play');
      old.forEach(o => setTimeout(() => { o.classList.add('instant'); o.classList.remove('in', 'play'); o.offsetHeight; o.classList.remove('instant'); }, 450));
    },
    hide() { [...root.children].forEach(s => { s.classList.remove('in'); setTimeout(() => s.classList.remove('play'), 450); }); },
    replay(id) { const s = screens[id]; s.classList.remove('play'); s.offsetHeight; s.classList.add('play'); },
    marker(on) { mark.style.display = on ? 'block' : 'none'; },
    credit(text) { credit.textContent = text || ''; credit.style.display = text ? 'block' : 'none'; },
    cursorShow(v) { cur.style.opacity = v ? '1' : '0'; },
    cursorJump(x, y) { cx = x; cy = y; place(); },
    cursorTo(x, y, ms) { const x0 = cx, y0 = cy; return anim(ms, e => { cx = x0 + (x - x0) * e; cy = y0 + (y - y0) * e; place(); }); },
    async press() {
      cur.classList.add('down');
      const r = document.createElement('div'); r.className = 'dm-rip'; r.style.left = cx + 'px'; r.style.top = cy + 'px';
      document.body.appendChild(r); setTimeout(() => r.remove(), 700);
      await sleep(130); cur.classList.remove('down');
    },
    center(sel) { const r = document.querySelector(sel).getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2, r.width, r.height]; },
    frameY(sel, mode, off) {
      const el = document.querySelector(sel);
      const anchor = el.closest('.report-figure') ? el.closest('.report') : el;   // #result is sticky: frame by its grid row
      const top = anchor.getBoundingClientRect().top + window.scrollY, h = el.getBoundingClientRect().height;
      let y = mode === 'center' ? top - Math.max(off, (innerHeight - h) / 2) : top - off;
      const max = document.documentElement.scrollHeight - innerHeight;
      return Math.round(Math.max(0, Math.min(max, y)));
    },
    scrollTo(y, ms) { const y0 = window.scrollY; return anim(ms, e => window.scrollTo(0, y0 + (y - y0) * e)); },
    ring(sel) {
      ringEl = document.querySelector(sel); ring.classList.add('on'); cancelAnimationFrame(ringLoop);
      const follow = () => { if (!ringEl) return; const r = ringEl.getBoundingClientRect();
        Object.assign(ring.style, { left: (r.left - 8) + 'px', top: (r.top - 8) + 'px', width: (r.width + 16) + 'px', height: (r.height + 16) + 'px' });
        ringLoop = requestAnimationFrame(follow); };
      follow();
    },
    ringOff() { ring.classList.remove('on'); setTimeout(() => { if (!ring.classList.contains('on')) { ringEl = null; cancelAnimationFrame(ringLoop); } }, 400); },
    async dragDrop(b64, mime, name, sizeLabel) {
      const bin = atob(b64); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      const file = new File([u8], name, { type: mime });
      const url = URL.createObjectURL(file);
      const drop = document.getElementById('drop'); const dr = drop.getBoundingClientRect();
      const chip = document.createElement('div'); chip.className = 'dm-chip';
      chip.innerHTML = `<img src="${url}" alt=""><div><b></b><span></span></div>`;
      chip.querySelector('b').textContent = name; chip.querySelector('span').textContent = sizeLabel;
      document.body.appendChild(chip);
      await chip.querySelector('img').decode().catch(() => {});
      const grabX = 46, grabY = 52;                       // where the cursor holds the card
      const x0 = -360, y0 = dr.top + dr.height / 2 + 150;
      const x1 = dr.left + dr.width / 2 - 165, y1 = dr.top + dr.height / 2 - 48;
      const dt = new DataTransfer(); dt.items.add(file);
      const fire = type => drop.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt }));
      let entered = false;
      const put = (x, y, rot) => { chip.style.transform = `translate(${x}px, ${y}px) rotate(${rot}deg)`; cx = x + grabX; cy = y + grabY; place(); };
      put(x0, y0, -6); cur.style.opacity = '1';
      await anim(1750, (e) => {
        const x = x0 + (x1 - x0) * e, y = y0 + (y1 - y0) * e - Math.sin(Math.PI * e) * 70;
        put(x, y, -6 * (1 - e));
        const px = x + grabX, py = y + grabY;
        const inside = px > dr.left && px < dr.right && py > dr.top && py < dr.bottom;
        if (inside && !entered) { fire('dragenter'); entered = true; }
        if (inside) fire('dragover');
      });
      for (let i = 0; i < 8; i++) { fire('dragover'); await sleep(40); }
      fire('drop');
      await anim(260, e => { chip.style.opacity = String(1 - e); chip.style.transform = `translate(${x1 + 165 * e * .4}px, ${y1 + 10 * e}px) scale(${1 - .45 * e})`; });
      chip.remove();
      return { thumbVisible: !document.getElementById('drop-thumb').hidden, name: document.getElementById('file').files[0]?.name };
    },
  };
}"""


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def mark_svg() -> str:
    svg = (REPO / "app" / "static" / "brand" / "mark.svg").read_text(encoding="utf-8")
    svg = re.sub(r"<title>.*?</title>", "", svg, flags=re.S)
    return re.sub(r"<!--.*?-->", "", svg, flags=re.S).strip()


def header_html(meta: str, mark: str) -> str:
    return (f'<div class="dm-head"><span class="dm-brand">{mark}<span>SEME<i>V</i>IA</span></span>'
            f'<span class="dm-meta">{esc(meta)}</span></div>')


TICK = '<span class="dm-tick"><svg viewBox="0 0 26 26"><path d="M5 13.5l5.2 5.2L21 7.5"/></svg></span>'


def card_html(scr: dict, speaker: str, mark: str, fine: str = "") -> str:
    title = scr.get("title", "")
    is_name = title == speaker
    # The narration is synthetic, so cards never name a speaker; they carry the team label (or a cue-given eyebrow).
    eyebrow = scr.get("eyebrow", "Team Hypermetropia · T35")
    t = esc(title).replace("semevia.web.app", "<b>semevia.web.app</b>")
    parts = [header_html("Computer Vision Hackathon 2026 · PS5", mark), '<span class="dm-c tl"></span><span class="dm-c br"></span>',
             '<div class="dm-body">', f'<p class="dm-eyebrow">{esc(eyebrow)}</p>',
             f'<h1 class="dm-title{" big" if is_name else ""}">{t}</h1>']
    if scr.get("subtitle"):
        parts.append(f'<p class="dm-sub">{esc(scr["subtitle"])}</p>')
    items = []
    for i, b in enumerate(scr.get("bullets", [])):
        b = re.sub(r"^\s*✓\s*", "", b)
        items.append(f'<li style="--i:{i}">{TICK}<span>{esc(b)}</span></li>')
    parts.append(f'<ul class="dm-list">{"".join(items)}</ul>')
    if fine:
        parts.append(f'<p class="dm-fine">{esc(fine)}</p>')
    parts.append("</div>")
    return "".join(parts)


def figure_html(scr: dict, mark: str) -> str:
    p = Path(scr["path"])
    if not p.is_absolute() or not p.exists():
        p = REPO / "reports" / "figures" / Path(scr["path"]).name
    data = base64.b64encode(p.read_bytes()).decode()
    return (header_html(f"reports/figures/{p.name}", mark) +
            f'<div class="dm-fig"><img src="data:image/png;base64,{data}" alt="">'
            f'<p class="dm-figcap">{esc(scr.get("caption", ""))}</p></div>')


# ----------------------------------------------------------------------------------------------- inputs
def judge_credit(f: Path) -> str:
    cj = f.parent / "credits.json"
    if cj.exists():
        try:
            c = json.loads(cj.read_text(encoding="utf-8")).get(f.name)
            if c:
                return f"Photo: {c['author']} · {c['license']} · Wikimedia Commons (resized)"
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: could not read {cj}: {e}")
    log(f"WARNING: no credit for {f.name} in {cj}; no credit will be shown. Add one if the licence needs it.")
    return ""


def pick_judge_image(folder: Path, named: str | None = None) -> tuple[Path, str]:
    if named:
        f = Path(named)
        if not f.is_absolute():
            f = DEMO_ROOT / f
        if f.is_file() and f.suffix.lower() in IMAGE_EXT:
            log(f"Minute 3 photo: {f} (named by the drop cue's screen.image)")
            return f, judge_credit(f)
        log(f"WARNING: drop cue names {named!r}, which is not an image file; using the first image in {folder}")
    files = sorted([f for f in folder.glob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXT],
                   key=lambda f: f.name.lower()) if folder.exists() else []
    if files:
        f = files[0]
        log(f"Minute 3 photo: {f} (first of {len(files)} image(s) in {folder})")
        return f, judge_credit(f)
    f = REPO / "app" / "static" / "examples" / "audi_r8.jpg"
    log(f"NOTE: no image in {folder}; falling back to the app example {f} for minute 3.")
    return f, example_credit("audi_r8")


def example_credit(key: str) -> str:
    for c in json.loads((REPO / "app" / "static" / "examples" / "credits.json").read_text(encoding="utf-8")):
        if c["key"] == key:
            return f"Example photo: {c['author']} · {c['license']} · Wikimedia Commons ({c.get('modified', 'resized')})"
    return ""


def srt_to_ass(srt: Path, ass: Path) -> int:
    blocks = re.split(r"\n\s*\n", srt.read_text(encoding="utf-8-sig").strip())
    ts = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")

    def a(h, m, s, ms):
        cs = int(round(int(ms) / 10))
        sec = int(s)
        if cs >= 100:
            cs, sec = cs - 100, sec + 1
        return f"{int(h)}:{int(m):02d}:{sec:02d}.{cs:02d}"
    events = []
    for b in blocks:
        lines = b.splitlines()
        for i, ln in enumerate(lines):
            mt = ts.search(ln)
            if mt:
                g = mt.groups()
                text = r"\N".join(x.strip() for x in lines[i + 1:] if x.strip())
                events.append(f"Dialogue: 0,{a(*g[:4])},{a(*g[4:])},Default,,0,0,0,,{text}")
                break
    head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Segoe UI,36,&H00FFFFFF,&H00FFFFFF,&H33171513,&H33171513,0,0,0,0,100,100,0,0,3,10,0,2,200,200,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass.write_text(head + "\n".join(events) + "\n", encoding="utf-8")
    return len(events)


# ----------------------------------------------------------------------------------------------- recorder
class Minute:
    def __init__(self, pg, cues: list[dict], name: str):
        self.pg, self.cues, self.name = pg, cues, name
        self.t0 = None
        self.overlay = False
        self.timing: list[dict] = []

    # timing
    def now(self) -> float:
        return time.perf_counter() - self.t0

    def until(self, t: float) -> None:
        d = t - self.now()
        if d > 0:   # wait inside Playwright so route handlers (image cache) keep being served while we wait
            self.pg.wait_for_timeout(d * 1000)

    # page helpers
    def js(self, expr: str, arg=None):
        return self.pg.evaluate(expr, arg)

    def hide_overlay(self) -> None:
        if self.overlay:
            self.js("() => __demo.hide()")
            self.overlay = False
            self.pg.wait_for_timeout(120)

    def scroll(self, sel: str, glide_to: str | None = None, ring: str | None = "") -> None:
        mode, off = FRAMING.get(sel, ("start", 60))
        y = self.js("([s, m, o]) => __demo.frameY(s, m, o)", [sel, mode, off])
        cur = self.js("() => window.scrollY")
        dist = abs(y - cur)
        ms = int(min(1400, max(650, 520 + dist * 0.33))) if dist > 2 else 0
        if glide_to:   # move the cursor towards where the button will be once the scroll ends, at the same time
            self.js("""([s, dy, ms]) => { const r = document.querySelector(s).getBoundingClientRect();
                        __demo.cursorTo(r.left + r.width / 2, r.top + r.height / 2 - dy, ms); }""", [glide_to, y - cur, max(ms, 500)])
        if ms:
            self.js("([y, ms]) => __demo.scrollTo(y, ms)", [y, ms])
        ring = sel if ring == "" else ring
        if ring:
            self.js("(s) => __demo.ring(s)", ring)

    def click(self, sel: str, glide_ms: int = 600) -> None:
        x, y, w, h = self.js("(s) => __demo.center(s)", sel)
        self.js("() => __demo.cursorShow(true)")
        self.js("([x, y, ms]) => __demo.cursorTo(x, y, ms)", [x, y, glide_ms])
        self.js("() => __demo.press()")
        self.pg.mouse.click(x, y)

    def wait_js(self, cond: str, deadline: float) -> bool:
        try:
            self.pg.wait_for_function(cond, timeout=max(200, int((deadline - self.now()) * 1000)), polling=50)
            return True
        except PWTimeout:
            return False

    # cue runner
    def run(self, judge: dict | None) -> None:
        for i, c in enumerate(self.cues):
            self.until(c["start"])
            scr = c["screen"]
            rec = {"i": i + 1, "start": c["start"], "end": c["end"], "type": scr["type"],
                   "what": scr.get("title") or scr.get("target") or Path(scr.get("path", "")).name,
                   "began": round(self.now(), 3), "notes": []}
            self.js("() => __demo.ringOff()")
            if scr["type"] in ("card", "figure"):
                if i > 0:
                    self.js("(id) => __demo.show(id, false)", f"s{i}")
                self.js("() => __demo.cursorShow(false)")
                self.overlay = True
                nxt = self.cues[i + 1]["screen"] if i + 1 < len(self.cues) else {}
                if nxt.get("type") == "app" and nxt.get("action") == "scroll":
                    # while the overlay hides the page, park it just above the next target so the reveal shows a
                    # short scroll instead of a long blur (keeps the next click early in its window)
                    self.pg.wait_for_timeout(550)
                    mode, off = FRAMING.get(nxt["target"], ("start", 60))
                    self.js("([s, m, o]) => window.scrollTo(0, Math.max(0, __demo.frameY(s, m, o) - 240))", [nxt["target"], mode, off])
            else:
                self.run_app(c, scr, rec, judge)
            rec["action_done"] = round(self.now(), 3)
            if rec["action_done"] > c["end"] + 0.05:
                rec["notes"].append(f"OVERRUN by {rec['action_done'] - c['end']:.2f} s")
                log(f"WARNING {self.name} cue {i + 1}: action finished at {rec['action_done']:.2f} s, cue ends {c['end']}")
            self.until(c["end"])
            self.timing.append(rec)
            log(f"{self.name} cue {i + 1:2d} {c['start']:5.1f}-{c['end']:5.1f} {scr['type']:6s} {rec['what']}: "
                f"began {rec['began']:.2f}, done {rec['action_done']:.2f} {' '.join(rec['notes'])}")

    def run_app(self, c: dict, scr: dict, rec: dict, judge: dict | None) -> None:
        start, end = c["start"], c["end"]
        act = scr.get("action")
        self.hide_overlay()
        self.js("() => __demo.cursorShow(true)")
        if act == "scroll":
            sel = scr.get("click")
            if sel == "challenge-region":    # the region whose blur cut the margin most (found off-camera)
                sel = f"#ch-buttons button:nth-child({int(self.js('() => window.__chBest') or 0) + 1})"
            self.scroll(scr["target"], glide_to=sel, ring=scr.get("ring", ""))
            if sel:
                if sel == "#run-compare":
                    self.until(start + 0.8)        # the cue sheet asks for the click about 1 s into the cue
                elif scr.get("click") == "challenge-region":
                    self.until(start + 1.2)        # let the viewer find the image first
                self.click(sel, 200 if sel == "#run-compare" else 450)
                rec["clicked"] = round(self.now(), 3)
                if sel == "#run-compare":
                    ok = self.wait_js("() => (document.getElementById('cmp-status').textContent || '').startsWith('Done')", end - 0.25)
                    rec["compare_done"] = round(self.now(), 3)
                    rec["notes"].append("compare Done" if ok else "compare NOT done by cue end")
                    if not ok:
                        log(f"WARNING {self.name}: comparison not done by {end - 0.25:.2f} s")
                    else:
                        x, y, *_ = self.js("() => __demo.center('#cmp-rows')")
                        self.js("([x, y]) => __demo.cursorTo(PARKX, y + 40, 700)", [x, y])
                elif scr.get("click") == "challenge-region":
                    ok = self.wait_js("""() => { const i = document.getElementById('ch-img');
                                          return i.complete && i.naturalWidth > 0 && i.src.indexOf('crop.jpg') < 0; }""", start + 2.6)
                    rec["region_image_at"] = round(self.now(), 3)
                    rec["notes"].append(f"region image loaded at {self.now():.2f}s" if ok else "region image NOT loaded")
                    if not ok:
                        log(f"WARNING {self.name}: blurred region image not loaded in time")
                    self.js("() => __demo.cursorTo(PARKX, 520, 900)")
            th = scr.get("then")
            if th:
                self.until(start + float(th.get("at", 0)))
                self.js("() => __demo.ringOff()")
                if th.get("action") == "ring":   # move the highlight inside the same view and point at it
                    self.pg.wait_for_timeout(150)
                    self.js("(s) => __demo.ring(s)", th["target"])
                    self.js("""(s) => { const e = document.querySelector(s); const li = e.querySelector('li') || e;
                                const r = li.getBoundingClientRect();
                                return __demo.cursorTo(r.left + Math.min(r.width * 0.55, 420), r.top + r.height / 2 + 6, 900); }""", th["target"])
                else:
                    self.js("() => __demo.cursorTo(PARKX, 640, 500)")
                    self.scroll(th["target"])
        elif act == "drop":
            self.until(start + 0.15)
            res = self.js("([b, m, n, s]) => __demo.dragDrop(b, m, n, s)",
                          [judge["b64"], judge["mime"], judge["name"], judge["size"]])
            rec["notes"].append(f"thumb visible={res['thumbVisible']} file={res['name']}")
            if not res["thumbVisible"]:
                log(f"WARNING {self.name}: drop thumbnail not visible")
            self.js("() => __demo.cursorTo(PARKX, 560, 900)")
        elif act == "analyze":
            self.click(scr.get("target", "#analyze"), 500)
            rec["clicked"] = round(self.now(), 3)
            ok = self.wait_js("() => !document.getElementById('verdict').hidden", end + 4)
            rec["result_at"] = round(self.now(), 3)
            rec["notes"].append(f"verdict at {rec['result_at']:.2f}s" if ok else "verdict MISSING")
            self.js("() => __demo.cursorTo(PARKX, 640, 700)")
        else:
            log(f"WARNING: unknown app action {act!r}")


def record_minute(p, key: str, cues: list[dict], args, judge: dict | None, raw_dir: Path) -> tuple[Path, list[dict]]:
    mark = mark_svg()
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="light",
                              record_video_dir=str(raw_dir), record_video_size={"width": W, "height": H})
    pg = ctx.new_page()
    pg.goto(args.url, wait_until="load")
    pg.wait_for_selector("#ready.on", timeout=90000)
    pg.add_style_tag(content=APP_ZOOM_CSS.format(z=args.zoom))
    # result images load as soon as a result arrives, not when scrolled into view (no half-loaded images on camera)
    pg.evaluate("() => document.querySelectorAll('img[loading=\"lazy\"]').forEach(i => i.loading = 'eager')")
    if key == "minute2":
        # the app serves result images with Cache-Control: no-store, so the browser re-downloads a blurred variant
        # (about 2 s from asia-south1) every time it is selected. Keep each one in memory after its first load.
        cache: dict[str, tuple[dict, bytes]] = {}

        def serve(route):
            u = route.request.url
            if u in cache:
                route.fulfill(status=200, headers=cache[u][0], body=cache[u][1])
                return
            r = route.fetch()
            body = r.body()
            if r.status == 200:
                cache[u] = ({"content-type": r.headers.get("content-type", "image/jpeg"), "cache-control": "no-store"}, body)
            route.fulfill(response=r, body=body)
        pg.route("**/request-image/**", serve)
    ready_title = pg.evaluate("document.getElementById('ready').title")
    log(f"{key}: page ready ({ready_title})")
    pg.evaluate(DEMO_JS, mark)
    m = Minute(pg, cues, key)

    # off-camera preparation
    if key == "minute2":
        examples = {scr.get("example") for scr in (c["screen"] for c in cues) if scr.get("example")}
        ex = sorted(examples)[0] if examples else "audi_r8"
        pg.click(f'button.ex[data-ex="{ex}"]')
        pg.wait_for_function("() => !document.getElementById('verdict').hidden", timeout=60000)
        if any(c["screen"].get("click") == "challenge-region" for c in cues):
            # run the challenge off-camera, so no spinner is ever filmed; load every blurred variant once (browser
            # cache), remember the region that cut the margin most, then show the original crop again
            t = time.perf_counter()
            pg.click("#run-challenge")
            pg.wait_for_function("() => !document.getElementById('ch-result').hidden", timeout=60000)
            log(f"{key}: challenge run off-camera in {time.perf_counter() - t:.2f} s")
            best = pg.evaluate("""() => { const b = [...document.querySelectorAll('#ch-buttons button')];
                const d = b.map(x => parseFloat((x.getAttribute('aria-label').match(/([+-]?[0-9.]+)$/) || [0, 0])[1]));
                return d.indexOf(Math.max(...d)); }""")
            n = pg.evaluate("() => document.querySelectorAll('#ch-buttons button').length")
            for k in range(n):
                pg.click(f"#ch-buttons button:nth-child({k + 1})")
                pg.wait_for_function("() => { const i = document.getElementById('ch-img'); return i.complete && i.naturalWidth > 0; }", timeout=30000)
            pg.click("#ch-restore")
            pg.wait_for_function("() => { const i = document.getElementById('ch-img'); return i.complete && i.naturalWidth > 0; }", timeout=30000)
            pg.evaluate("(k) => { window.__chBest = k; }", best)
            summary = pg.evaluate("document.getElementById('ch-summary').textContent")
            log(f"{key}: {n} blurred variants preloaded; strongest region = button {best + 1}; summary: {summary!r}")
        pg.wait_for_timeout(1200)
        vt = pg.evaluate("document.getElementById('v-title').textContent")
        log(f"{key}: example {ex} analysed off-camera: {vt!r}")
        pg.evaluate("() => window.scrollTo(0, document.getElementById('verdict').getBoundingClientRect().top + scrollY - 40)")
        pg.evaluate("(t) => __demo.credit(t)", example_credit(ex))
    else:
        # one warm-up upload of the same photo (off-camera), then back to the home state
        pg.evaluate("([b, m, n, s]) => __demo.dragDrop(b, m, n, s)", [judge["b64"], judge["mime"], judge["name"], judge["size"]])
        t = time.perf_counter()
        pg.click("#analyze")
        pg.wait_for_function("() => !document.getElementById('verdict').hidden", timeout=60000)
        st = pg.evaluate("""() => ({ title: document.getElementById('v-title').textContent,
                                     pp: document.getElementById('pp-state').textContent,
                                     cand: document.getElementById('cand-h').textContent })""")
        log(f"{key}: warm-up analysis took {time.perf_counter() - t:.2f} s: {st['title']!r}; passport: {st['pp']!r}; {st['cand']!r}")
        if not re.search(r"\b1 category\b", st["cand"]) or "not verified" in st["title"].lower():
            log(f"WARNING {key}: this photo does not give an identified, one-category result. The minute-3 lines "
                f"'The verdict reads the make, model and year' and 'Here the candidate set holds one category' "
                f"would not match the screen: pick another photo or change those lines in cues.json.")
        pg.click("#reset")
        pg.wait_for_timeout(1200)
        pg.evaluate("() => window.scrollTo(0, 0)")
        pg.mouse.move(5, 5)
        pg.evaluate("() => __demo.cursorJump(-80, 600)")
        pg.evaluate("(t) => __demo.credit(t)", judge["credit"])

    for i, c in enumerate(cues):
        scr = c["screen"]
        if scr["type"] == "card":
            fine = ""
            jc = judge["credit"].replace("Photo: ", "") if judge and judge.get("credit") else ""
            if any("{judge_credit}" in b for b in scr.get("bullets", [])):
                scr = dict(scr, bullets=[b.replace("{judge_credit}", jc or "see judge_inputs/CREDITS.txt")
                                         for b in scr["bullets"]])
            elif key == "minute3" and i == len(cues) - 1 and jc:
                fine = "Uploaded photo — " + jc
            pg.evaluate("([id, h]) => __demo.prepare(id, h)", [f"s{i}", card_html(scr, c["speaker"], mark, fine)])
        elif scr["type"] == "figure":
            pg.evaluate("([id, h]) => __demo.prepare(id, h)", [f"s{i}", figure_html(scr, mark)])
    pg.wait_for_timeout(600)
    pg.evaluate("() => __demo.cursorShow(false)")

    # sync marker, then t = 0
    pg.evaluate("() => __demo.marker(true)")
    first = cues[0]["screen"]["type"]
    if first in ("card", "figure"):
        pg.evaluate("() => __demo.show('s0', true)")
        pg.evaluate("() => __demo.replay('s0')")  # restart animations at t = 0 below
        m.overlay = True
    pg.wait_for_timeout(900)
    pg.evaluate("() => { __demo.marker(false); const s = document.querySelector('#dm-root .dm-scr.in'); if (s) { s.classList.remove('play'); s.offsetHeight; s.classList.add('play'); } }")
    m.t0 = time.perf_counter()
    m.run(judge)
    m.until(SECONDS + 1.6)    # tail so the cut always has 1800 frames
    video = pg.video
    ctx.close()
    browser.close()
    raw = Path(video.path())
    dst = raw_dir / f"{key}_raw.webm"
    if dst.exists():
        dst.unlink()
    raw.rename(dst)
    return dst, m.timing


# ----------------------------------------------------------------------------------------------- ffmpeg
def find_sync_frame(ffmpeg: Path, raw: Path) -> int:
    cmd = [str(ffmpeg), "-v", "error", "-i", str(raw), "-vf", f"fps={FPS},scale=16:9:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    fs = 16 * 9 * 3
    n = len(data) // fs
    mag = []
    for k in range(n):
        f = data[k * fs:(k + 1) * fs]
        r = sum(f[0::3]) / 144; g = sum(f[1::3]) / 144; b = sum(f[2::3]) / 144
        mag.append(r > 200 and g < 70 and b > 200)
    try:
        first = mag.index(True)
    except ValueError:
        raise RuntimeError(f"sync marker not found in {raw}")
    last = first
    while last + 1 < n and mag[last + 1]:
        last += 1
    start = last + 1
    log(f"sync marker in {raw.name}: frames {first}-{last} (of {n}); t=0 is frame {start} ({start / FPS:.3f} s)")
    if n < start + FRAMES:
        raise RuntimeError(f"raw video too short: {n} frames, need {start + FRAMES}")
    return start


def encode(ffmpeg: Path, raw: Path, start: int, out: Path, ass: Path | None = None) -> None:
    vf = f"fps={FPS},trim=start_frame={start}:end_frame={start + FRAMES},setpts=PTS-STARTPTS"
    cwd = None
    if ass is not None:
        vf += f",subtitles={ass.name}"
        cwd = str(ass.parent)
    cmd = [str(ffmpeg), "-y", "-v", "error", "-i", str(raw), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "slow",
           "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS), "-profile:v", "high", "-movflags", "+faststart",
           "-frames:v", str(FRAMES), str(out)]
    subprocess.run(cmd, check=True, cwd=cwd)


def probe(ffmpeg: Path, f: Path) -> dict:
    ffprobe = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix == ".exe" else "ffprobe")
    out = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_packets:format=duration",
                          "-of", "json", str(f)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    a = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(f)],
                       capture_output=True, text=True).stdout.strip()
    return {"codec": s["codec_name"], "w": s["width"], "h": s["height"], "pix_fmt": s["pix_fmt"], "fps": s["r_frame_rate"],
            "frames": int(s["nb_read_packets"]), "duration": float(j["format"]["duration"]), "audio": bool(a)}


def stills(ffmpeg: Path, clean: Path, capd: Path, cues: list[dict], key: str, check: Path) -> None:
    check.mkdir(parents=True, exist_ok=True)
    for f in check.glob(f"{key}_*.png"):
        f.unlink()
    t = 2.5
    while t < SECONDS:
        subprocess.run([str(ffmpeg), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(capd), "-frames:v", "1",
                        "-vf", "scale=960:-1", str(check / f"{key}_t{t:04.1f}.png")], check=True)
        t += 5.0
    for i, c in enumerate(cues):
        te = max(c["start"] + 0.2, c["end"] - 0.25)
        subprocess.run([str(ffmpeg), "-y", "-v", "error", "-ss", f"{te:.2f}", "-i", str(clean), "-frames:v", "1",
                        "-vf", "scale=960:-1", str(check / f"{key}_cue{i + 1:02d}_end{te:04.1f}.png")], check=True)


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="https://semevia.web.app")
    ap.add_argument("--cues", type=Path, default=DEMO_ROOT / "script" / "cues.json")
    ap.add_argument("--srt-dir", type=Path, default=None)
    ap.add_argument("--images", type=Path, default=DEMO_ROOT / "judge_inputs")
    ap.add_argument("--out", type=Path, default=DEMO_ROOT / "recordings")
    ap.add_argument("--only", choices=["minute2", "minute3", "both"], default="both")
    ap.add_argument("--ffmpeg", type=Path, default=FFMPEG_DEFAULT if FFMPEG_DEFAULT.exists() else Path(shutil.which("ffmpeg") or "ffmpeg"))
    ap.add_argument("--check", action="store_true", help="write verification stills to <out>/check")
    ap.add_argument("--zoom", type=float, default=1.3, help="CSS zoom for the app (default 1.3)")
    args = ap.parse_args()

    cues_all = json.loads(args.cues.read_text(encoding="utf-8"))
    srt_dir = args.srt_dir or args.cues.parent
    out = args.out
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    keys = ["minute2", "minute3"] if args.only == "both" else [args.only]
    for k in keys:
        cs = cues_all[k]
        assert abs(cs[0]["start"]) < 1e-9 and abs(cs[-1]["end"] - SECONDS) < 1e-9, f"{k}: cues must cover 0-60 s"
        for a, b in zip(cs, cs[1:]):
            assert abs(a["end"] - b["start"]) < 1e-9, f"{k}: cue gap at {a['end']}"

    judge = None
    if "minute3" in keys:
        named = next((c["screen"].get("image") for c in cues_all["minute3"] if c["screen"].get("action") == "drop"), None)
        f, credit = pick_judge_image(args.images, named if named and named != "judge" else None)
        data = f.read_bytes()
        mime = mimetypes.guess_type(f.name)[0] or "image/jpeg"
        judge = {"b64": base64.b64encode(data).decode(), "mime": mime, "name": "unseen_photo" + f.suffix.lower(),
                 "size": f"{len(data) / 1048576:.1f} MB · image", "credit": credit, "path": str(f)}

    results = {}
    with sync_playwright() as p:
        for k in keys:
            log(f"=== recording {k} ===")
            raw, timing = record_minute(p, k, cues_all[k], args, judge, raw_dir)
            (out / f"{k}_timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
            results[k] = (raw, timing)

    for k in keys:
        raw, _ = results[k]
        start = find_sync_frame(args.ffmpeg, raw)
        clean = out / f"{k}_clean.mp4"
        capd = out / f"{k}_captioned.mp4"
        srt_src = srt_dir / f"captions_{k}.srt"
        srt_dst = out / f"captions_{k}.srt"
        shutil.copyfile(srt_src, srt_dst)
        ass = out / f"captions_{k}.ass"
        n = srt_to_ass(srt_src, ass)
        log(f"{k}: {n} caption events -> {ass.name}")
        encode(args.ffmpeg, raw, start, clean)
        encode(args.ffmpeg, raw, start, capd, ass)
        for f in (clean, capd):
            info = probe(args.ffmpeg, f)
            ok = (info["w"], info["h"]) == (W, H) and info["fps"] == f"{FPS}/1" and abs(info["duration"] - SECONDS) <= 0.15 \
                and info["pix_fmt"] == "yuv420p" and info["codec"] == "h264"
            log(f"{'OK ' if ok else 'BAD'} {f.name}: {info}")
        if args.check:
            stills(args.ffmpeg, clean, capd, cues_all[k], k, out / "check")
            log(f"{k}: stills in {out / 'check'}")
    with open(out / "record_log.txt", "a", encoding="utf-8") as fh:   # appended, one block per run
        fh.write("\n".join(LOG_LINES) + "\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
