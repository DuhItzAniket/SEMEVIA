"""Release acceptance checks against a public URL; writes reports/release_acceptance.json.

Usage: python scripts/check_release.py --url https://<public-host> [--n 20] [--concurrent 4]

Every value is measured from live HTTP responses (client-side wall time, so it includes the network path).
Images come from the Stanford Cars test folder (SEMEVIA_DATA_ROOT); bad inputs are generated here.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "semevia-release-check/1.0"}


def request(url: str, data: bytes | None = None, ctype: str | None = None) -> tuple[int, bytes, float]:
    headers = dict(UA)
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=headers)
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body, code = r.read(), r.status
    except urllib.error.HTTPError as e:
        body, code = e.read(), e.code
    return code, body, (time.perf_counter() - t) * 1000


def multipart(name: str, payload: bytes, ctype: str, mode: str = "hybrid") -> tuple[bytes, str]:
    b = "semeviarelease"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{mode}\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + payload + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def predict(base: str, name: str, payload: bytes, ctype: str = "image/jpeg") -> dict:
    body, ct = multipart(name, payload, ctype)
    code, out, ms = request(f"{base}/predict", body, ct)
    try:
        js = json.loads(out)
    except ValueError:
        js = {}
    err = js.get("error") or {}
    return {"file": name, "http": code, "status": js.get("status"), "error_code": err.get("code"),
            "message": err.get("message"), "ms": round(ms, 1),
            "server_total_ms": (js.get("timings") or {}).get("total_ms")}


def pctl(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--host-note", default="")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    data = Path(os.environ.get("SEMEVIA_DATA_ROOT", "")) / "cars_test"
    imgs = sorted(data.glob("*.jpg"))
    if len(imgs) < args.n + args.concurrent + 100:
        raise SystemExit("set SEMEVIA_DATA_ROOT to the Stanford Cars data (cars_test needed)")
    pick = imgs[100:100 + args.n]                      # not the images used by check_container.py
    rep = {"url": base, "created_at": datetime.now(timezone.utc).isoformat(), "host_note": args.host_note,
           "client": "this PC over its internet connection; times are client wall time incl. network"}

    code, body, ms = request(f"{base}/")
    rep["page"] = {"http": code, "ms": round(ms, 1), "bytes": len(body), "has_viewport_meta": b"name=\"viewport\"" in body}
    code, body, ms = request(f"{base}/readyz")
    rep["readyz"] = {"http": code, "ms": round(ms, 1), **(json.loads(body) if code == 200 else {})}

    rows = [predict(base, f"cars_test/{p.name}", p.read_bytes()) for p in pick]
    lat = [r["ms"] for r in rows if r["http"] == 200]
    rep["fresh_uploads"] = {"count": len(rows), "ok_http": sum(r["http"] == 200 for r in rows),
                            "statuses": {s: sum(r["status"] == s for r in rows) for s in sorted({str(r["status"]) for r in rows})},
                            "p50_ms": pctl(lat, 0.5) if lat else None, "p95_ms": pctl(lat, 0.95) if lat else None,
                            "max_ms": max(lat) if lat else None, "rows": rows}

    conc = imgs[100 + args.n:100 + args.n + args.concurrent]
    t = time.perf_counter()
    with ThreadPoolExecutor(args.concurrent) as ex:
        crow = list(ex.map(lambda p: predict(base, f"cars_test/{p.name}", p.read_bytes()), conc))
    rep["concurrent"] = {"clients": args.concurrent, "wall_ms": round((time.perf_counter() - t) * 1000, 1),
                         "http": [r["http"] for r in crow], "statuses": [r["status"] for r in crow],
                         "ms": [r["ms"] for r in crow]}

    from PIL import Image

    bio = io.BytesIO()
    Image.new("RGB", (4000, 3000), (120, 120, 120)).save(bio, "PNG")
    noise = os.urandom(11 * 1024 * 1024)
    jpg = pick[0].read_bytes()
    bad = {"empty_file": predict(base, "empty.jpg", b""),
           "text_file": predict(base, "notes.txt", b"this is not an image\n" * 20, "text/plain"),
           "truncated_jpeg": predict(base, "broken.jpg", jpg[: len(jpg) // 3]),
           "random_bytes_named_jpg": predict(base, "random.jpg", os.urandom(50_000)),
           "over_10_mb": predict(base, "big.jpg", noise),
           "no_car_flat_image": predict(base, "flat.png", bio.getvalue(), "image/png")}
    rep["bad_inputs"] = bad
    handled = all(v["http"] in (200, 400, 413, 415, 422) and v["status"] in ("invalid_image", "no_vehicle", "uncertain", "ok")
                  for v in bad.values())
    rep["checks"] = {
        "page_ok": rep["page"]["http"] == 200,
        "ready_with_recognition": bool(rep["readyz"].get("ready") and rep["readyz"].get("recognition")),
        "all_fresh_uploads_answered": rep["fresh_uploads"]["ok_http"] == len(rows),
        "p95_under_10s": bool(lat) and pctl(lat, 0.95) < 10_000,
        "concurrent_all_answered": all(c in (200, 503) for c in rep["concurrent"]["http"]),
        "bad_inputs_handled_with_typed_errors": handled,
    }
    rep["passed"] = all(rep["checks"].values())
    (ROOT / "reports/release_acceptance.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rep[k] for k in ("checks", "passed")}, indent=1))
    print(json.dumps({k: rep["fresh_uploads"][k] for k in ("count", "statuses", "p50_ms", "p95_ms", "max_ms")}))
    print(json.dumps(rep["concurrent"]))
    print(json.dumps({k: (v["http"], v["status"], v["error_code"]) for k, v in bad.items()}))
    if not rep["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
