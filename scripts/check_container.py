"""Record local container checks (step 8) in reports/container_check.json.

Usage (Docker running, image semevia:local built):
  python scripts/check_container.py
Starts nothing itself except a temporary `--network none` container; expects `docker compose up -d`
to be serving on localhost:8000. Every value written comes from a live response or docker inspect.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.hardware import hardware_label  # noqa: E402

DATA = Path(__import__("os").environ.get("SEMEVIA_DATA_ROOT", ""))


def sh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True)


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def post_image(path: Path, mode: str) -> tuple[dict, float]:
    boundary = "semeviacheck"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{mode}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request("http://localhost:8000/predict", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t) * 1000


def main() -> None:
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "image": "semevia:local",
              "image_size": sh("docker", "images", "semevia:local", "--format", "{{.Size}}").stdout.strip(),
              "host": hardware_label(), "note": "local Docker Desktop checks; not the public-host release acceptance"}
    report["online_readyz"] = get("http://localhost:8000/readyz")
    imgs = sorted((DATA / "cars_test").glob("*.jpg"))[:5] if DATA.is_dir() else []
    rows = []
    for p in imgs:
        out, ms = post_image(p, "hybrid")
        rows.append({"file": f"cars_test/{p.name}", "status": out.get("status"), "ms_client": round(ms, 1),
                     "detection_state": (out.get("detection") or {}).get("state")})
    report["predict_samples"] = rows
    name = "semevia_offline_check"
    sh("docker", "container", "rm", "--force", name)
    sh("docker", "run", "-d", "--name", name, "--network", "none", "semevia:local")
    ready = None
    for _ in range(60):
        r = sh("docker", "exec", name, "curl", "-fsS", "http://localhost:8000/readyz")
        if r.returncode == 0 and '"ready":true' in r.stdout:
            ready = json.loads(r.stdout)
            break
        time.sleep(3)
    outbound = sh("docker", "exec", name, "curl", "-sS", "-m", "5", "https://github.com", "-o", "/dev/null")
    report["offline"] = {"network_mode": sh("docker", "inspect", name, "--format", "{{.HostConfig.NetworkMode}}").stdout.strip(),
                         "readyz": ready, "outbound_curl_exit_code": outbound.returncode,
                         "outbound_error": outbound.stderr.strip()}
    sh("docker", "container", "rm", "--force", name)
    ok = (report["online_readyz"].get("ready") and report["online_readyz"].get("recognition") and ready
          and ready.get("recognition") and outbound.returncode != 0
          and all(r["status"] in ("ok", "uncertain") and r["detection_state"] == "detected" for r in rows))
    report["passed"] = bool(ok)
    (ROOT / "reports/container_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": report["passed"], "samples": rows, "offline_ready": bool(ready)}, indent=1))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
