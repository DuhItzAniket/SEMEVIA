"""Release artifact manifest: fetch (build time) and verify (readiness).

Usage at build time:  python -m src.release fetch
Usage for checks:     python -m src.release verify
Runtime never downloads; readiness fails if any listed asset is missing or has a wrong hash.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "release_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(manifest: dict | None = None) -> list[str]:
    """Return a list of problems (empty when every asset exists with the expected checksum)."""
    manifest = manifest or load_manifest()
    problems = []
    for a in manifest["assets"]:
        p = ROOT / a["path"]
        if not p.is_file():
            problems.append(f"missing asset {a['name']} at {a['path']}")
        elif sha256_file(p) != a["sha256"]:
            problems.append(f"checksum mismatch for {a['name']}")
    return problems


def fetch(manifest: dict | None = None) -> None:
    manifest = manifest or load_manifest()
    for a in manifest["assets"]:
        p = ROOT / a["path"]
        if p.is_file() and sha256_file(p) == a["sha256"]:
            print(f"ok (present) {a['name']}")
            continue
        if not a.get("url"):
            raise SystemExit(f"asset {a['name']} missing locally and has no public url")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        print(f"fetching {a['name']} from {a['url']}")
        urllib.request.urlretrieve(a["url"], tmp)
        got = sha256_file(tmp)
        if got != a["sha256"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"checksum mismatch for {a['name']}: {got}")
        tmp.replace(p)
        print(f"ok (fetched) {a['name']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "fetch":
        fetch()
    else:
        probs = verify()
        print("\n".join(probs) if probs else "all release assets verified")
        sys.exit(1 if probs else 0)
