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


RELEASE_CONFIG = ROOT / "artifacts" / "release_config.json"
OVERRIDABLE = ("backbone", "backbone_tag", "backbone_status", "calibration", "precision", "encoder_runtime",
               "hybrid_head_runtime")


def recognition_config() -> dict:
    """configs/runtime.yaml recognition settings, overridden by artifacts/release_config.json (written by
    scripts/promote_bundle.py) for the backbone, its tag/status and the calibration file in this release."""
    import yaml

    rc = dict(yaml.safe_load((ROOT / "configs" / "runtime.yaml").read_text(encoding="utf-8"))["recognition"])
    if RELEASE_CONFIG.exists():
        rel = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
        rc.update({k: rel[k] for k in OVERRIDABLE if k in rel})
    return rc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def asset_status(manifest: dict | None = None) -> dict[str, str]:
    """name -> 'ok' | 'missing' | 'checksum_mismatch'."""
    manifest = manifest or load_manifest()
    out = {}
    for a in manifest["assets"]:
        p = ROOT / a["path"]
        if not p.is_file():
            out[a["name"]] = "missing"
        elif sha256_file(p) != a["sha256"]:
            out[a["name"]] = "checksum_mismatch"
        else:
            out[a["name"]] = "ok"
    return out


def verify(manifest: dict | None = None) -> list[str]:
    """Problems that must block readiness: any required asset missing, and any asset (required
    or optional) present with a wrong checksum. Missing optional assets only disable features."""
    manifest = manifest or load_manifest()
    status = asset_status(manifest)
    problems = []
    for a in manifest["assets"]:
        s = status[a["name"]]
        if s == "checksum_mismatch":
            problems.append(f"checksum mismatch for {a['name']}")
        elif s == "missing" and not a.get("optional", False):
            problems.append(f"missing asset {a['name']} at {a['path']}")
    return problems


def fetch(manifest: dict | None = None) -> None:
    manifest = manifest or load_manifest()
    for a in manifest["assets"]:
        p = ROOT / a["path"]
        if p.is_file() and sha256_file(p) == a["sha256"]:
            print(f"ok (present) {a['name']}")
            continue
        if a.get("derived_from"):
            continue   # built from another asset after the loop
        if not a.get("url"):
            if a.get("optional", False):
                print(f"skip (optional, no public url) {a['name']}")
                continue
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
    for a in manifest["assets"]:
        p = ROOT / a["path"]
        if a.get("derived_from") and not (p.is_file() and sha256_file(p) == a["sha256"]):
            if a.get("optional", False) and not _source_ok(manifest, a):
                print(f"skip (optional, source {a['derived_from']} not present) {a['name']}")
                continue
            got = _derive(manifest, a, p)
            if got != a["sha256"]:
                p.unlink(missing_ok=True)
                raise SystemExit(f"derived {a['name']} has sha256 {got}, manifest says {a['sha256']}")
            print(f"ok (derived from {a['derived_from']}) {a['name']}")


def _source_ok(manifest: dict, a: dict) -> bool:
    src = next((s for s in manifest["assets"] if s["name"] == a["derived_from"]), None)
    return src is not None and (ROOT / src["path"]).is_file() and sha256_file(ROOT / src["path"]) == src["sha256"]


def _derive(manifest: dict, a: dict, out: Path) -> str:
    """Rebuild a derived asset (the INT8 hybrid head) from its verified source asset; return the new file's sha256."""
    from src.models.head_onnx import build_int8

    src = next((s for s in manifest["assets"] if s["name"] == a["derived_from"]), None)
    if src is None or sha256_file(ROOT / src["path"]) != src["sha256"] or src["sha256"] != a.get("derived_from_sha256"):
        raise SystemExit(f"cannot derive {a['name']}: source {a['derived_from']} missing or not the recorded file")
    return build_int8(ROOT / src["path"], out)["sha256"]


def check_derived(manifest: dict | None = None) -> list[str]:
    """Regenerate every derived asset into a temporary file and compare with the manifest (proves the build is
    byte-identical on this platform even when the file is already present)."""
    import tempfile

    manifest = manifest or load_manifest()
    problems = []
    for a in manifest["assets"]:
        if a.get("derived_from"):
            if a.get("optional", False) and not _source_ok(manifest, a):
                print(f"check-derived {a['name']}: skipped (optional source {a['derived_from']} not present)")
                continue
            with tempfile.TemporaryDirectory() as td:
                got = _derive(manifest, a, Path(td) / a["name"])
            print(f"check-derived {a['name']}: rebuilt {got}, manifest {a['sha256']}")
            if got != a["sha256"]:
                problems.append(f"derived {a['name']} is not byte-identical on this platform")
    return problems


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "fetch":
        fetch()
    elif cmd == "check-derived":
        probs = check_derived()
        print("\n".join(probs) if probs else "derived assets reproduce exactly")
        sys.exit(1 if probs else 0)
    else:
        probs = verify()
        print("\n".join(probs) if probs else "all release assets verified")
        sys.exit(1 if probs else 0)
