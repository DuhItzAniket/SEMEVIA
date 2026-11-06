"""Verify Stanford Cars data and write frozen split manifests (P1A).

Usage: python scripts/prepare_data.py --config configs/data.yaml [--data-root PATH]
Exits non-zero when a required invariant fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.data.manifest import (  # noqa: E402
    assign_partitions, audit_path, load_annotations, load_class_names, make_record,
    sha256_file, validate_record, write_manifest,
)
from src.data.metadata import METADATA_VERSION, build_metadata  # noqa: E402


def fail(msg: str) -> None:
    print(f"INVARIANT FAILED: {msg}", file=sys.stderr)
    sys.exit(2)


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(8):
        c += np.unpackbits((x & np.uint64(0xFF)).astype(np.uint8)[..., None], axis=-1).sum(-1).astype(np.uint8)
        x = x >> np.uint64(8)
    return c


def near_duplicates(keys: list[str], hashes: list[int], thr: int) -> list[tuple[str, str, int]]:
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        d = popcount64(h[i + 1:] ^ h[i])
        for j in np.nonzero(d <= thr)[0]:
            pairs.append((keys[i], keys[i + 1 + j], int(d[j])))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    data_root = Path(args.data_root or os.environ.get(cfg["data_root_env"], ""))
    if not data_root.is_dir():
        fail(f"data root not found: {data_root} (set {cfg['data_root_env']} or --data-root)")
    t0 = time.time()
    audit: dict = {"manifest_version": cfg["manifest_version"], "source": cfg["source"]["mirror"],
                   "generated_unix": int(t0)}

    # 1. Metadata files and published checksums
    files = {
        "cars_meta.mat": data_root / "devkit" / "cars_meta.mat",
        "cars_train_annos.mat": data_root / "devkit" / "cars_train_annos.mat",
        "cars_test_annos.mat": data_root / "devkit" / "cars_test_annos.mat",
        "cars_test_annos_withlabels.mat": data_root / "cars_test_annos_withlabels.mat",
    }
    checks = {}
    for name, p in files.items():
        if not p.is_file():
            fail(f"missing {p}")
        sha = sha256_file(p)
        published = cfg["source"]["published_sha256"].get(name)
        checks[name] = {"sha256": sha, "bytes": p.stat().st_size, "published_sha256": published,
                        "matches_published": (sha == published) if published else None}
    audit["file_checksums"] = checks
    if checks["cars_test_annos_withlabels.mat"]["matches_published"] is not True:
        fail("labeled test annotation SHA256 does not match TFDS published value")

    # 2. Class metadata
    names = load_class_names(files["cars_meta.mat"])
    meta = build_metadata(names)
    makes = sorted({m.make for m in meta})
    fam_sizes = Counter(m.family for m in meta)
    meta_out = {"metadata_version": METADATA_VERSION, "num_classes": len(meta), "num_makes": len(makes),
                "makes": makes, "num_families": len(fam_sizes),
                "multi_member_families": {f: n for f, n in sorted(fam_sizes.items()) if n > 1},
                "classes": [m.to_dict() for m in meta]}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "class_metadata.json").write_text(json.dumps(meta_out, indent=1), encoding="utf-8")
    years = sorted({m.year for m in meta})
    audit["metadata"] = {"num_classes": len(meta), "num_makes": len(makes), "num_families": len(fam_sizes),
                         "years": years, "multi_member_families": len(meta_out["multi_member_families"])}

    # 3. Annotations
    train_rows = load_annotations(files["cars_train_annos.mat"])
    test_rows = load_annotations(files["cars_test_annos_withlabels.mat"])
    exp = cfg["expected"]
    if len(train_rows) != exp["train_records"] or len(test_rows) != exp["test_records"]:
        fail(f"record counts {len(train_rows)}/{len(test_rows)}")
    for split, rows in (("train", train_rows), ("test", test_rows)):
        if len({r["fname"] for r in rows}) != len(rows):
            fail(f"duplicate filenames in {split}")
        cls = Counter(r["class"] for r in rows)
        if set(cls) != set(range(1, 197)):
            fail(f"{split} class coverage")
        on_disk = {p.name for p in (data_root / f"cars_{split}").glob("*.jpg")}
        listed = {r["fname"] for r in rows}
        audit[f"{split}_coverage"] = {"annotated": len(listed), "on_disk": len(on_disk),
                                      "annotated_missing_on_disk": len(listed - on_disk),
                                      "on_disk_unannotated": len(on_disk - listed),
                                      "per_class_min": min(cls.values()), "per_class_max": max(cls.values())}
        if listed - on_disk:
            fail(f"{split}: {len(listed - on_disk)} annotated images missing")

    # 4. Decode every image (parallel)
    jobs = [(s, r) for s, rows in (("train", train_rows), ("test", test_rows)) for r in rows]
    paths = [str(data_root / f"cars_{s}" / r["fname"]) for s, r in jobs]
    print(f"auditing {len(paths)} images with {args.workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(audit_path, paths, chunksize=64))
    bad = [r for r in results if not r["ok"]]
    audit["decode_failures"] = bad
    if bad:
        fail(f"{len(bad)} images failed to decode")

    records, problems, edge = [], [], Counter()
    for (split, row), res in zip(jobs, results):
        rec = make_record(split, row, res, res["sha256"], METADATA_VERSION)
        p = validate_record(rec)
        if p:
            problems.append({"image_key": rec.image_key, "problems": p, "raw": row, "size": [res["width"], res["height"]]})
        edge["x1_eq_1"] += row["x1"] == 1
        edge["y1_eq_1"] += row["y1"] == 1
        edge["x1_eq_0"] += row["x1"] == 0
        edge["x2_eq_W"] += row["x2"] == res["width"]
        edge["y2_eq_H"] += row["y2"] == res["height"]
        edge["x2_gt_W"] += row["x2"] > res["width"]
        edge["y2_gt_H"] += row["y2"] > res["height"]
        edge["exif_rotated"] += res["exif_orientation"] != 1
        edge["non_rgb_mode"] += res["mode"] != "RGB"
        edge["near_uniform"] += res["std"] < 2.0
        records.append(rec)
    audit["box_edge_statistics"] = dict(edge)
    audit["exif_orientation_values"] = dict(Counter(res["exif_orientation"] for res in results))
    audit["exif_policy"] = ("Dataset images are used in their stored pixel frame; EXIF orientation is NOT applied. "
                            "Verified on the only tag-6 image (train/07389.jpg): stored pixels are upright and the "
                            "box matches them. Uploads do apply EXIF orientation.")
    audit["invalid_box_records"] = problems
    audit["modes"] = dict(Counter(r["mode"] for r in results))

    # 5. Duplicates
    by_sha: dict[str, list[str]] = {}
    for rec in records:
        by_sha.setdefault(rec.content_sha256, []).append(rec.image_key)
    exact = [v for v in by_sha.values() if len(v) > 1]
    keys = [rec.image_key for rec in records]
    near = near_duplicates(keys, [r["phash"] for r in results], cfg["audit"]["phash_hamming_threshold"])
    cross = [p for p in near if p[0].split("/")[0] != p[1].split("/")[0]]
    rec_by_key = {r.image_key: r for r in records}
    conflicts = [[(k, rec_by_key[k].class_id) for k in g] for g in exact
                 if len({rec_by_key[k].class_id for k in g}) > 1]
    audit["duplicates"] = {"exact_groups": len(exact), "exact_examples": exact[:50],
                           "exact_groups_with_conflicting_labels": len(conflicts),
                           "conflicting_label_groups": conflicts,
                           "test_images_with_exact_copy_outside_test": sorted(
                               {k for g in exact for k in g if k.startswith("test/")
                                and any(not o.startswith("test/") for o in g)}),
                           "phash_threshold": cfg["audit"]["phash_hamming_threshold"],
                           "near_pairs": len(near), "near_pairs_cross_split": len(cross),
                           "near_examples": near[:100],
                           "note": "Recorded only; official populations are not modified."}

    # 6. Partitions
    train_recs = [r for r in records if r.source_split == "train"]
    test_recs = [r for r in records if r.source_split == "test"]
    sp = cfg["splits"]
    assign_partitions(train_recs, sp["seed"], sp["calibration_size"], sp["tune_size"])
    parts = {p: [r for r in records if r.partition == p] for p in ("train", "tune", "calibration", "test")}
    keysets = {p: {r.image_key for r in v} for p, v in parts.items()}
    for a in keysets:
        for b in keysets:
            if a < b and keysets[a] & keysets[b]:
                fail(f"partitions {a} and {b} overlap")
    if sum(len(v) for v in keysets.values()) != len(records):
        fail("partition sizes do not sum to total")
    if len({r.class_id for r in parts["train"]}) != 196 or len({r.class_id for r in parts["tune"]}) != 196:
        fail("train/tune do not contain all 196 classes")
    content_overlap = {}
    shas = {p: {r.content_sha256 for r in v} for p, v in parts.items()}
    for a in shas:
        for b in shas:
            if a < b:
                content_overlap[f"{a}|{b}"] = len(shas[a] & shas[b])
    audit["partition_content_overlap_exact"] = content_overlap

    mdir = ROOT / "artifacts" / "manifests"
    index = {"manifest_version": cfg["manifest_version"], "metadata_version": METADATA_VERSION,
             "seed": sp["seed"], "bbox_format": records[0].bbox_format, "partitions": {}}
    for p, recs in parts.items():
        sha = write_manifest(recs, mdir / f"{p}_{cfg['manifest_version']}.csv")
        index["partitions"][p] = {"file": f"{p}_{cfg['manifest_version']}.csv", "count": len(recs), "sha256": sha,
                                  "classes": len({r.class_id for r in recs})}
    (mdir / "manifest_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    audit["partitions"] = index["partitions"]

    # 7. Overlay contact sheet (seeded sample of 30)
    rng = np.random.RandomState(sp["seed"])
    sample = [records[i] for i in rng.choice(len(records), cfg["audit"]["overlay_samples"], replace=False)]
    tiles = []
    for rec in sample:
        im = Image.open(data_root / rec.rel_path).convert("RGB")
        d = ImageDraw.Draw(im)
        l, t, r, b = rec.box
        d.rectangle([l, t, r - 1, b - 1], outline=(255, 0, 0), width=max(2, im.width // 200))
        im.thumbnail((240, 240))
        tiles.append(im)
    sheet = Image.new("RGB", (6 * 240, 5 * 240), "white")
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 6) * 240, (i // 6) * 240))
    fig = ROOT / "reports" / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    sheet.quantize(colors=128).save(fig / f"data_overlays_{cfg['manifest_version']}.png", optimize=True)
    audit["overlay_samples"] = [r.image_key for r in sample]

    # 8. Corner crops at native resolution for coordinate-convention inspection
    corner = [r for r in records if r.left == 0 or r.top == 0][:4] + sample[:2]
    tiles = []
    for rec in corner:
        im = Image.open(data_root / rec.rel_path).convert("RGB")
        l, t, r, b = rec.box
        for (cx, cy) in ((l, t), (r, b)):
            x0, y0 = int(max(0, cx - 12)), int(max(0, cy - 12))
            tile = im.crop((x0, y0, x0 + 24, y0 + 24)).resize((96, 96), Image.NEAREST)
            dd = ImageDraw.Draw(tile)
            px, py = (cx - x0) * 4, (cy - y0) * 4
            dd.line([px, 0, px, 96], fill=(255, 0, 0))
            dd.line([0, py, 96, py], fill=(255, 0, 0))
            tiles.append(tile)
    sheet = Image.new("RGB", (4 * 96, ((len(tiles) + 3) // 4) * 96), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * 96, (i // 4) * 96))
    sheet.save(fig / f"data_box_corners_{cfg['manifest_version']}.png")

    audit["elapsed_seconds"] = round(time.time() - t0, 1)
    rep = ROOT / "reports"
    (rep / "data_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("partitions", "box_edge_statistics", "metadata", "elapsed_seconds")}, indent=1))
    if problems:
        fail(f"{len(problems)} records with invalid boxes; see reports/data_audit.json")


if __name__ == "__main__":
    main()
