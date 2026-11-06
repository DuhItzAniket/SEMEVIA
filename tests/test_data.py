import json
from pathlib import Path

import pytest

from src.data.metadata import parse_class_name

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "artifacts" / "class_metadata.json"
INDEX = ROOT / "artifacts" / "manifests" / "manifest_index.json"


def test_parse_multiword_makes():
    a = parse_class_name("AM General Hummer SUV 2000", 0)
    assert (a.make, a.model, a.year, a.family) == ("AM General", "Hummer SUV", 2000, "AM General Hummer SUV")
    b = parse_class_name("Aston Martin V8 Vantage Convertible 2012", 5)
    assert b.make == "Aston Martin" and b.original_class_id == 6
    c = parse_class_name("Land Rover Range Rover SUV 2012", 9)
    assert c.make == "Land Rover"


def test_parse_rejects_missing_year():
    with pytest.raises(ValueError):
        parse_class_name("Foo Bar", 0)


@pytest.mark.skipif(not META.exists(), reason="run scripts/prepare_data.py first")
def test_all_196_metadata_rows():
    meta = json.loads(META.read_text(encoding="utf-8"))
    classes = meta["classes"]
    assert len(classes) == 196
    assert [c["class_id"] for c in classes] == list(range(196))
    assert all(c["original_class_id"] == c["class_id"] + 1 for c in classes)
    assert meta["num_makes"] == 49
    assert all(1990 <= c["year"] <= 2012 for c in classes)
    assert all(c["family"].startswith(c["make"] + " ") for c in classes)
    by_orig = {c["original_class_id"]: c for c in classes}
    # Plan fixtures (original one-based ids)
    assert by_orig[34]["name"] == "BMW M3 Coupe 2012"
    assert by_orig[23]["family"] == by_orig[24]["family"] == "Audi S4 Sedan"
    assert by_orig[42]["family"] == by_orig[43]["family"] == "Bentley Continental GT Coupe"
    assert by_orig[21]["family"] != by_orig[22]["family"]          # S5 Convertible vs Coupe kept apart
    assert by_orig[21]["make"] == by_orig[22]["make"] == "Audi"


@pytest.mark.skipif(not INDEX.exists(), reason="run scripts/prepare_data.py first")
def test_manifest_partitions_disjoint_and_counted():
    from src.data.manifest import read_manifest

    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    parts = {p: read_manifest(INDEX.parent / v["file"]) for p, v in idx["partitions"].items()}
    assert len(parts["test"]) == 8041
    assert len(parts["calibration"]) == 815 and len(parts["tune"]) == 815
    assert len(parts["train"]) == 8144 - 815 - 815
    keys = [r.image_key for recs in parts.values() for r in recs]
    assert len(keys) == len(set(keys)) == 16185
    for p, recs in parts.items():
        assert all(r.partition == p for r in recs)
        assert all(r.image_key.startswith(r.source_split + "/") for r in recs)
    assert {r.source_split for r in parts["test"]} == {"test"}
    assert {r.source_split for p in ("train", "tune", "calibration") for r in parts[p]} == {"train"}
