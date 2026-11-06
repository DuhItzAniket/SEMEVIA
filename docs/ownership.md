# Module ownership

The plan assigns two lanes. In this implementation run a single coding agent works through the
lanes sequentially, so there is no simultaneous editing; the ownership table still defines which
files a task card may change.

| Area | Lane | Files |
|---|---|---|
| Data manifests, metadata, coordinates | Model/data (A) | `src/data/`, `src/vision/coordinates.py`, `scripts/prepare_data.py`, `configs/data.yaml`, `artifacts/manifests/` |
| Training | A | `src/models/backbone.py`, `src/models/training.py`, `scripts/train.py`, `configs/train.yaml` |
| Classical features and proposals | A | `src/vision/features.py`, `src/vision/proposals.py` |
| Heads, caches, inference | A | `src/models/heads.py`, `src/models/inference.py`, `scripts/cache_features.py`, `scripts/fit_heads.py` |
| Passport (conformal, hierarchy, challenge) | Shared, one editor at a time | `src/passport/` |
| Evaluation | A computes, B presents | `src/evaluation/`, `scripts/evaluate.py`, `scripts/report.py` |
| Detector adapter, decode | Application (B) | `src/models/detector.py`, `src/vision/decode.py` |
| Web service and UI | B | `app/` |
| Docker and deployment | B | `Dockerfile`, `compose.yaml`, `.dockerignore`, `artifacts/release_manifest.json` |
| Handoff | B | `NEXT.md`, phase tags |

Contracts frozen in P0: `docs/api_schema.json` (response fields and statuses), class-ID convention
(`class_id = original - 1`), canonical box format (`configs/data.yaml: coordinates`), split seed,
and artifact naming (`<artifact>_<version>_<sha256[:12]>`).
