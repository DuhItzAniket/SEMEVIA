# Team and ownership

Team **T35**, problem statement **PS5** (Computer Vision Hackathon 2026).

| Member | USN | Lane | Presents in the viva |
|---|---|---|---|
| Aniket Kumar | 1RUA24CSE0051 | **A: model, data, evaluation** | Data audit and partitions, classical features, EfficientNet-B0 fine-tuning (r1 and r2), heads, hybrid fusion and ablations, freeze and calibration runs, metrics, robustness, failure analysis, optimization |
| C Lakshanaa | 1RUA24CSE0097 | **B: application, deployment, documentation** | Upload and decoding, detector adapter and target rule, web service and UI, Docker image, release manifest, public URL and fallback, demo, README and design documents, ethics and privacy |
| Both | | **Shared: Identity Evidence Passport** | Conformal candidate set, decision states, hierarchy, Evidence Challenge, references |

Each member should be able to give the one-sentence answer for the other lane too; the detailed answers are in
[viva_prep.md](viva_prep.md), and the demo split is in [demo.md](demo.md).

**AI assistance.** An AI coding agent (Claude Code) helped write the implementation and documents under the
team's direction. The team chose the approach, ran the experiments, reviewed the results and is responsible for
explaining every lane. The viewpoint labels and their blind second pass were also made by AI agents
(`reports/viewpoint_labels.csv`, `reports/viewpoint_label_check.json`); no human relabelled them.

## Files per lane

| Area | Lane | Files |
|---|---|---|
| Data manifests, metadata, coordinates | A | `src/data/`, `src/vision/coordinates.py`, `scripts/prepare_data.py`, `scripts/declare_subsets.py`, `configs/data.yaml`, `artifacts/manifests/`, `artifacts/class_metadata.json` |
| Training | A | `src/models/backbone.py`, `src/models/training.py`, `scripts/train.py`, `configs/train.yaml` (r1), `configs/train_v2.yaml` (r2) |
| Classical features and proposals | A | `src/vision/features.py`, `src/vision/proposals.py`, `configs/features.yaml` |
| Feature caches, heads, inference | A | `src/models/cache_io.py`, `src/models/heads.py`, `src/models/inference.py`, `scripts/cache_features.py`, `scripts/fit_heads.py`, `configs/heads.yaml` |
| Freeze and calibration runs | A | `scripts/freeze.py`, `scripts/calibrate.py`, `configs/calibration.yaml`, `artifacts/production_freeze.json`, `artifacts/calibration_final.json` |
| Evaluation and report | A | `src/evaluation/`, `scripts/evaluate.py`, `scripts/evaluate_detector.py`, `scripts/evaluate_negatives.py`, `scripts/analyze_results.py`, `scripts/report_robustness.py`, `scripts/report.py`, `configs/evaluation.yaml`, `reports/`, `eval_P5_T35.md` (generated) |
| Optimization | A | `src/models/export_onnx.py`, `scripts/optimize.py`, `scripts/optimize_head.py` |
| Negative suite | A collects and runs, B documents | `artifacts/negative_manifests/`, `docs/negative_suite.md` |
| Passport (conformal, hierarchy, decision, challenge, references) | Shared, one editor at a time | `src/passport/`, `scripts/build_references.py` |
| Upload decoding, detector adapter | B | `src/vision/decode.py`, `src/models/detector.py` |
| Web service and UI | B | `app/main.py`, `app/static/`, `configs/runtime.yaml`, `docs/api_schema.json` |
| Release bundle and container | B | `src/release.py`, `scripts/promote_bundle.py`, `artifacts/release_manifest.json`, `artifacts/release_config.json`, `Dockerfile`, `compose.yaml`, `requirements-infer.txt`, `scripts/check_container.py` |
| Public deployment | B | `scripts/check_release.py`, `scripts/prepare_hf_space.py`, `docs/hosting_lock.md`, `reports/release_acceptance.json` |
| Documentation | B (A reviews the model sections) | `README.md`, `docs/architecture.md`, `docs/design.md`, `docs/timeline.md`, `docs/demo.md`, `docs/viva_prep.md`, `docs/runbook_other_pc.md`, `docs/training_handoff.md`, `NEXT.md` |
| Tests | Owner of the code under test | `tests/` |

## Contracts both lanes rely on

Frozen early and not changed without both members agreeing: the response contract `docs/api_schema.json`,
the class-ID convention (`class_id = original - 1`), the canonical box format (`configs/data.yaml: coordinates`),
the split seed, and artifact naming (`<artifact>_<version>_<sha256[:12]>`).
