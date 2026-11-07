# Report kit (for the team's own write-up)

The final report PDF must be **written by the team**: the faculty rule caps AI-written content at 30% and asks
for AI use to be declared. This page therefore gives **no ready-made prose**. It gives the structure, the
questions each section should answer, where every fact and number lives, and the figures. Write each section
in your own words; copy numbers only from the files named here (or from `eval_P5_T35.md`, which is generated
from them).

Team **Hypermetropia** (T35), PS5. Members: Aniket Kumar (1RUA24CSE0051), C Lakshanaa (1RUA24CSE0097).
Sections of each member: fill in.

The project has **four model iterations**: r1 (last-stage fine-tuning, 224 px; superseded), r2 (full fine-tuning,
224 px; superseded), r3 FP32 (r2's recipe at 320 px; superseded, bundle `ft_ckpt_98b1fb55cc6e_5d755416fd`) and
r3-int8 (**released**, bundle `ft_ckpt_98b1fb55cc6e_f3734d4f40`; bundle at its freeze
`ft_ckpt_98b1fb55cc6e_0d835360cc`). r3-int8 is r3 with one change: the hybrid (production) head's final logistic
layer is served as a row-wise ONNX INT8 graph, derived from the published `head_hybrid.joblib` by
`src/models/head_onnx.py` and pinned by sha256. The encoder stays PyTorch FP32 (no INT8 encoder passed the
accuracy policy); deep and classical heads stay sklearn FP32. Each iteration was frozen and then evaluated end to
end once on test, so the test split was used end to end four times (the `reports/e2e_test_*.json` files). The
ground-truth-crop (standard protocol) run was used twice (the frozen r2 bundle after its result was known, and r3
FP32; `reports/oracle_test_*.json`) and was not re-run for r3-int8. The 392-image stress subset was run for four
bundles and the 40 held-out negatives for three (`reports/robustness_stress_test_*.json`,
`reports/negatives_test_*.json`). All of this is disclosed in the status line of `eval_P5_T35.md`. r3's release
rule was committed in `NEXT.md` before r3 was trained; the "Iteration r3-int8 decision record" was committed
before any per-image, end-to-end, calibration or test INT8 number. Report r3-int8 as the result and r1, r2 and r3
FP32 as history ("Model iterations" in `eval_P5_T35.md`; earlier freezes in `artifacts/history/r1/`,
`artifacts/history/r2/` and `artifacts/history/r3/`).

## Suggested structure (8-12 pages)

| # | Section | Answer these questions | Facts and numbers from |
|---|---|---|---|
| 1 | Title, team, abstract | What did you build, for whom, and what is the single headline result? | `eval_P5_T35.md` section 1 |
| 2 | Problem and goals | What does PS5 ask for? Which requirements did you treat as hard (end-to-end, unseen test, honest confidence)? | the problem statement PDF; `docs/design.md` section 1 |
| 3 | Data | Where did the data come from and how did you verify it? How were train/tune/calibration/test split and why four parts? What did the audit find? | `reports/data_audit.json`, `artifacts/manifests/manifest_index.json`, `docs/resource_lock.md` |
| 4 | System architecture | What happens to one uploaded photo, step by step? Why a generic detector plus a fine-grained classifier? | `docs/architecture.md` (copy or redraw a diagram) |
| 5 | Classical computer vision | Which handcrafted features (edges, contours, HOG, LBP, HSV, geometry), and why? How does the contour proposal work? | `src/vision/features.py`, `src/vision/proposals.py`, `configs/features.yaml` |
| 6 | Deep learning | Why EfficientNet-B0 and YOLO11n? How was the network fine-tuned (r1, r2, r3; r3-int8 reuses r3's backbone)? | `configs/train*.yaml`, `reports/training/*_summary.json`, figure `training_tune_top1.png` |
| 7 | Hybrid fusion | How are learned and handcrafted features combined? What did the ablations show? | `configs/heads.yaml`, eval section 7, figure `ablations_tune.png` |
| 8 | Identity Evidence Passport | What is a conformal candidate set, what does "90%" mean and not mean, what are the decision states? | `docs/design.md` section 6, `src/passport/` |
| 9 | Evaluation protocol | Why end-to-end top-1 (box IoU >= 0.5 and right class) and how it differs from the usual ground-truth-crop number. Why freeze before testing? How many times was the test split used, and why is that disclosed? | eval "Model iterations" and section 1; `NEXT.md` decision records ("Iteration r3 decision record", "Iteration r3-int8 decision record"); the standard-protocol (ground-truth crop) report `reports/oracle_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json` from `scripts/evaluate_oracle.py` (r3 FP32; not re-run for r3-int8, as pre-registered) |
| 10 | Results | Top-1/Top-5, IoU, mAP, model size and latency, robustness, confidence/threshold (in the PS order); the standard-protocol number beside the end-to-end one (say it is r3 FP32's); how uncertain each number is (bootstrap intervals, and whether deep and hybrid differ); whether confidence is calibrated; why deep and hybrid are not cross-validated on train | `eval_P5_T35.md` sections 1-7; `reports/e2e_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json`, `reports/oracle_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json` (r3 FP32, not re-run), `reports/extra_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json` (`bootstrap`, `reliability`), `reports/robustness_stress_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.md`, `reports/negatives_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json`, `reports/resolution_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json`, `reports/crossval_classical.json`; figures `iterations_test_top1.png`, `robustness_hybrid.png`, `reject_threshold.png`, `reliability_test.png` |
| 11 | Optimization | What did ONNX FP32 and INT8 give? Why the encoder stays PyTorch FP32 (no INT8 encoder passed the policy) while the INT8 hybrid head is deployed (PS5 item 4.6: applied and evaluated end to end on test); why no meaningful end-to-end speed gain is claimed; INT8 vs FP32 on the same test images | eval section 4, `reports/optimization_ckpt_98b1fb55cc6e.json`, `reports/optimization_head_hybrid_ckpt_98b1fb55cc6e.json` (row-wise; old batch values under `superseded_batchwise`), `reports/int8_vs_fp32_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json`, `reports/int8_container_parity.json`, `NEXT.md` ("Iteration r3-int8 decision record") |
| 12 | Unsupported inputs | How the 80-image negative suite was built and what stopped each photo | `docs/negative_suite.md`, eval section 6, figure `negatives_test.png` |
| 13 | Failure analysis | Three rule-selected failures and their causes; which kinds of mix-up dominate (other model year, same make, badge-engineered twin, different make) | eval section 8; `reports/analysis_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json`, `reports/extra_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json` (`confusions`); figure `confusions_test.png` |
| 14 | Deployment and UI | How the app is served, the checks done on the public URL, error handling, the one-tap example row, the "Model comparison" panel (classical, deep and hybrid on the same photo, with each mode's official test results from `/benchmarks`), what `/readyz` reports (`head_runtime`, `encoder_precision`, `head_int8_sha256`, `model_fingerprint`) | `docs/hosting_lock.md`, `reports/release_acceptance.json`, `reports/container_check.json`, `app/static/examples/credits.json`, `docs/demo.md` |
| 15 | Limitations and ethics | Data ends at 2012; label noise; tunnel hosting; agent-made viewpoint labels; licences; no images stored | `docs/design.md` sections 10-11 |
| 16 | AI-use declaration | See below | |
| 17 | Conclusion and future work | What would you do with another week? | the GitHub issues #1-#5 on the repository |

## Figures (generated from the reports by `scripts/make_figures.py`)

| File | Shows |
|---|---|
| `reports/figures/iterations_test_top1.png` | End-to-end test top-1 per mode for the last three iterations (r2, r3 with the FP32 head, r3-int8); r1 is in the eval table only |
| `reports/figures/training_tune_top1.png` | Tune top-1 per training epoch for each run |
| `reports/figures/robustness_hybrid.png` | Hybrid end-to-end top-1 under the seven stress conditions |
| `reports/figures/reject_threshold.png` | Known-image retention and negative false acceptance against the reject threshold |
| `reports/figures/negatives_test.png` | What stopped each held-out unsupported photo |
| `reports/figures/ablations_tune.png` | Component ablations on tune |
| `reports/figures/reliability_test.png` | Reliability on correctly localised test images: accuracy against confidence, raw softmax versus temperature-scaled (from `reports/extra_test_<bundle>.json`) |
| `reports/figures/confusions_test.png` | The ten most frequent test mix-ups of the production mode, coloured by kind (from `reports/extra_test_<bundle>.json`) |

Extra analysis files (no new inference; read numbers from them, do not retype):

| File | Script | Shows |
|---|---|---|
| `reports/extra_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json` | `scripts/analyze_extra.py` | Bootstrap 95% intervals (2000 image-level resamples of the saved test predictions) for end-to-end top-1/top-5 of every mode and for the paired deep-minus-hybrid difference; most frequent confusion pairs by kind; reliability bins raw versus temperature-scaled |
| `reports/int8_vs_fp32_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json` | `scripts/compare_runs.py` | INT8 vs FP32 hybrid head per image, from the r3 FP32 and r3-int8 test prediction files (no extra test run): paired bootstrap, exact McNemar, the unchanged deep head as run-to-run control |
| `reports/resolution_test_ft_ckpt_98b1fb55cc6e_f3734d4f40.json` | `scripts/analyze_resolution.py` | End-to-end top-1 by width of the annotated car box, bins fixed before computing |
| `reports/int8_container_parity.json` | `scripts/int8_parity_host.py`, `scripts/int8_parity_container.py` | INT8 hybrid head logits on the same fused features of 16 tune crops, in the container and on the host |
| `reports/crossval_classical.json` | `scripts/crossval_classical.py` | 5-fold stratified cross-validation of the classical head on train; why deep and hybrid are not cross-validated there (the CNN was trained on those images, so the score would be inflated) |

Architecture and timeline diagrams: take screenshots of the Mermaid diagrams in `docs/architecture.md` and
`docs/timeline.md` on GitHub, or redraw them.

## AI-use declaration (adapt and write in your own words)

State plainly, in your own sentences:

* which tool you used (an AI coding assistant, Claude Code) and for what: implementing and testing code you
  specified, running the pipeline, writing repository documentation, collecting the negative-suite photos with
  licence checks, and a second-pass viewpoint labelling;
* what the team decided and did: the problem framing and design choices, the decisions recorded in `NEXT.md`
  (including the iteration rules), reviewing results, and writing this report;
* that every reported number is produced by scripts from saved report files, and how a reader can regenerate
  them (`scripts/report.py`).

## Keeping the report under the AI-content limit

* Write every paragraph yourselves. Tables of numbers and the generated figures are data, not prose.
* Do not paste text from `docs/*.md` or `eval_P5_T35.md` into the report; cite those files instead.
* If you use AI help to check grammar, say so in the declaration.
