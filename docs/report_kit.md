# Report kit (for the team's own write-up)

The final report PDF must be **written by the team**: the faculty rule caps AI-written content at 30% and asks
for AI use to be declared. This page therefore gives **no ready-made prose**. It gives the structure, the
questions each section should answer, where every fact and number lives, and the figures. Write each section
in your own words; copy numbers only from the files named here (or from `eval_P5_T35.md`, which is generated
from them).

Team **Hypermetropia** (T35), PS5. Members: Aniket Kumar (1RUA24CSE0051), C Lakshanaa (1RUA24CSE0097).
Sections of each member: fill in.

The project has **three model iterations**: r1 (last-stage fine-tuning, 224 px; superseded), r2 (full fine-tuning,
224 px; superseded) and r3 (r2's recipe at 320 px; **released**, bundle `ft_ckpt_98b1fb55cc6e_5d755416fd`). Each
was frozen and then evaluated end to end once on test, so the test split was used three times (plus one
standard-protocol, ground-truth-crop run of the frozen r2 bundle after its result was known, disclosed in the
status line of `eval_P5_T35.md`); r3's release rule was
committed in `NEXT.md` before r3 was trained. Report r3 as the result and r1/r2 as history ("Model iterations" in
`eval_P5_T35.md`; earlier freezes in `artifacts/history/r1/` and `artifacts/history/r2/`).

## Suggested structure (8-12 pages)

| # | Section | Answer these questions | Facts and numbers from |
|---|---|---|---|
| 1 | Title, team, abstract | What did you build, for whom, and what is the single headline result? | `eval_P5_T35.md` section 1 |
| 2 | Problem and goals | What does PS5 ask for? Which requirements did you treat as hard (end-to-end, unseen test, honest confidence)? | the problem statement PDF; `docs/design.md` section 1 |
| 3 | Data | Where did the data come from and how did you verify it? How were train/tune/calibration/test split and why four parts? What did the audit find? | `reports/data_audit.json`, `artifacts/manifests/manifest_index.json`, `docs/resource_lock.md` |
| 4 | System architecture | What happens to one uploaded photo, step by step? Why a generic detector plus a fine-grained classifier? | `docs/architecture.md` (copy or redraw a diagram) |
| 5 | Classical computer vision | Which handcrafted features (edges, contours, HOG, LBP, HSV, geometry), and why? How does the contour proposal work? | `src/vision/features.py`, `src/vision/proposals.py`, `configs/features.yaml` |
| 6 | Deep learning | Why EfficientNet-B0 and YOLO11n? How was the network fine-tuned (r1, r2, r3)? | `configs/train*.yaml`, `reports/training/*_summary.json`, figure `training_tune_top1.png` |
| 7 | Hybrid fusion | How are learned and handcrafted features combined? What did the ablations show? | `configs/heads.yaml`, eval section 7, figure `ablations_tune.png` |
| 8 | Identity Evidence Passport | What is a conformal candidate set, what does "90%" mean and not mean, what are the decision states? | `docs/design.md` section 6, `src/passport/` |
| 9 | Evaluation protocol | Why end-to-end top-1 (box IoU >= 0.5 and right class) and how it differs from the usual ground-truth-crop number. Why freeze before testing? How many times was the test split used, and why is that disclosed? | eval "Model iterations" and section 1; `NEXT.md` decision records (r3: "Iteration r3 decision record"); the standard-protocol (ground-truth crop) report `reports/oracle_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json` from `scripts/evaluate_oracle.py` |
| 10 | Results | Top-1/Top-5, IoU, mAP, model size and latency, robustness, confidence/threshold (in the PS order); the standard-protocol number beside the end-to-end one | `eval_P5_T35.md` sections 1-6; `reports/e2e_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`, `reports/oracle_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`; figures `iterations_test_top1.png`, `robustness_hybrid.png`, `reject_threshold.png` |
| 11 | Optimization | What did ONNX FP32 and INT8 give, why the release keeps PyTorch FP32 | eval section 4, `reports/optimization_ckpt_98b1fb55cc6e.json`, `reports/optimization_head_hybrid_ckpt_98b1fb55cc6e.json` |
| 12 | Unsupported inputs | How the 80-image negative suite was built and what stopped each photo | `docs/negative_suite.md`, eval section 6, figure `negatives_test.png` |
| 13 | Failure analysis | Three rule-selected failures and their causes | eval section 8 |
| 14 | Deployment and UI | How the app is served, the checks done on the public URL, error handling, the one-tap example row | `docs/hosting_lock.md`, `reports/release_acceptance.json`, `reports/container_check.json`, `app/static/examples/credits.json`, `docs/demo.md` |
| 15 | Limitations and ethics | Data ends at 2012; label noise; tunnel hosting; agent-made viewpoint labels; licences; no images stored | `docs/design.md` sections 10-11 |
| 16 | AI-use declaration | See below | |
| 17 | Conclusion and future work | What would you do with another week? | the GitHub issues #1-#5 on the repository |

## Figures (generated from the reports by `scripts/make_figures.py`)

| File | Shows |
|---|---|
| `reports/figures/iterations_test_top1.png` | End-to-end test top-1 per mode for each of the three model iterations |
| `reports/figures/training_tune_top1.png` | Tune top-1 per training epoch for each run |
| `reports/figures/robustness_hybrid.png` | Hybrid end-to-end top-1 under the seven stress conditions |
| `reports/figures/reject_threshold.png` | Known-image retention and negative false acceptance against the reject threshold |
| `reports/figures/negatives_test.png` | What stopped each held-out unsupported photo |
| `reports/figures/ablations_tune.png` | Component ablations on tune |

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
