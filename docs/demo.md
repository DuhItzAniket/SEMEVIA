# Three-minute demo and viva notes

Use the live public URL (not localhost). Preselected images are **illustrative**; say so. Use one
fresh, unseen upload from the judges if offered. Numbers shown come from `eval_P5_<TeamID>.md`.

| Time | Action | What to say |
|---|---|---|
| 0:00-0:25 | Open the URL on a phone, upload a car photo, Analyze | Real YOLO11n box; best guess with make, model, year; classification and detector scores are separate |
| 0:25-1:00 | Show the Passport on an ambiguous photo | Full candidate count; why the supported answer stops at family or make; the 90% target is a population setting, not this photo's probability |
| 1:00-1:40 | Run the six-region challenge, select a cell, Restore original | Fixed A/B pair; margin change in logit units; "hiding this region reduced the A-versus-B margin under this blur", not "this region is the grille" |
| 1:40-2:05 | References panel; an unsupported or rejected example | Similarity is not identity; rejection is a fallible softmax rule; show the "could not verify" state |
| 2:05-2:35 | Open the evaluation file / benchmark link | Classical vs deep vs hybrid, ablations, Top-1/Top-5, detection AP, ONNX/INT8 results |
| 2:35-3:00 | One real failure case and limitations | From `reports/analysis_test_<bundle>.json`; dataset covers 196 categories to 2012; label noise; imperfect rejection |

## Questions both members should be able to answer

* Why a compact pretrained CNN (EfficientNet-B0) and why fine-tune only the last stage.
* Why a separate generic detector, and the largest-box target rule.
* What HOG, LBP, HSV and the geometry features capture, and what the ablations showed.
* Why matched heads on the same crops (controlled comparison) and why a linear fusion head.
* Why separate tune / calibration / test partitions; what the conformal 90% means and does not mean.
* Why softmax rejection is fallible; how the negative suite was used.
* Why a heatmap or patch cannot name car parts; what the challenge does measure.
* What the optimization changed: ONNX FP32 (speed), INT8 encoder failure, INT8 head (size).
* How training resumes after an interruption (`--resume auto`) and how results are reproduced.
