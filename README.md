<p align="center"><img src="docs/assets/semevia-mark.svg" alt="SEMEVIA logo" width="96"></p>

<h1 align="center">SEMEVIA</h1>

<p align="center"><b>Vehicle identity through evidence</b><br>
Upload a photo of a car. SEMEVIA tells you its make, model and year, and how sure it really is.</p>

<p align="center">
<a href="https://semevia-298534486429.asia-south1.run.app"><img src="https://img.shields.io/badge/demo-live-brightgreen?logo=googlecloud&logoColor=white" alt="Live demo"></a>
<a href="https://github.com/DuhItzAniket/SEMEVIA/actions/workflows/ci.yml"><img src="https://github.com/DuhItzAniket/SEMEVIA/actions/workflows/ci.yml/badge.svg?branch=semevia/implementation" alt="CI"></a>
<a href="https://doi.org/10.5281/zenodo.24026963"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.24026963.svg" alt="DOI 10.5281/zenodo.24026963"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL-3.0"></a>
<a href="eval_P5_T35.md"><img src="https://img.shields.io/badge/test%20top--1-91.8%25-success" alt="Test top-1 91.8%"></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
<img src="https://img.shields.io/badge/PyTorch-2.15-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.15">
<img src="https://img.shields.io/badge/YOLO11n-Ultralytics-111F68" alt="YOLO11n (Ultralytics)">
<img src="https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white" alt="FastAPI 0.162">
<img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker ready">
<img src="https://img.shields.io/badge/dataset-Stanford%20Cars%20(196%20classes)-8C1515" alt="Dataset: Stanford Cars, 196 classes">
</p>

<p align="center">
<a href="https://semevia-298534486429.asia-south1.run.app"><b>Try it live</b></a> ·
<a href="eval_P5_T35.md">Results</a> ·
<a href="#video-showing-it-live">Demo video</a>
</p>

---

## Introduction

Many cars look almost the same: the same brand, the same shape, only a different year or trim. SEMEVIA is our
answer to **Intelligent Fine-Grained Vehicle Recognition and Analysis**.

You give it a photo. It finds the car, names it among **196 car models** from the Stanford Cars dataset, and shows its top 5 guesses. It also gives an **Evidence Passport**: instead of
always claiming one exact answer, it says what the photo actually supports. Sometimes that is the exact model;
sometimes only the model family or just the brand.

**Live app:** <https://semevia-298534486429.asia-south1.run.app>


## Technologies Used

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Finding the car in the photo | YOLO11n (Ultralytics), a small ready-made detector |
| Recognising the model | EfficientNet-B0 (PyTorch), a pretrained network we fine-tuned on a GPU |
| Classic image features | OpenCV, scikit-image (edges, shapes, texture, colour) |
| Final classifier and confidence | scikit-learn |
| Website | FastAPI (backend), plain HTML, CSS and JavaScript (frontend) |
| Packaging and hosting | Docker, Google Cloud Run; model files on GitHub Releases |
| Speed-up and compression | ONNX Runtime (8-bit final classifier) |

## Features

- **Upload, take a photo or tap an example.** Works on phone and laptop; four one-tap example photos are built in.
- **Finds the car** and draws a box around it.
- **Names the car**: make, model and year, with the top 5 guesses and a confidence score.
- **Three recognition modes**:
  - **Classical**: hand-made image features only.
  - **Deep**: the trained neural network.
  - **Hybrid** (default): both together.
- **Evidence Passport**: shows which car models are still possible and whether the photo supports the exact
  model, only the family, or only the brand.
- **Six-region challenge**: blurs one part of the car at a time to show which parts the answer depends on.
- **Similar training cars**: the closest matches from the training data.
- **Behind the scenes**: the cropped car, edges, outlines and simple measurements.
- **Compare all three modes** on the same photo with one click.
- **Download the result** as a small JSON file.
- **Clear messages** for wrong files (not an image, too large, empty) and for photos with no car.
- **Privacy**: uploaded photos are never saved; they are kept in memory briefly and then deleted.


## The Process

1. **Prepare the data.** We used Stanford Cars (16,185 photos, 196 models). We checked every photo and box, then
   split the photos into four groups: training, tuning, calibration and a test set that stayed untouched until
   the very end.
2. **Find the car.** A small ready-made detector (YOLO11n) finds the car. It already worked on over 99% of
   photos, so we did not need to train our own.
3. **Recognise the model.** We first tried classic hand-made image features, then a pretrained neural network
   (EfficientNet-B0), which we fine-tuned on a GPU over three training rounds. The third round, trained on
   larger images, was the best and is the one in the app. Combining both kinds of features gives the Hybrid mode.
4. **Make the confidence honest.** We tuned the confidence on a separate set of photos, so the Passport only
   claims what the evidence supports.
5. **Lock it, then test once.** We froze the final model, then tested it a single time on 8,041 photos it had
   never seen.
6. **Try to make it smaller and faster.** We converted the network to ONNX and tried 8-bit compression.
   Compressing the whole network lost too much accuracy, so it stays full precision. Only the small final
   classifier runs in 8-bit, with the same test accuracy.
7. **Ship it.** We packaged everything in Docker and put it online on Google Cloud Run.

### How well it works

Measured on the 8,041 unseen test photos (full details in [eval_P5_T35.md](eval_P5_T35.md)):

| Mode | Correct first guess (Top-1) | Right answer in top 5 guesses (Top-5) |
|---|---|---|
| **Hybrid** (used in the app) | **91.8%** | **98.0%** |
| Deep | 91.8% | 97.9% |
| Classical | 6.3% | 14.8% |

- The car was found correctly in **99.1%** of the photos.
- A result comes back in about **half a second** on the live site (508 ms typical).
- The classical features on their own are weak at telling 196 similar models apart; the neural network makes
  the big difference.

## What We Learnt

- **Finding a car is easy; telling similar cars apart is hard.** The ready-made detector was enough, and almost
  all the effort went into recognising the model.
- **Pretrained networks are powerful.** Fine-tuning one took us from about 6% (classic features) to about 92%.
- **Keep the test set hidden until the end.** Otherwise it is easy to fool yourself with results that are too good.
- **A confidence score is not a guarantee.** It has to be checked on separate photos before you can trust it.
- **Smaller is not always better.** Compressing the model made it faster but broke its accuracy.
- **Deployment takes real work.** Disk space, Docker images and free hosting limits took as much care as the model.

## How It Could Be Improved

- **Newer cars.** The dataset stops at 2012 models; newer cars and other vehicle types are not recognised.
- **Several cars in one photo.** The app always picks the largest car.
- **A smaller, faster model** that keeps its accuracy, for phones and cheaper servers.
- **Show pictures of similar cars.** Reference matches are shown as names only, because of the dataset's image rights.

## How to Run the Project

You need [Docker](https://www.docker.com/) and about 4 GB of free disk space. No passwords or keys are needed.

```bash
git clone https://github.com/DuhItzAniket/SEMEVIA
cd SEMEVIA
git checkout semevia/implementation
docker compose up --build
```

Open <http://localhost:8000> in your browser. The first build downloads the model files automatically and
checks them.

<details>
<summary>Without Docker (for development)</summary>

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m src.release fetch
venv/Scripts/python -m uvicorn app.main:app --port 8000
```

On macOS or Linux use `venv/bin/` instead of `venv/Scripts/`. Optional local settings: copy `.env.example`
to `.env`. To retrain or re-run the evaluation, follow [docs/runbook_other_pc.md](docs/runbook_other_pc.md).
</details>

## Video Showing It Live

*Coming soon. The demo video will be added here once the demo is recorded.*

## Citation

If you use SEMEVIA, please cite it as:

> Aniket Kumar and C Lakshanaa (2026). *SEMEVIA: fine-grained vehicle recognition with an Identity Evidence
> Passport* (Version 1.0.0). Zenodo. <https://doi.org/10.5281/zenodo.24026963>

The project is archived on Zenodo (the DOI above covers all versions; v1.0.0 alone is
[10.5281/zenodo.24026964](https://doi.org/10.5281/zenodo.24026964)). GitHub's **Cite this repository** button,
in the sidebar, gives the same citation in APA and BibTeX from [CITATION.cff](CITATION.cff).

<details>
<summary>BibTeX (for LaTeX)</summary>

```bibtex
@software{kumar_lakshanaa_2026_semevia,
  author    = {Kumar, Aniket and Lakshanaa, C},
  title     = {SEMEVIA: fine-grained vehicle recognition with an Identity Evidence Passport},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24026963},
  url       = {https://doi.org/10.5281/zenodo.24026963}
}
```
</details>

**Dataset used:** Stanford Cars, by J. Krause, M. Stark, J. Deng and L. Fei-Fei, "3D Object Representations
for Fine-Grained Categorization", 4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013.

---

**Team Hypermetropia (T35)**: Aniket Kumar (1RUA24CSE0051), C Lakshanaa (1RUA24CSE0097)

**More detail:** [results](eval_P5_T35.md) · [how it works](docs/architecture.md) ·
[design](docs/design.md) · [licences](LICENSE-MODELS.md) (code AGPL-3.0; model files non-commercial,
because Stanford Cars is for non-commercial research only)
