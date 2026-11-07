<p align="center"><img src="docs/assets/semevia-mark.svg" alt="SEMEVIA logo" width="96"></p>

<h1 align="center">SEMEVIA</h1>

<p align="center"><b>Vehicle identity through evidence</b><br>
Upload a photo of a car. SEMEVIA tells you its make, model and year, and how sure it really is.</p>

<p align="center">
<a href="https://readings-agency-respond-bacteria.trycloudflare.com"><b>Try it live</b></a> ·
<a href="eval_P5_T35.md">Results</a> ·
<a href="#video-showing-it-live">Demo video</a>
</p>

---

## Intro

Many cars look almost the same: the same brand, the same shape, only a different year or trim. SEMEVIA is our
answer to **PS5, Intelligent Fine-Grained Vehicle Recognition and Analysis** (Computer Vision Hackathon 2026).

You give it a photo. It finds the car, names it among **196 car models** from the Stanford Cars dataset (for
example *Audi R8 Coupe 2012*), and shows its top 5 guesses. It also gives an **Evidence Passport**: instead of
always claiming one exact answer, it says what the photo actually supports. Sometimes that is the exact model;
sometimes only the model family (e.g. *Ford F-150*) or just the brand.

**Live app:** <https://readings-agency-respond-bacteria.trycloudflare.com>
(works on phone and laptop; currently served from a team laptop, so it is online only while that laptop is on)

## Technologies Used

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Finding the car in the photo | YOLO11n (Ultralytics), a small ready-made detector |
| Recognising the model | EfficientNet-B0 (PyTorch), a pretrained network we fine-tuned on a GPU |
| Classic image features | OpenCV, scikit-image (edges, shapes, texture, colour) |
| Final classifier and confidence | scikit-learn |
| Website | FastAPI (backend), plain HTML, CSS and JavaScript (frontend) |
| Packaging and hosting | Docker, Cloudflare Tunnel; model files on GitHub Releases |
| Speed-up experiments | ONNX Runtime |

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

## Keyboard Shortcuts

There are no special shortcuts. The whole page works with a normal keyboard: **Tab** moves between controls, and
**Enter** or **Space** presses a button or picks a mode.

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
6. **Try to make it smaller and faster.** We converted the network to ONNX and tried 8-bit compression. The
   compressed network lost too much accuracy, so the app keeps the full-precision model.
7. **Ship it.** We packaged everything in Docker and put it online.

### How well it works

Measured on the 8,041 unseen test photos (full details in [eval_P5_T35.md](eval_P5_T35.md)):

| Mode | Correct first guess (Top-1) | Right answer in top 5 guesses (Top-5) |
|---|---|---|
| **Hybrid** (used in the app) | **91.8%** | **98.0%** |
| Deep | 91.8% | 97.9% |
| Classical | 6.3% | 14.8% |

- The car was found correctly in **99.1%** of the photos.
- A result comes back in about **half a second** on the live site (463 ms typical).
- The classical features on their own are weak at telling 196 similar models apart; the neural network makes
  the big difference.

## What I Learnt

- **Finding a car is easy; telling similar cars apart is hard.** The ready-made detector was enough, and almost
  all the effort went into recognising the model.
- **Pretrained networks are powerful.** Fine-tuning one took us from about 6% (classic features) to about 92%.
- **Keep the test set hidden until the end.** Otherwise it is easy to fool yourself with results that are too good.
- **A confidence score is not a guarantee.** It has to be checked on separate photos before you can trust it.
- **Smaller is not always better.** Compressing the model made it faster but broke its accuracy.
- **Deployment takes real work.** Disk space, Docker images and free hosting limits took as much care as the model.

## How It Could Be Improved

- **Newer cars.** The dataset stops at 2012 models; newer cars and other vehicle types are not recognised.
- **Better rejection of non-car photos.** About 1 in 8 of our test non-car photos was still accepted as a car
  (for example, a city bus was named as a Hummer).
- **Several cars in one photo.** The app always picks the largest car.
- **Always-on hosting.** Move the app from the team laptop to a cloud server, so the link never goes down.
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

---

**Team Hypermetropia (T35)**: Aniket Kumar (1RUA24CSE0051), C Lakshanaa (1RUA24CSE0097)

**More detail:** [results](eval_P5_T35.md) · [how it works](docs/architecture.md) ·
[design](docs/design.md) · [licences](LICENSE-MODELS.md) (code AGPL-3.0; model files non-commercial,
because Stanford Cars is for non-commercial research only)
