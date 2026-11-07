# Hosting lock

Status on 2026-11-07: **live at <https://semevia-298534486429.asia-south1.run.app>** on Google Cloud Run. It no
longer depends on the team laptop.

## What runs where

* Google Cloud Run service `semevia` in project `semevia`, region `asia-south1` (Mumbai). It runs the same release
  image as `docker compose` (bundle named in `artifacts/release_manifest.json`; iteration r3-int8), pushed to
  Artifact Registry as `asia-south1-docker.pkg.dev/semevia/semevia/semevia:r3-int8`.
* Settings: 2 CPUs, 4 GiB memory (the same limits as `compose.yaml`), one minimum instance so the models are
  always loaded (no cold start), at most 2 instances, 8 concurrent requests per instance, 300 s request timeout,
  startup CPU boost, `SEMEVIA_TORCH_THREADS=2`, public (unauthenticated) access.
* The image is built locally with `docker compose build`, which fetches the model files from GitHub Release
  `model-v3`, rebuilds the INT8 hybrid head and checks every SHA256 (`python -m src.release fetch`, `verify`,
  `check-derived`). The same Dockerfile builds in CI from a clean checkout.
* Checks through the public URL: [reports/release_acceptance.json](../reports/release_acceptance.json)
  (`scripts/check_release.py --url https://semevia-298534486429.asia-south1.run.app`); local container checks:
  [reports/container_check.json](../reports/container_check.json).

## Redeploy after a new release

```
docker compose build
docker tag semevia:local asia-south1-docker.pkg.dev/semevia/semevia/semevia:<tag>
docker push asia-south1-docker.pkg.dev/semevia/semevia/semevia:<tag>
gcloud run deploy semevia --image asia-south1-docker.pkg.dev/semevia/semevia/semevia:<tag> --region asia-south1
python scripts/check_release.py --url https://semevia-298534486429.asia-south1.run.app
```

The URL stays the same across redeploys.

## Limits the team must manage

* The minimum instance is billed while it runs (covered by the Google Cloud free-trial credit). After judging,
  set `--min-instances 0` to stop that cost; the first request after an idle period then waits for the models to
  load (about 15 s of warm-up, `warmup_ms` in `/readyz`).
* The CPU differs from the team laptop, so latency in `reports/release_acceptance.json` is Cloud Run's, not the
  laptop's.

## Options considered

* Cloudflare quick tunnel to the team laptop: used until 2026-11-07 (address
  `readings-agency-respond-bacteria.trycloudflare.com`); it worked only while the laptop, Docker Desktop and
  `cloudflared` were running.
* Hugging Face Docker Space: Hugging Face answered `402 Payment Required` (Docker Spaces on free CPU need a PRO
  subscription). Ready to use with `scripts/prepare_hf_space.py` if the team subscribes.
* Render or a VPS: not needed after Cloud Run.
