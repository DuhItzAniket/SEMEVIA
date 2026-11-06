# Hosting lock

Status on 2026-11-07: **live at <https://readings-agency-respond-bacteria.trycloudflare.com>**, served from the team laptop.

## What runs where

* The release image (`docker compose up -d`, bundle named in `artifacts/release_manifest.json`; iteration r2 since 2026-11-07 03:47) runs in Docker Desktop
  on the team laptop (Intel i7-14650HX), limited to 2 CPUs and 4 GB by `compose.yaml`, restart `unless-stopped`.
* A Cloudflare quick tunnel (`cloudflared tunnel --url http://localhost:8000`, release 2026.10.7, SHA256 and
  Authenticode signature checked) gives the public HTTPS address. Quick tunnels need no account and show no
  interstitial page.
* Checks through the public URL: [reports/release_acceptance.json](../reports/release_acceptance.json)
  (`scripts/check_release.py`).

## Limits the team must manage

* The URL works only while the laptop is on, awake, online, and both Docker Desktop and `cloudflared` run.
  Keep it plugged in, lid open, sleep disabled during judging.
* A quick-tunnel address changes every time `cloudflared` restarts. If it restarts, update the URL in the
  README and the submission form, and rerun `scripts/check_release.py --url <new url>`.
* Cloudflare documents quick tunnels as a testing feature with no uptime guarantee.

## Options considered

* Hugging Face Docker Space: chosen first; on 2026-11-07 Hugging Face answered `402 Payment Required` (Docker
  Spaces on free CPU need a PRO subscription). Ready to use with `scripts/prepare_hf_space.py` if the team
  subscribes.
* Render paid instance, Google Cloud Run with one minimum instance, or a VPS: need billing; not chosen.
