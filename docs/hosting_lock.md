# Hosting lock

Status on 2026-11-06: **no public host provisioned. This is a release blocker.**

The user chose to decide the host later. Until then the application is built and tested
locally with `docker compose up --build`; no public URL exists and none is claimed.

Requirements the chosen host must meet (from the plan and rubric):

* Always on: no idle sleep (Render Free and Hugging Face CPU Basic sleep or are not guaranteed).
* Runs the Docker image from this repository; starting capacity about 2 vCPU and 4 GB RAM,
  confirmed by measurement.
* Public HTTPS URL, no login or tunnel interstitial.
* Billing cap and judging period confirmed by the account owner.

Options offered to the user: Render paid instance, Google Cloud Run with one minimum instance,
or a VPS with Docker and Caddy for HTTPS. Each needs the user's account and billing approval.
