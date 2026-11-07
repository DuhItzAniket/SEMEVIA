# Licences for model files, data products and third-party material

The **source code** in this repository is licensed under the GNU Affero General Public License v3.0
(see [LICENSE](LICENSE)). AGPL-3.0 was chosen because the application ships Ultralytics YOLO11n, which is
itself AGPL-3.0.

## Trained model files

The fine-tuned EfficientNet-B0 backbones, the classical / deep / hybrid heads and the training-only reference
indexes published as GitHub Releases `model-v1`, `model-v2` and `model-v3` (listed with SHA256 in
[artifacts/release_manifest.json](artifacts/release_manifest.json)) are licensed under
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**,
<https://creativecommons.org/licenses/by-nc/4.0/>.

They were trained on the Stanford Cars dataset, which its authors make available for non-commercial research
purposes only; the non-commercial condition is carried over to the trained weights.

## Reports, figures and documentation

Files under `reports/`, `docs/` and the evaluation file `eval_P5_T35.md` are licensed under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**, <https://creativecommons.org/licenses/by/4.0/>.

## Third-party material (not covered by the licences above)

| Material | Licence | Notes |
|---|---|---|
| Stanford Cars dataset (Krause et al., 2013) | Non-commercial research use | Not redistributed by this repository or the container |
| YOLO11n weights and the Ultralytics library | AGPL-3.0 | Downloaded at build time from the Ultralytics release |
| torchvision ImageNet EfficientNet-B0 weights (starting point for fine-tuning) | BSD-3-Clause (torchvision) | Used only to initialise training |
| Example photos in `app/static/examples/` | CC BY 2.0 / CC BY-SA 4.0 | Authors and sources in `app/static/examples/credits.json` and on the app page |
| Negative-suite photos | CC0, public domain, CC BY, CC BY-SA | Sources, authors and licences in `artifacts/negative_manifests/`; images not in Git |
