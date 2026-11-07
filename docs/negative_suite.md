# Negative (unsupported-input) suite

Purpose: tune and test the reject rule on inputs outside the 196 Stanford Cars categories. It is a
small stress set, not an open-world guarantee.

**Status (2026-11-07): collected.** 80 photos, 40 tune + 40 test, 10 per bucket per partition, all from
Wikimedia Commons under CC0, public domain, CC BY or CC BY-SA. Each row of
`artifacts/negative_manifests/negatives_<partition>.csv` records the file page, licence and author (attribution).
Collection was done by four AI agents (one per bucket) that checked licences through the Commons API, looked at
every image, and confirmed the `unlisted_car_model` cars are not among the 196 classes; tune and test use
different photos and mostly different subjects. Images are stored outside Git (`SEMEVIA_NEG_ROOT`, on the team
laptop `C:\semevia\negatives`). The tune half fed `scripts/freeze.py --negatives`, which chose the reject
threshold at the freeze of the released iteration r3
([reports/negatives_tune_ft_ckpt_98b1fb55cc6e_f99e386191.json](../reports/negatives_tune_ft_ckpt_98b1fb55cc6e_f99e386191.json));
the test half was evaluated once after that freeze
([reports/negatives_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json](../reports/negatives_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json)),
and its false-acceptance table is in [eval_P5_T35.md](../eval_P5_T35.md) (section 6).

History: r2 was frozen and tested the same way
([tune](../reports/negatives_tune_ft_ckpt_4f3584f597d5_c13883508a.json),
[test](../reports/negatives_test_ft_ckpt_4f3584f597d5_9662c2e703.json)); r1 was frozen before the suite existed,
without `--negatives`.

## Target (plan)

80 verified images: 40 tune + 40 test, 10 per bucket per partition. If collection reaches its
45-minute cap, close a complete 40-image fallback (20 + 20, 5 per bucket per partition).

| Bucket | Examples | Rule |
|---|---|---|
| `non_vehicle` | street scenes without vehicles, animals, furniture | no vehicle visible |
| `two_wheeler` | bicycles, motorcycles, scooters | |
| `unsupported_vehicle_type` | buses, lorries, tractors, trains, boats | must not be a Stanford category (some pickups, vans and SUVs ARE supported) |
| `unlisted_car_model` | cars whose make/model/year is not in `artifacts/class_metadata.json` (e.g. models after 2012) | check the label list for every image |

Tune and test must use different photos, sources or physical vehicles, and preferably different
unsupported model families.

## How to record each image

1. Save the image under `SEMEVIA_NEG_ROOT` (outside Git), e.g. `D:/semevia_work/negatives/tune/nv_001.jpg`.
2. Add a row to `artifacts/negative_manifests/negatives_<tune|test>.csv`:
   `neg_id,rel_path,bucket,source_url,license,author,notes`. Rows without a source URL and licence
   are rejected by the evaluator. Suitable sources: Wikimedia Commons files with a CC BY / CC BY-SA /
   public-domain licence (record author and licence), or the team's own photos (`license: team-owned`).
3. Run `python scripts/evaluate_negatives.py --partition tune`, then
   `python scripts/freeze.py --mode hybrid --negatives reports/negatives_tune_<bundle>.json`.
   The test set is evaluated only after the freeze.
