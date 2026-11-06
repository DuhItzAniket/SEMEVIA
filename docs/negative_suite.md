# Negative (unsupported-input) suite

Purpose: tune and test the reject rule on inputs outside the 196 Stanford Cars categories. It is a
small stress set, not an open-world guarantee. **Status: not collected yet.**

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
