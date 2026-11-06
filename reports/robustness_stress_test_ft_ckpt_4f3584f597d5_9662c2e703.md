# Robustness on `stress_test` (392 images, bundle `ft_ckpt_4f3584f597d5_9662c2e703`)

Synthetic, fixed conditions applied before detection (design choices, not a standard benchmark). Artificial darkening is not real night imagery; downsampling is not distance.

| Condition | Mode | E2E top-1 | Change vs clean | E2E top-5 | Box recall@0.5 | No box |
|---|---|---|---|---|---|---|
| clean | classical | 0.048 | +0.000 | 0.112 | 0.633 | 86 |
| clean | deep | 0.878 | +0.000 | 0.964 | 0.990 | 2 |
| clean | hybrid | 0.883 | +0.000 | 0.962 | 0.990 | 2 |
| bright_0.5 | classical | 0.041 | -0.008 | 0.135 | 0.684 | 27 |
| bright_0.5 | deep | 0.872 | -0.005 | 0.962 | 0.990 | 2 |
| bright_0.5 | hybrid | 0.870 | -0.013 | 0.962 | 0.990 | 2 |
| bright_1.5 | classical | 0.038 | -0.010 | 0.097 | 0.582 | 107 |
| bright_1.5 | deep | 0.880 | +0.003 | 0.959 | 0.995 | 1 |
| bright_1.5 | hybrid | 0.880 | -0.003 | 0.957 | 0.995 | 1 |
| down_0.50 | classical | 0.048 | +0.000 | 0.128 | 0.651 | 61 |
| down_0.50 | deep | 0.875 | -0.003 | 0.962 | 0.992 | 3 |
| down_0.50 | hybrid | 0.875 | -0.008 | 0.962 | 0.992 | 3 |
| down_0.25 | classical | 0.031 | -0.018 | 0.087 | 0.508 | 93 |
| down_0.25 | deep | 0.745 | -0.133 | 0.908 | 0.972 | 8 |
| down_0.25 | hybrid | 0.742 | -0.140 | 0.906 | 0.972 | 8 |
| occl_0.15 | classical | 0.026 | -0.023 | 0.097 | 0.640 | 84 |
| occl_0.15 | deep | 0.781 | -0.097 | 0.926 | 0.987 | 2 |
| occl_0.15 | hybrid | 0.781 | -0.102 | 0.926 | 0.987 | 2 |
| occl_0.30 | classical | 0.020 | -0.028 | 0.074 | 0.633 | 87 |
| occl_0.30 | deep | 0.653 | -0.224 | 0.832 | 0.952 | 6 |
| occl_0.30 | hybrid | 0.651 | -0.232 | 0.824 | 0.952 | 6 |
