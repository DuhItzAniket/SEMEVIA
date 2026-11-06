# Robustness on `stress_tune` (392 images, bundle `dev_imagenet_frozen_504165ac73`)

Synthetic, fixed conditions applied before detection (design choices, not a standard benchmark). Artificial darkening is not real night imagery; downsampling is not distance.

| Condition | Mode | E2E top-1 | Change vs clean | E2E top-5 | Box recall@0.5 | No box |
|---|---|---|---|---|---|---|
| clean | classical | 0.074 | +0.000 | 0.133 | 0.638 | 77 |
| clean | deep | 0.645 | +0.000 | 0.898 | 0.995 | 0 |
| clean | hybrid | 0.648 | +0.000 | 0.913 | 0.995 | 0 |
| bright_0.5 | classical | 0.064 | -0.010 | 0.151 | 0.707 | 18 |
| bright_0.5 | deep | 0.615 | -0.031 | 0.885 | 0.995 | 1 |
| bright_0.5 | hybrid | 0.630 | -0.018 | 0.893 | 0.995 | 1 |
| bright_1.5 | classical | 0.061 | -0.013 | 0.135 | 0.566 | 106 |
| bright_1.5 | deep | 0.612 | -0.033 | 0.875 | 0.992 | 1 |
| bright_1.5 | hybrid | 0.612 | -0.036 | 0.872 | 0.992 | 1 |
| down_0.50 | classical | 0.066 | -0.008 | 0.140 | 0.671 | 51 |
| down_0.50 | deep | 0.594 | -0.051 | 0.875 | 0.995 | 2 |
| down_0.50 | hybrid | 0.597 | -0.051 | 0.872 | 0.995 | 2 |
| down_0.25 | classical | 0.046 | -0.028 | 0.122 | 0.518 | 80 |
| down_0.25 | deep | 0.367 | -0.278 | 0.661 | 0.964 | 8 |
| down_0.25 | hybrid | 0.383 | -0.265 | 0.658 | 0.964 | 8 |
| occl_0.15 | classical | 0.038 | -0.036 | 0.117 | 0.643 | 79 |
| occl_0.15 | deep | 0.554 | -0.092 | 0.816 | 0.987 | 3 |
| occl_0.15 | hybrid | 0.559 | -0.089 | 0.832 | 0.987 | 3 |
| occl_0.30 | classical | 0.033 | -0.041 | 0.087 | 0.651 | 78 |
| occl_0.30 | deep | 0.337 | -0.309 | 0.605 | 0.946 | 5 |
| occl_0.30 | hybrid | 0.344 | -0.304 | 0.605 | 0.946 | 5 |
