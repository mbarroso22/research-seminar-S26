# Stats Appendix (n=2000)

This appendix is auto-generated from CSV outputs in `results_n2000/`.

## Failure levels (50% drop rule)

### EfficientNet — Appearance

| model             | axis       | transform   | metric          |   baseline_level0 |   threshold_50pct |   failure_level_50pct |
|:------------------|:-----------|:------------|:----------------|------------------:|------------------:|----------------------:|
| efficientnet_v2_s | appearance | blur        | mean_confidence |          0.352268 |          0.176134 |                     1 |
| efficientnet_v2_s | appearance | pixelate    | mean_confidence |          0.352268 |          0.176134 |                     2 |

### EfficientNet — Geometry

| model             | axis     | transform   | metric          |   baseline_level0 |   threshold_50pct |   failure_level_50pct |
|:------------------|:---------|:------------|:----------------|------------------:|------------------:|----------------------:|
| efficientnet_v2_s | geometry | rotate      | mean_confidence |          0.352268 |          0.176134 |                   nan |
| efficientnet_v2_s | geometry | shear_x     | mean_confidence |          0.352268 |          0.176134 |                   nan |
| efficientnet_v2_s | geometry | translate   | mean_confidence |          0.352268 |          0.176134 |                   nan |

### OpenCLIP — Appearance

| model             | axis       | transform   | metric   |   baseline_level0 |   threshold_50pct |   failure_level_50pct |
|:------------------|:-----------|:------------|:---------|------------------:|------------------:|----------------------:|
| openclip_vit_b_32 | appearance | blur        | accuracy |            0.8885 |           0.44425 |                     2 |
| openclip_vit_b_32 | appearance | pixelate    | accuracy |            0.8885 |           0.44425 |                     1 |

### OpenCLIP — Geometry

| model             | axis     | transform   | metric   |   baseline_level0 |   threshold_50pct |   failure_level_50pct |
|:------------------|:---------|:------------|:---------|------------------:|------------------:|----------------------:|
| openclip_vit_b_32 | geometry | rotate      | accuracy |            0.8885 |           0.44425 |                     2 |
| openclip_vit_b_32 | geometry | shear_x     | accuracy |            0.8885 |           0.44425 |                   nan |
| openclip_vit_b_32 | geometry | translate   | accuracy |            0.8885 |           0.44425 |                   nan |

## Behavioral summaries

### EfficientNet — Appearance (mean confidence)

| model             | dataset   | axis       | transform   |   level |   mean_confidence |    n |
|:------------------|:----------|:-----------|:------------|--------:|------------------:|-----:|
| efficientnet_v2_s | cifar10   | appearance | blur        |       0 |         0.352268  | 2000 |
| efficientnet_v2_s | cifar10   | appearance | blur        |       1 |         0.16917   | 2000 |
| efficientnet_v2_s | cifar10   | appearance | blur        |       2 |         0.0956593 | 2000 |
| efficientnet_v2_s | cifar10   | appearance | blur        |       3 |         0.0773403 | 2000 |
| efficientnet_v2_s | cifar10   | appearance | pixelate    |       0 |         0.352268  | 2000 |
| efficientnet_v2_s | cifar10   | appearance | pixelate    |       1 |         0.209928  | 2000 |
| efficientnet_v2_s | cifar10   | appearance | pixelate    |       2 |         0.0895516 | 2000 |
| efficientnet_v2_s | cifar10   | appearance | pixelate    |       3 |         0.157709  | 2000 |

### EfficientNet — Geometry (mean confidence)

| model             | dataset   | axis     | transform   |   level |   mean_confidence |    n |
|:------------------|:----------|:---------|:------------|--------:|------------------:|-----:|
| efficientnet_v2_s | cifar10   | geometry | rotate      |       0 |          0.352268 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | rotate      |       1 |          0.208495 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | rotate      |       2 |          0.19786  | 2000 |
| efficientnet_v2_s | cifar10   | geometry | rotate      |       3 |          0.210025 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | shear_x     |       0 |          0.352268 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | shear_x     |       1 |          0.323046 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | shear_x     |       2 |          0.294459 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | shear_x     |       3 |          0.25356  | 2000 |
| efficientnet_v2_s | cifar10   | geometry | translate   |       0 |          0.352268 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | translate   |       1 |          0.344027 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | translate   |       2 |          0.322921 | 2000 |
| efficientnet_v2_s | cifar10   | geometry | translate   |       3 |          0.304452 | 2000 |

### OpenCLIP — Appearance (accuracy + mean confidence)

| model             | dataset   | axis       | transform   |   level |   accuracy |   mean_confidence |    n |
|:------------------|:----------|:-----------|:------------|--------:|-----------:|------------------:|-----:|
| openclip_vit_b_32 | cifar10   | appearance | blur        |       0 |     0.8885 |          0.110567 | 2000 |
| openclip_vit_b_32 | cifar10   | appearance | blur        |       1 |     0.602  |          0.104962 | 2000 |
| openclip_vit_b_32 | cifar10   | appearance | blur        |       2 |     0.217  |          0.101794 | 2000 |
| openclip_vit_b_32 | cifar10   | appearance | blur        |       3 |     0.1505 |          0.101768 | 2000 |
| openclip_vit_b_32 | cifar10   | appearance | pixelate    |       0 |     0.8885 |          0.110567 | 2000 |
| openclip_vit_b_32 | cifar10   | appearance | pixelate    |       1 |     0.318  |          0.10308  | 2000 |
| openclip_vit_b_32 | cifar10   | appearance | pixelate    |       2 |     0.1725 |          0.102662 | 2000 |
| openclip_vit_b_32 | cifar10   | appearance | pixelate    |       3 |     0.1415 |          0.102683 | 2000 |

### OpenCLIP — Geometry (accuracy + mean confidence)

| model             | dataset   | axis     | transform   |   level |   accuracy |   mean_confidence |    n |
|:------------------|:----------|:---------|:------------|--------:|-----------:|------------------:|-----:|
| openclip_vit_b_32 | cifar10   | geometry | rotate      |       0 |     0.8885 |          0.110567 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | rotate      |       1 |     0.66   |          0.106419 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | rotate      |       2 |     0.433  |          0.104568 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | rotate      |       3 |     0.5115 |          0.10515  | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | shear_x     |       0 |     0.8885 |          0.110567 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | shear_x     |       1 |     0.871  |          0.109625 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | shear_x     |       2 |     0.8445 |          0.109067 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | shear_x     |       3 |     0.7945 |          0.108393 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | translate   |       0 |     0.8885 |          0.110567 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | translate   |       1 |     0.869  |          0.110126 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | translate   |       2 |     0.825  |          0.109326 | 2000 |
| openclip_vit_b_32 | cifar10   | geometry | translate   |       3 |     0.7835 |          0.108449 | 2000 |

## Embedding drift summaries

### EfficientNet — Appearance

| model             | axis       | transform   |   level |   mean_cosine_similarity |   mean_drift |    n |
|:------------------|:-----------|:------------|--------:|-------------------------:|-------------:|-----:|
| efficientnet_v2_s | appearance | blur        |       0 |                 1        |     0        | 2000 |
| efficientnet_v2_s | appearance | blur        |       1 |                 0.677783 |     0.322217 | 2000 |
| efficientnet_v2_s | appearance | blur        |       2 |                 0.383661 |     0.616339 | 2000 |
| efficientnet_v2_s | appearance | blur        |       3 |                 0.261893 |     0.738107 | 2000 |
| efficientnet_v2_s | appearance | pixelate    |       0 |                 1        |     0        | 2000 |
| efficientnet_v2_s | appearance | pixelate    |       1 |                 0.464866 |     0.535134 | 2000 |
| efficientnet_v2_s | appearance | pixelate    |       2 |                 0.212839 |     0.787161 | 2000 |
| efficientnet_v2_s | appearance | pixelate    |       3 |                 0.103973 |     0.896027 | 2000 |

### EfficientNet — Geometry

| model             | axis     | transform   |   level |   mean_cosine_similarity |   mean_drift |    n |
|:------------------|:---------|:------------|--------:|-------------------------:|-------------:|-----:|
| efficientnet_v2_s | geometry | rotate      |       0 |                 1        |    0         | 2000 |
| efficientnet_v2_s | geometry | rotate      |       1 |                 0.590469 |    0.409531  | 2000 |
| efficientnet_v2_s | geometry | rotate      |       2 |                 0.515768 |    0.484232  | 2000 |
| efficientnet_v2_s | geometry | rotate      |       3 |                 0.670327 |    0.329673  | 2000 |
| efficientnet_v2_s | geometry | shear_x     |       0 |                 1        |    0         | 2000 |
| efficientnet_v2_s | geometry | shear_x     |       1 |                 0.944156 |    0.0558442 | 2000 |
| efficientnet_v2_s | geometry | shear_x     |       2 |                 0.893532 |    0.106468  | 2000 |
| efficientnet_v2_s | geometry | shear_x     |       3 |                 0.827927 |    0.172073  | 2000 |
| efficientnet_v2_s | geometry | translate   |       0 |                 1        |    0         | 2000 |
| efficientnet_v2_s | geometry | translate   |       1 |                 0.935426 |    0.0645743 | 2000 |
| efficientnet_v2_s | geometry | translate   |       2 |                 0.896453 |    0.103547  | 2000 |
| efficientnet_v2_s | geometry | translate   |       3 |                 0.850004 |    0.149996  | 2000 |

### OpenCLIP — Appearance

| axis       | transform   |   level |   mean_cosine_similarity |   mean_drift |    n |
|:-----------|:------------|--------:|-------------------------:|-------------:|-----:|
| appearance | blur        |       0 |                 1        |     0        | 2000 |
| appearance | blur        |       1 |                 0.741176 |     0.258824 | 2000 |
| appearance | blur        |       2 |                 0.582907 |     0.417093 | 2000 |
| appearance | blur        |       3 |                 0.565264 |     0.434736 | 2000 |
| appearance | pixelate    |       0 |                 1        |     0        | 2000 |
| appearance | pixelate    |       1 |                 0.640729 |     0.359271 | 2000 |
| appearance | pixelate    |       2 |                 0.521409 |     0.478591 | 2000 |
| appearance | pixelate    |       3 |                 0.48834  |     0.51166  | 2000 |

### OpenCLIP — Geometry

| axis     | transform   |   level |   mean_cosine_similarity |   mean_drift |    n |
|:---------|:------------|--------:|-------------------------:|-------------:|-----:|
| geometry | rotate      |       0 |                 1        |    0         | 2000 |
| geometry | rotate      |       1 |                 0.753703 |    0.246297  | 2000 |
| geometry | rotate      |       2 |                 0.688415 |    0.311585  | 2000 |
| geometry | rotate      |       3 |                 0.779726 |    0.220274  | 2000 |
| geometry | shear_x     |       0 |                 1        |    0         | 2000 |
| geometry | shear_x     |       1 |                 0.933036 |    0.066964  | 2000 |
| geometry | shear_x     |       2 |                 0.898772 |    0.101228  | 2000 |
| geometry | shear_x     |       3 |                 0.86447  |    0.13553   | 2000 |
| geometry | translate   |       0 |                 1        |    0         | 2000 |
| geometry | translate   |       1 |                 0.917055 |    0.0829451 | 2000 |
| geometry | translate   |       2 |                 0.900809 |    0.0991909 | 2000 |
| geometry | translate   |       3 |                 0.851561 |    0.148439  | 2000 |

### DINOv2 — Appearance

| model         | axis       | transform   |   level |   mean_cosine_similarity |   mean_drift |    n |
|:--------------|:-----------|:------------|--------:|-------------------------:|-------------:|-----:|
| dinov2_vits14 | appearance | blur        |       0 |                1         |     0        | 2000 |
| dinov2_vits14 | appearance | blur        |       1 |                0.646119  |     0.353881 | 2000 |
| dinov2_vits14 | appearance | blur        |       2 |                0.350895  |     0.649105 | 2000 |
| dinov2_vits14 | appearance | blur        |       3 |                0.233896  |     0.766104 | 2000 |
| dinov2_vits14 | appearance | pixelate    |       0 |                1         |     0        | 2000 |
| dinov2_vits14 | appearance | pixelate    |       1 |                0.422403  |     0.577597 | 2000 |
| dinov2_vits14 | appearance | pixelate    |       2 |                0.0821324 |     0.917868 | 2000 |
| dinov2_vits14 | appearance | pixelate    |       3 |                0.0544942 |     0.945506 | 2000 |

### DINOv2 — Geometry

| model         | axis     | transform   |   level |   mean_cosine_similarity |   mean_drift |    n |
|:--------------|:---------|:------------|--------:|-------------------------:|-------------:|-----:|
| dinov2_vits14 | geometry | rotate      |       0 |                 1        |    0         | 2000 |
| dinov2_vits14 | geometry | rotate      |       1 |                 0.531621 |    0.468379  | 2000 |
| dinov2_vits14 | geometry | rotate      |       2 |                 0.371233 |    0.628767  | 2000 |
| dinov2_vits14 | geometry | rotate      |       3 |                 0.461886 |    0.538114  | 2000 |
| dinov2_vits14 | geometry | shear_x     |       0 |                 1        |    0         | 2000 |
| dinov2_vits14 | geometry | shear_x     |       1 |                 0.887533 |    0.112467  | 2000 |
| dinov2_vits14 | geometry | shear_x     |       2 |                 0.814279 |    0.185721  | 2000 |
| dinov2_vits14 | geometry | shear_x     |       3 |                 0.735832 |    0.264168  | 2000 |
| dinov2_vits14 | geometry | translate   |       0 |                 1        |    0         | 2000 |
| dinov2_vits14 | geometry | translate   |       1 |                 0.93334  |    0.0666597 | 2000 |
| dinov2_vits14 | geometry | translate   |       2 |                 0.894031 |    0.105969  | 2000 |
| dinov2_vits14 | geometry | translate   |       3 |                 0.839343 |    0.160657  | 2000 |

