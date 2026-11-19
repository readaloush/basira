# 01 · Leakage audit (Kaggle, 18 Nov 2026)

Near-duplicates: pHash Hamming <= 10, confirmed by normalised cross-correlation (64x64, grayscale) >= 0.90.
"Direct twin" = best correlation between a test image and any training image (no chaining).
Model: original MobileNetV2 recipe (last 30 layers unfrozen, Adam 1e-4, 10 epochs, early stopping), seed 42, T4.

## Dataset audit
| | v1 (5,712 + 1,311) | v2 (5,600 + 1,600, "de-duplicated") |
|---|---|---|
| Test images byte-identical to a training image | 103 (7.9%) | 0 |
| Test images with a near-identical twin (>= 0.98) | 503 (38.4%) | 479 (29.9%) |
| very similar (0.95-0.98) | 202 | 202 |
| similar (0.90-0.95) | 307 | 329 |
| no twin | 299 (22.8%) | 590 (36.9%) |
| Images with an exact copy anywhere in the set | 491 | 340 |
| Near-duplicate groups carrying two labels | 17 (165 images) | 20 (201 images) |

v2 test, near-identical twins by class: glioma 10, meningioma 105, notumor 362 / 400, pituitary 2.
v2 test, no twin by class: glioma 245, meningioma 176, notumor 26, pituitary 143.

## Accuracy by direct-twin tier (official test set)
| Tier | v1 n | v1 acc | v2 n | v2 acc |
|---|---|---|---|---|
| no twin | 299 | 92.6 | 590 | 83.6 |
| similar | 307 | 95.1 | 329 | 95.7 |
| very similar | 202 | 99.0 | 202 | 97.5 |
| near-identical | 503 | 98.0 | 479 | 97.5 |
| all | 1,311 | 96.3 | 1,600 | 92.0 |

v2 glioma with no twin: 66.1% (n=245). v1 glioma: 92.0 (no twin) -> 100 (near-identical).

## Other
- Run-to-run variance: two v2 runs with the same seed gave 92.56% and 92.00% on the official test (GPU non-determinism).
- v2 clean split (near-duplicate groups kept on one side; label-conflict groups dropped): 94.84% / 93.84% (two runs), 1,007 test images.
- Caveat: a correlation of 0.98 on 64x64 thumbnails may still pair different but very similar slices, especially in the no-tumor class; pairs need a visual check.
