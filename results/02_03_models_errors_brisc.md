# 02 · Stronger models (Kaggle v2 official test, 18 Nov 2026)
Group-split validation (no twins between train and val), brain crop, 288 px, AdamW + cosine, label smoothing, flip TTA. ~33 min on a T4.

| Model | All (1,600) | No twin in training (590) | Glioma, no twin (245) | Missed tumors |
|---|---|---|---|---|
| MobileNetV2 (original recipe, nb 01) | 92.0 | 83.6 | 66.1 | 40 |
| ConvNeXt-Tiny (in22k) | 95.50 | 89.15 | 74.29 | 24 |
| EfficientNetV2-S (in21k) | 95.88 | 90.17 | 76.73 | 32 |
| Ensemble of both | 95.75 | 89.83 | 75.92 | 27 |

Validation accuracy 99.6-99.9%: the validation set does not contain the hard images, so the abstention threshold chosen on it (0.58) answers 99% of test images and does not help.

# 03 · Error analysis and BRISC 2025
Ensemble confusion on Kaggle v2 test: glioma 339/400 correct; 35 -> meningioma, 26 -> no tumor. Other classes: 1-5 errors each.
Almost all errors are gliomas without a twin in training (59 of 61).

Source fingerprint (image size): all 1,400 training gliomas are 512x512 (figshare). 324 test gliomas are 512x512 -> 99.4% correct;
76 are other sizes (other sources) -> 22.4% correct. The figure shows many of the missed gliomas are T2/FLAIR or differently acquired
T1 slices; the training gliomas are all one kind (figshare, contrast T1).
-> The model learned "glioma = figshare-looking image". Class and source are confounded in the training set.

BRISC 2025 (5,000 train / 1,000 test, T1 only):
- BRISC test vs BRISC train: 7 exact copies, 109 near-identical (>=0.98), 182 very similar, 297 similar, 412 no twin.
- BRISC test vs Kaggle v2 train: 6 exact, 115 near-identical, 179 very similar, 288 similar, 418 no twin.
- BRISC train vs Kaggle v2 test: 107 exact copies of Kaggle test images inside BRISC train; 215 near-identical.

External test (nb 02 ensemble, never trained on BRISC) on BRISC test: 99.2% overall, 99.28% on images with no twin in Kaggle train,
glioma no-twin 97.7%, 1 missed tumor. Matches the 99.4% in the LinkedIn post without training on BRISC: BRISC test is the same
kind of image (T1, figshare-like). It does not test the case that fails on Kaggle (gliomas from other sources/sequences).

# 04 · Multi-source pool (Kaggle v2 + BRISC + SARTAJ), test twins removed from the pool
Removed for having a twin in a test set: kaggle 1,817 / 5,600, brisc 991 / 5,000, sartaj 860 / 3,264. Pool 8,886 train / 1,310 val.
Every test image is now "no twin" relative to training. EfficientNetV2-S, same recipe as nb 02. 32.5 min.

| | Kaggle test | BRISC test |
|---|---|---|
| Accuracy (all twin-free) | 93.56 | 95.10 |
| glioma | 86.8 | 99.6 |
| meningioma | 87.8 | 84.3 |
| notumor / pituitary | 100 / 99.8 | 100 / 100 |
| Missed tumors | 76 (38 glioma, 38 meningioma -> notumor) | 39 |

Kaggle gliomas from other sources: 35.5% (nb 02: 22.4%). Gliomas 512x512: 98.8%. Meningiomas from other sources: 62.4%.
nb 02 for reference: BRISC 99.2 (BRISC test had twins in Kaggle train), 27 missed tumors on Kaggle.
Several things changed at once (sources added AND 1,817 Kaggle training images removed), so this run cannot say which change did what.

# 05 · Source ablation (same twin removal; EfficientNetV2-S 288 px, 16 epochs)
| Pool | Train | Kaggle acc | Kaggle missed | BRISC acc | BRISC missed | Glioma other sources | Meningioma other sources |
|---|---|---|---|---|---|---|---|
| K (Kaggle) | 3,313 | 94.00 | 73 | 95.50 | 33 | 36.8 | 66.7 |
| KB (+BRISC) | 6,783 | **94.81** | **57** | **96.80** | **22** | 34.2 | 76.9 |
| KBS (+SARTAJ, nb 04) | 8,886 | 93.56 | 76 | 95.10 | 39 | 35.5 | 62.4 |

- The drop from nb 02 (95.9 / 99.2) to ~94-95 comes from removing training images that have twins in the test sets, not from SARTAJ:
  the Kaggle-only pool under the same removal scores the same as KBS. nb 02's numbers were partly twin-assisted.
- BRISC helps (more patients of the same kinds); SARTAJ hurts (known label inconsistencies).
- No pool fixes the gliomas from other sources (~35%): these are mostly T2/FLAIR or differently acquired slices that no training pool covers.

# 06 · Final model (KB pool, test twins removed, 384 px)
Choices made on the BRISC test (dev); Kaggle test scored once afterwards.
| Model | BRISC (dev) | Kaggle (final, argmax) |
|---|---|---|
| EfficientNetV2-S 384 | 97.50 | 95.25 |
| ConvNeXt-Small 384 | 97.10 | 95.19 |
| Ensemble | 97.50 | - |
Chosen: EfficientNetV2-S (ensemble no better on dev).

No-tumor threshold (say "no tumor" only if p(no tumor) >= t), on BRISC dev:
t 0.5: 97.7% / 15 missed; 0.8: 98.3 / 9; **0.9: 98.6 / 5 missed / 0 false alarms**; 0.95: 85.0 / 0 missed / 140 false alarms. Chosen 0.9.

| Kaggle test (final, never used for a choice) | argmax | with threshold 0.9 |
|---|---|---|
| Accuracy | 95.25 | **96.31** |
| Missed tumors (tumor -> no tumor) | 52 | **15** |
| False alarms (no tumor -> tumor), of 400 | 0 | 7 |
| Glioma / meningioma / no tumor / pituitary | 87.2 / 95.0 / 100 / 98.8 | 90.5 / 97.8 / 98.2 / 98.8 |
| Gliomas from other sources (76) | 35.5 | **52.6** |
| Meningiomas from other sources | 82.9 | 92.3 |
BRISC dev with threshold: 98.6%, 5 missed tumors, 0 false alarms. Run time 92 min on a T4.

Progression on the Kaggle v2 test: MobileNetV2 92.0 (twin-assisted) -> ... -> 96.31 with every training image that has a twin in the test removed.

# 07 · Other sequences (Feltrin: glioma, meningioma, normal in T1 / T1C+ / T2) and seed variance
Same clean protocol, EfficientNetV2-S 384. Feltrin held-out test: 729 images (20% of its near-duplicate groups; no patient ids,
so other slices of the same patient may be shared with training). Thresholds chosen on BRISC.

| Run | BRISC (dev) acc / missed | Kaggle acc / missed / false alarms | Kaggle gliomas, other sources | Feltrin acc (T1 / T1C+ / T2) |
|---|---|---|---|---|
| KB seed 42 (nb 06) | 98.6 / 5 | 96.31 / 15 / 7 | 52.6 | - |
| KB seed 7 | 98.6 / 4 | 95.75 / 16 / 3 | 32.9 | 48.0 (46.9 / 58.2 / 35.2) |
| KBF seed 42 | 97.3 / 3 | 95.88 / 7 / 11 | 64.5 | 90.1 (88.4 / 94.2 / 86.7) |
| KBF seed 7 | 97.9 / 2 | 96.19 / 8 / 13 | 61.8 | 91.8 (91.5 / 93.5 / 89.8) |
| **KBF, 2 seeds averaged** | 98.3 / **0** | 96.00 / **2** / 27 | **69.7** | **91.9** (90.3 / 94.9 / 89.8) |

Argmax (no threshold) Kaggle: KB s7 94.81 (56 missed), KBF s42 95.44 (29), KBF s7 95.25 (39).
Feltrin with KB s7 (argmax): 41.4%; gliomas 13.2%: a model trained on the standard sets does not recognise gliomas in T1/T2.

Abstention (KB s7, confidence threshold chosen on BRISC): target 99% -> Kaggle answers 96.4% of images, 97.15% correct;
target 99.5% -> answers 83.0%, 98.87% correct.

Reading:
- Kaggle accuracy is the same with or without Feltrin (KB 96.31 / 95.75, mean 96.0; KBF 95.88 / 96.19, mean 96.0):
  the ceiling of this test is ~96% for these models.
- What Feltrin changes is safety and robustness: missed tumors on Kaggle 15-16 -> 7-8 (2 with two seeds),
  gliomas from other sources 33-53% -> 62-70%, other sequences 48% -> 90-92%.
- Seed-to-seed spread on Kaggle accuracy: 0.3-0.6 points.

# 08 · Three seeds (1, 7, 42), mean ± std, and three-seed ensembles (EfficientNetV2-S 384, clean protocol)
| | KB (Kaggle + BRISC) | KBF (+ Feltrin T1/T1C+/T2) |
|---|---|---|
| Kaggle accuracy, threshold (final) | 96.06 ± 0.28 | 95.94 ± 0.23 |
| Kaggle accuracy, argmax | 95.04 ± 0.22 | 95.25 ± 0.19 |
| Kaggle missed tumors (of 1,200) | 15.0 ± 1.0 | 9.7 ± 3.8 |
| Kaggle false alarms (of 400) | 6.3 ± 3.1 | 11.0 ± 2.0 |
| Kaggle gliomas from other sources (76) | 45.6 ± 11.0 | 61.0 ± 4.0 |
| BRISC (dev) accuracy | 98.53 ± 0.12 | 97.57 ± 0.31 |
| Feltrin held-out (729; T1/T1C+/T2) | 48.4 ± 0.6 (2 seeds) | 90.8 ± 0.9 |

Three-seed ensembles (threshold 0.9 chosen on BRISC):
| | KB ensemble | KBF ensemble |
|---|---|---|
| Kaggle accuracy | **96.56** | 96.44 |
| Kaggle missed / false alarms | 8 / 5 | **3** / 19 |
| Kaggle gliomas, other sources | 48.7 | **71.1** |
| BRISC accuracy / missed / false alarms | 98.9 / 3 / 0 | 98.2 / 0 / 2 |
| Feltrin | 49.8 | **91.4** |

Note on choice: by the dev rule alone (BRISC accuracy) the KB ensemble would be chosen; BRISC contains only contrast T1, so it
cannot measure sequence robustness. The KBF ensemble is the recommended model for use; both are reported.

## Notebook 11 · Tumor outline (U-Net, EfficientNet-B0, 320 px, BRISC 2025 masks)

Kaggle version 371327848. Pool: BRISC segmentation train (3,933) + BRISC classification no-tumor train (1,067, empty masks).
Twin removal against all test sets (BRISC seg test, BRISC cls test, Kaggle v2 test): **984 of 3,933 BRISC segmentation training images removed** (25%), 7 of 1,067 no-tumor.
Train 3,507 / validation 502 (group split). 40 epochs, Dice + BCE, AdamW 5e-4, cosine; best validation Dice 0.860 (epoch 37).
Outline rule chosen on validation: threshold 0.4, drop regions under 0.1% of the slice.

| BRISC seg test | n | Dice | IoU | Tumors found |
|---|---|---|---|---|
| All | 860 | 0.8585 | 0.7874 | 98.95% |
| Glioma | 254 | 0.7628 | 0.6727 | 98.03% |
| Meningioma | 306 | 0.9304 | 0.8833 | 99.02% |
| Pituitary | 300 | 0.8663 | 0.7868 | 99.67% |

Outline drawn on no-tumor slices: BRISC test 1 / 140 (0.7%); Kaggle v2 test 48 / 400 (12%).
Browser model (fp16 weights, fp32 compute, no flip TTA): Dice 0.8539, IoU 0.7835, found 98.37%.
