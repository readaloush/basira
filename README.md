# Basira · بصيرة

**Brain MRI tumor classification, evaluated without the leakage that inflates public benchmarks.**
Four classes: glioma, meningioma, pituitary tumor, no tumor.

**Demo:** [readaloush.github.io/basira](https://readaloush.github.io/basira): drop an MRI slice; the classifier runs in your browser (ONNX Runtime Web, 40 MB), and when it finds a tumor a second model (12 MB) outlines it.

Public brain-tumor MRI sets are routinely reported above 99% accuracy. This project measures how much of that comes from near-duplicate images shared between training and test, and trains a model under a protocol that removes them.

## Findings

| | |
|---|---|
| **30%** | of the test images of the Kaggle brain-tumor set **version 2** (2026, released as de-duplicated) still have a near-identical (correlation ≥ 0.98) twin in its training folder; 362 of the 400 no-tumor test images. Version 1: 103 byte-identical copies, 38% near-identical. |
| **107** | Kaggle test images appear byte for byte in the training split of **BRISC 2025**, so BRISC is not an independent test for models trained on the Kaggle set. |
| **22%** | of test gliomas from other sources were recognised by an EfficientNetV2 scoring 95.9% overall: every training glioma is a 512×512 figshare slice, and the model learned the source (gliomas from figshare: 99.4%). |
| **48% → 91%** | accuracy on held-out T1 / contrast T1 / T2 slices when training includes those sequences; models trained on the standard sets recognise 13% of gliomas imaged that way. |

## Protocol

1. Pool the training sets and remove every training image with an exact copy or a near twin (pHash Hamming ≤ 10 and normalised correlation ≥ 0.90) in **any** test set.
2. Group near-duplicates; split validation by group.
3. Choose the model and the "no tumor" threshold on the BRISC test; score the Kaggle test once with those choices fixed.
4. Say "no tumor" only when p(no tumor) ≥ 0.90, chosen on BRISC to miss fewer tumors.

## Results (Kaggle v2 test, 1,600 images, every image without a twin in training)

| Model | Accuracy | Tumors missed (of 1,200) | False alarms (of 400) |
|---|---|---|---|
| Original MobileNetV2 recipe, official split (twin-assisted) | 92.0 | 40 | 2 |
| EfficientNetV2-S 384, Kaggle + BRISC, 3 seeds | 96.06 ± 0.28 | 15.0 ± 1.0 | 6.3 ± 3.1 |
| EfficientNetV2-S 384, + T1/T2 data, 3 seeds | 95.94 ± 0.23 | 9.7 ± 3.8 | 11.0 ± 2.0 |
| Same, three seeds averaged | **96.44** | **3** | 19 |

## Tumor outline (BRISC 2025 segmentation test, 860 tumors)

U-Net with an EfficientNet-B0 encoder, 320 px, trained on the radiologist masks of BRISC 2025 after the same twin removal: **984 of the 3,933 BRISC segmentation training images (25%) have a near-identical twin in a test set** and were removed.

| | Dice | IoU | Tumors found |
|---|---|---|---|
| All | **0.859** | 0.787 | 98.9% |
| Meningioma (306) | 0.930 | 0.883 | 99.0% |
| Pituitary (300) | 0.866 | 0.787 | 99.7% |
| Glioma (254) | 0.763 | 0.673 | 98.0% |

Outline drawn on normal slices: 1 of 140 (BRISC test), 12% of 400 (Kaggle test, other sources); the demo runs the outline only when the classifier reports a tumor. The browser model (fp16 weights, no flip averaging) gives Dice 0.854.

Accuracy on the Kaggle test is ~96% whichever way the training data is extended; adding other sequences changes safety (missed tumors, other sources, other sequences), not this number.

## Notebooks (Kaggle, T4)

| | |
|---|---|
| 01 | Leakage audit of dataset v1 and v2; the original MobileNetV2 recipe by twin tier |
| 02 | ConvNeXt-T / EfficientNetV2-S with group-split validation |
| 03 | Error analysis (source fingerprint), BRISC audit, BRISC as external test |
| 04-05 | Multi-source pools with test twins removed; ablation K / KB / KBS |
| 06 | 384 px models, choices on BRISC, Kaggle scored once |
| 07-08 | T1/T2 data (Feltrin), three seeds, ensembles, abstention |
| 09 | ONNX export with class-activation output, int8 check |
| 10 | Pack the demo model as text parts |
| 11 | Tumor outline: U-Net on BRISC masks, same clean protocol, browser export |

`notebooks/build_XX.py` generates each notebook. Results: `results/`. Data is downloaded inside the notebooks and not stored here.

## Data

Kaggle brain tumor MRI dataset (M. Nickparvar, CC BY 4.0), BRISC 2025 (Fateh et al., Scientific Data, CC BY 4.0), Brain Tumor Classification (S. Bhuvaji et al.), Brain Tumor MRI Images 17 Classes (F. Feltrin). Example images in the demo come from the Kaggle set's test folder.

## Not a medical device

For research and education. Single 2D slices, not volumes; no T1/T2 pituitary images in training; twin removal catches near-identical slices, not every slice of the same patient.
