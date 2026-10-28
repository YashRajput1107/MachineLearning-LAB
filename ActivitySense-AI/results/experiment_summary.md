# Experiment Summary

**Project:** ActivitySense AI — Human Activity Recognition (UCI HAR)
**Author:** Yash Rajput (2427010652)
**Dataset:** UCI HAR — 7352 train / 2947 test samples, 561 features, 6 activities, 30 subjects
(official subject-wise split preserved).

## Preprocessing
- 561 feature names assigned from `features.txt`; 84 duplicate names made unique by index suffix.
- No missing/non-finite values; train/test columns match.
- Scaling done inside pipelines (StandardScaler → KNN/SVM); Random Forest unscaled.

## RQ1 — Model comparison (full 561 features, official test)
| Model | Accuracy | Macro-F1 | Weighted-F1 | Train (s) | Predict (s) |
|---|---|---|---|---|---|
| SVM (RBF) | 0.954 | 0.953 | 0.954 | 5.6 | 4.98 |
| Random Forest | 0.926 | 0.924 | 0.926 | 16.8 | 0.15 |
| KNN | 0.888 | 0.885 | 0.887 | 0.3 | 2.95 |

SVM most accurate; RF close and fastest at prediction; KNN weakest and slow at inference.

## RQ2 / RQ3 — Feature groups (accuracy)
| Group | #Feat | KNN | SVM | RF |
|---|---|---|---|---|
| Full | 561 | 0.888 | 0.954 | 0.926 |
| Accelerometer | 350 | 0.859 | 0.914 | 0.890 |
| Gyroscope | 213 | 0.700 | 0.803 | 0.811 |
| Reduced (SelectKBest, k=50) | 50 | 0.845 | 0.881 | 0.854 |

- **RQ2:** accelerometer features alone are far more informative than gyroscope-only.
- **RQ3:** 50 selected features retain most of the full-feature accuracy (SVM 0.881 vs 0.954).

## RQ4 — Activity confusion
Best model (SVM) confusion concentrates on the **static** activities, chiefly
**sitting ↔ standing**; the dynamic/static boundary is almost never crossed.

## Limitations
UCI HAR features are pre-engineered (so even simple models score high) and the manual-entry app
mode uses a separate small-feature model for demonstration. Confidence scores are not calibrated.
