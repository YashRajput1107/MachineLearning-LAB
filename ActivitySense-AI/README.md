# ActivitySense AI
### Smartphone Sensor-Based Human Activity Recognition (UCI HAR)

**Author:** Yash Rajput &nbsp;|&nbsp; **Registration No:** 2427010652

Recognises six human activities — **walking, walking upstairs, walking downstairs, sitting,
standing, laying** — from smartphone accelerometer and gyroscope features, and serves the trained
models through an interactive Streamlit app.

> **Research questions:** RQ1 — which of KNN / SVM / Random Forest best balances accuracy and cost?
> RQ2 — accelerometer vs gyroscope features? RQ3 — can a reduced feature set match the full 561?
> RQ4 — which activities are confused?

---

## Results (official UCI HAR test set, 2947 samples)

**Full 561-feature models (RQ1):**

| Model | Accuracy | Macro-F1 | Train time | Predict time |
|---|---|---|---|---|
| **SVM (RBF)** | **0.954** | 0.953 | 5.6 s | 4.98 s |
| Random Forest | 0.926 | 0.924 | 16.8 s | 0.15 s |
| KNN | 0.888 | 0.885 | 0.3 s | 2.95 s |

SVM is most accurate; Random Forest is close and **predicts fastest**; KNN trains instantly but is
slow at inference (it compares every test point to all training points).

**Feature groups (RQ2 / RQ3), accuracy:**

| Group | #Features | KNN | SVM | RF |
|---|---|---|---|---|
| Full | 561 | 0.888 | **0.954** | 0.926 |
| Accelerometer | 350 | 0.859 | 0.914 | 0.890 |
| Gyroscope | 213 | 0.700 | 0.803 | 0.811 |
| Reduced (SelectKBest) | 50 | 0.845 | 0.881 | 0.854 |

Accelerometer features alone clearly beat gyroscope-only (RQ2); a reduced 50-feature subset keeps
most of the full accuracy (RQ3). **RQ4:** remaining errors are mainly **sitting ↔ standing**.

> **Note:** scores shown in the app are model estimates, not calibrated probabilities.

---

## Project structure
```
ActivitySense-AI/
├── data/raw/UCI HAR Dataset/   (download - see below; gitignored)
├── notebooks/ 01_eda · 02_feature_experiments · 03_model_training · 04_evaluation
├── src/ config · data_loader · preprocessing · feature_selection · train · evaluate · predict · visualize
├── app/ dashboard.py · components.py      (Streamlit, 5 pages)
├── models/ knn · svm · random_forest · manual_input_model · metadata (.joblib)
├── results/ metrics.csv · predictions.csv · figures/
├── tests/ test_data_loader · test_features · test_predictions
├── requirements.txt · README.md · .gitignore · .env.example
```

| Module | Responsibility |
|---|---|
| `data_loader.py` | load official files, name 561 features (de-duplicated), map labels |
| `feature_selection.py` | accelerometer / gyroscope / reduced (SelectKBest) groups |
| `preprocessing.py` | validation + leakage-safe pipelines (scaler inside the pipeline) |
| `train.py` | tune KNN/SVM/RF × 4 feature groups, save models/metrics/figures |
| `evaluate.py` | accuracy, macro/weighted-F1, per-class recall, confusion |
| `predict.py` | load models, predict, validate uploaded CSVs |
| `app/dashboard.py` | 5-page app: Home, Predict, Model Comparison, Feature Analysis, History |

---

## Setup

```bash
pip install -r requirements.txt
```

**Download the dataset** (not committed) and extract into `data/raw/` so that
`data/raw/UCI HAR Dataset/` exists:

```bash
# fast mirror of the official UCI HAR Dataset
curl -L -o har.zip "https://github.com/MaxBenChrist/human-activity-dataset/blob/master/UCI%20HAR%20Dataset.zip?raw=true"
# then unzip har.zip into data/raw/
```
Original source: [UCI Machine Learning Repository — Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
(Anguita et al., 2013). The official **subject-wise train/test split** is preserved.

## Usage

```bash
python -m src.train              # train all models -> models/ and results/
pytest tests/                    # run the test suite
streamlit run app/dashboard.py   # launch the interactive app
```
Notebooks in `notebooks/` reproduce the analysis (run top to bottom).

---

## Methodology notes
- **Official split preserved** (subjects disjoint between train/test) — never a random row split.
- **No leakage:** `StandardScaler` lives inside each pipeline (fit on training folds only);
  `SelectKBest` is fitted on training data only.
- **Class imbalance** is mild; metrics include macro-F1 (equal weight per activity).
- The 561 UCI features are already heavily engineered, which is why even simple models do well.
- **Limitation:** manual entry in the app is a *demonstration* using a separate small-feature model;
  the main models need all 561 features. Confidence scores are not calibrated.
