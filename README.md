# MachineLearning-Lab

**Author:** Yash Rajput &nbsp;|&nbsp; **Registration No:** 2427010652

Coursework and mini projects for the AIML (Machine Learning) laboratory. The repository covers lab
experiments in data preprocessing, supervised learning and clustering, plus two end-to-end machine
learning projects with interactive Streamlit applications.

---

## Projects

| Project | Task | Dataset | Models | Result |
|---|---|---|---|---|
| [ActivitySense-AI](ActivitySense-AI/README.md) | Human activity recognition (6 classes) | UCI HAR (smartphone sensors) | KNN, SVM, Random Forest | SVM: ~95% test accuracy |
| [AutoValue-AI](AutoValue-AI/README.md) | Used car price prediction + deal checker | CarDekho *Car details v3* | Linear Regression, Random Forest, Gradient Boosting | Linear Regression: test MAE ₹84,331, R² 0.805 |

### ActivitySense-AI: Smartphone Sensor-Based Human Activity Recognition
Recognises walking, walking upstairs/downstairs, sitting, standing and laying from 561 engineered
accelerometer and gyroscope features. Compares three classifiers across four feature groups (full,
accelerometer, gyroscope, reduced) and serves the trained models in a five-page Streamlit app.

### AutoValue-AI: Used Car Price Predictor & Deal Checker
Estimates a used car's resale price from its details and compares the estimate with a seller's
asking price. Uses leakage-safe cleaning and cross-validated model selection, and evaluates the
selected model on held-out data. Includes a three-tab Streamlit app.

---

## Lab exercises (`notebooks/`)

| Notebook | Topic |
|---|---|
| `LAB1` | Handling missing data (imputation) |
| `LAB2` | Standardization & normalization |
| `LAB3` | Exploratory data analysis |
| `LAB4` | Random Forest classification |
| `LAB5` | Simple & multiple linear regression |
| `LAB6` | Decision tree classification |
| `LAB7` | K-Means clustering (elbow + silhouette) |
| `LAB8` | DBSCAN clustering |
| `lab9`, `Clustering_KMeans_DBSCAN_EEG` | Clustering (K-Means/K-Means++/DBSCAN) on EEG data |

> Some lab notebooks read CSV files that are not included in this repository; they are kept as a
> record of the lab work, with their outputs saved.

---

## Repository structure

```
MachineLearning-Lab/
├── notebooks/            Lab exercise notebooks
├── ActivitySense-AI/     Human activity recognition project (src/, app/, notebooks/, tests/, report)
├── AutoValue-AI/         Used car price prediction project (src/, app.py, notebooks/, tests/, report)
├── .gitignore
└── README.md
```

## Tech stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit · Joblib · pytest · Jupyter

---

## Getting started

Each project is self-contained with its own `requirements.txt` and README.

```bash
# ActivitySense-AI
cd ActivitySense-AI
pip install -r requirements.txt
# download the UCI HAR dataset (see the project README), then:
python -m src.train
streamlit run app/dashboard.py
```

```bash
# AutoValue-AI
cd AutoValue-AI
pip install -r requirements.txt
python -m src.download_data
python -m src.train
streamlit run app.py
```

## Notes
- Virtual environments, trained model files and raw datasets are not committed. Recreate them with
  each project's `requirements.txt`, download instructions and training script.
- Each project folder contains a PDF report with the methodology, results and limitations.
