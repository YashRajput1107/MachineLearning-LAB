# 🚗 AutoValue AI — Used Car Price Predictor & Deal Checker

**Author:** Yash Rajput &nbsp;|&nbsp; **Registration No:** 2427010652

AutoValue AI estimates a used car's resale price from its details using a trained machine-learning
model, then compares that estimate with a seller's asking price. It's an ML mini-project covering the
full lifecycle: EDA, preprocessing, comparison of three regression models, evaluation, an interactive
Streamlit app, tests and a report.

> **Historical data, educational estimate.** The model is trained on CarDekho listings from
> **1983–2020**. Its estimates reflect those historical prices, not today's market, and the app
> only accepts manufacturing years inside that range.

---

## Results

Models were compared with **5-fold cross-validation on the training set** and the winner was chosen
by **lowest CV MAE**. It was then evaluated **once** on a held-out test set of 1,384 cars. All errors
are in rupees.

| Model | CV MAE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| **Linear Regression** (selected) | ₹84,134 ± 5,386 | 0.825 | **₹84,331** | ₹2,13,694 | **0.805** |
| Gradient Boosting | ₹84,433 ± 4,285 | 0.800 | ₹84,772 | ₹2,28,885 | 0.777 |
| Random Forest | ₹90,489 ± 3,635 | 0.776 | ₹92,388 | ₹2,38,679 | 0.757 |

- Linear Regression and Gradient Boosting are **effectively tied**: the difference is far smaller
  than the CV spread. Linear Regression wins by the selection rule, and it's also the simplest and
  most interpretable model. Trained on log(price) with brand and model encoded, it behaves like a
  multiplicative depreciation model, which suits this data.
- **RMSE is much larger than MAE** because a few expensive cars have large absolute errors.
- **Robustness check (grouped by car model):** when validation folds contain only car models the
  model never saw, MAE rises to **₹2,18,995** (R² 0.464). The headline result applies to car
  models represented in the training data, which is how the app is used: it only offers known
  models and warns when a model has few examples.
- **Permutation importance:** car **model**, **brand** and **year** matter most. Km driven,
  transmission, owner and seller type add much less once those are known.

---

## Data

- **Source:** [Vehicle dataset from CarDekho — Kaggle (nehalbirla)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho),
  file `Car details v3.csv` (8,128 rows × 13 columns).
- **Cleaning** (`src/preprocessing.py`), 8,128 → **6,918** rows:
  - 1,203 exact duplicates on the modelled columns removed *before* splitting, so the same car
    can't appear in both train and test
  - 5 "Test Drive Car" rows excluded (they don't fit the previous-owner categories)
  - 2 implausible odometer readings above 10 lakh km removed
- **Inputs vs features:** the app takes the **7 original input fields** (`name, year, km_driven,
  fuel, seller_type, transmission, owner`). `name` has 2,058 variants, most with very few rows, so
  it is split into **brand** (32) and **model** (212). The model therefore receives **8 features**.
  Engine, mileage, power, torque and seats are not used in v1.
- **Target:** `selling_price`, trained on log(price) because prices are highly skewed (₹30k – ₹1 crore).

---

## Project structure

```
AutoValue-AI/
├── data/raw/                 Car details v3.csv (downloaded, gitignored)
├── data/processed/           (empty — preprocessing lives in the pipeline)
├── notebooks/                01_data_exploration · 02_model_training
├── src/
│   ├── preprocessing.py      load, clean, brand/model features, ColumnTransformer, input validation
│   ├── train.py              models, cross-validation, evaluation, grouped check, importance, plots
│   ├── predict.py            load pipeline, predict price, deal comparison, ₹ formatting
│   └── download_data.py      fetch the dataset
├── models/                   best_model.joblib, metrics.json, model_comparison.csv (generated)
├── reports/figures/          EDA and evaluation charts
├── tests/test_prediction.py
├── app.py                    Streamlit app
└── Project_Report.pdf
```

`notebooks/02_model_training.ipynb` imports the functions in `src/train.py`, so the notebook, the
saved model, the app and the report all share one implementation and one set of results.

---

## Setup and usage

```bash
pip install -r requirements.txt
python -m src.download_data      # fetch Car details v3.csv into data/raw/
python -m src.train              # train, evaluate, save models/ and reports/figures/
pytest tests/                    # run the tests
streamlit run app.py             # launch the app
```

### The app
- **Price Estimator:** brand → model, year, km driven, fuel, seller type, transmission and ownership.
  Inputs are validated (e.g. negative km, years outside 1983–2020). The estimate is shown
  prominently, followed by a separate note on the model's test MAE. Any range shown is labelled
  *illustrative, not a statistical prediction interval*.
- **Deal Checker:** compares the estimate with the asking price, giving the difference and % of the
  estimate, plus a label: < −5% *Potentially below estimate*, within ±5% *Close to estimate*,
  > +5% *Potentially above estimate*. These thresholds are a project design choice, not a market
  standard.
- **Model Insights:** test metrics, model comparison, robustness check, evaluation charts, data
  source and limitations.

---

## Limitations
- Historical data only (up to 2020); no newer cars or current market prices.
- Mechanical condition, accident and service history, location and other factors are not modelled.
- Rare car models and small categories (CNG, LPG, Trustmark Dealer) have few examples, so their
  estimates are less reliable.
- The estimate is educational. It is not a valuation or a buy/don't-buy recommendation.
