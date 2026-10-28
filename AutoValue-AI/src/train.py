"""Train, compare and evaluate the regression models; save the best pipeline.

All functions here are reused by notebooks/02_model_training.ipynb, so the
notebook, the app and the report share one implementation.

Usage:  python -m src.train
"""
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import (FEATURES, FIGURES_DIR, METRICS_PATH, MODEL_PATH, MODELS_DIR,
                            RANDOM_SEED, TARGET, build_preprocessor, clean, load_raw)

MODEL_LABELS = {"linear_regression": "Linear Regression",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting"}
SCORING = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error",
           "r2": "r2"}


# --------------------------------------------------------------------------- models
def _log_target(regressor, scale_numeric=False):
    """Preprocessing + regressor, trained on log1p(price); predictions in rupees."""
    pipe = Pipeline([("prep", build_preprocessor(scale_numeric)), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def make_models():
    """name -> (estimator, hyper-parameter grid)."""
    return {
        "linear_regression": (_log_target(LinearRegression(), scale_numeric=True), {}),
        "random_forest": (
            _log_target(RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
            {"regressor__reg__n_estimators": [200, 400],
             "regressor__reg__max_depth": [None, 20]}),
        "gradient_boosting": (
            _log_target(GradientBoostingRegressor(random_state=RANDOM_SEED, learning_rate=0.1)),
            {"regressor__reg__n_estimators": [200, 400],
             "regressor__reg__max_depth": [3, 5]}),
    }


# --------------------------------------------------------------------------- data
def load_clean_data():
    return clean(load_raw())


def split_data(df, test_size=0.2):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)


# --------------------------------------------------------------------------- training
def cross_validate_models(X_train, y_train, models=None, cv_folds=5):
    """Same 5-fold CV procedure for every model, on the training set only.
    Returns (fitted searches, comparison DataFrame)."""
    models = models or make_models()
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    searches, rows = {}, []
    for name, (estimator, grid) in models.items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring=SCORING, refit="mae", n_jobs=-1)
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        rows.append({
            "model": name,
            "cv_mae": -res["mean_test_mae"][i],
            "cv_mae_std": res["std_test_mae"][i],
            "cv_rmse": -res["mean_test_rmse"][i],
            "cv_r2": res["mean_test_r2"][i],
            "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
        })
        searches[name] = search
    table = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    return searches, table


def select_best(table):
    """Pick the model with the lowest cross-validation MAE (never the test set)."""
    return table.iloc[0]["model"]


def evaluate(model, X, y):
    pred = model.predict(X)
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}


def grouped_check(estimator, X_train, y_train, cv_folds=5):
    """Robustness check: GroupKFold by car model, so each fold's validation
    models were never seen during that fold's training."""
    gkf = GroupKFold(n_splits=cv_folds)
    res = cross_validate(clone(estimator), X_train, y_train, groups=X_train["model"],
                         cv=gkf, scoring=SCORING, n_jobs=-1)
    return {"mae": float(-res["test_mae"].mean()),
            "rmse": float(-res["test_rmse"].mean()),
            "r2": float(res["test_r2"].mean())}


def permutation_importance_table(model, X, y, n_repeats=5):
    """Permutation importance of the 8 original features on the full pipeline
    (increase in MAE, in rupees, when a feature is shuffled)."""
    result = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                                    n_repeats=n_repeats, random_state=RANDOM_SEED, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns,
                          "mae_increase": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("mae_increase", ascending=False).reset_index(drop=True))


def build_app_metadata(df_train):
    """Input ranges and category lists derived from the training data."""
    cats = {c: sorted(df_train[c].unique().tolist())
            for c in ["fuel", "seller_type", "transmission", "owner", "brand"]}
    brand_models = {b: sorted(g["model"].unique().tolist()) for b, g in df_train.groupby("brand")}
    return {
        "year_min": int(df_train["year"].min()),
        "year_max": int(df_train["year"].max()),
        "km_max": int(df_train["km_driven"].max()),
        "categories": cats,
        "brand_models": brand_models,
        "model_counts": df_train["model"].value_counts().to_dict(),
    }


# --------------------------------------------------------------------------- plots
def _save(fig, path):
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_mae_comparison(table, path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [MODEL_LABELS[m] for m in table["model"]]
    ax.bar(labels, table["cv_mae"], yerr=table["cv_mae_std"], capsize=6,
           color=["#4C72B0", "#55A868", "#DD8452"][:len(table)])
    for i, (v, sd) in enumerate(zip(table["cv_mae"], table["cv_mae_std"])):
        ax.text(i, v + sd * 1.15, f"₹{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, (table["cv_mae"] + table["cv_mae_std"]).max() * 1.12)
    ax.set_ylabel("Cross-validation MAE (₹)")
    ax.set_title("Model comparison - 5-fold CV MAE on training data (lower is better)")
    fig.tight_layout()
    return _save(fig, path)


def plot_actual_vs_predicted(y_true, y_pred, title, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true / 1e5, y_pred / 1e5, s=10, alpha=0.4, color="#4C72B0")
    lim = max(y_true.max(), y_pred.max()) / 1e5
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual price (₹ lakh)")
    ax.set_ylabel("Predicted price (₹ lakh)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_residuals(y_true, y_pred, path=None):
    resid = (y_true - y_pred) / 1e5
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(y_pred / 1e5, resid, s=10, alpha=0.4, color="#55A868")
    ax.axhline(0, color="red", lw=1.2, ls="--")
    ax.set_xlabel("Predicted price (₹ lakh)")
    ax.set_ylabel("Residual: actual - predicted (₹ lakh)")
    ax.set_title("Residuals on the test set")
    fig.tight_layout()
    return _save(fig, path)


def plot_importance(imp, path=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    d = imp.iloc[::-1]
    ax.barh(d["feature"], d["mae_increase"], xerr=d["std"], color="teal")
    ax.set_xlabel("Increase in test MAE when shuffled (₹)")
    ax.set_title("Permutation importance (best model)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- workflow
def main():
    df, clean_report = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"clean rows {len(df)} | train {len(X_train)} | test {len(X_test)}")
    print("cleaning:", clean_report)

    searches, table = cross_validate_models(X_train, y_train)
    print("\nCross-validation (training set):")
    print(table[["model", "cv_mae", "cv_rmse", "cv_r2"]].round(3).to_string(index=False))

    best_name = select_best(table)
    best_model = searches[best_name].best_estimator_
    test_metrics = {n: evaluate(s.best_estimator_, X_test, y_test) for n, s in searches.items()}
    print(f"\nSelected by CV MAE: {best_name}")
    print("Test metrics:", {n: {k: round(v, 3) for k, v in m.items()} for n, m in test_metrics.items()})

    grouped = grouped_check(searches[best_name].best_estimator_, X_train, y_train)
    print("Grouped-by-model CV:", {k: round(v, 3) for k, v in grouped.items()})

    imp = permutation_importance_table(best_model, X_test, y_test)

    # ---- save artifacts ----------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    comparison = table.copy()
    for k in ["mae", "rmse", "r2"]:
        comparison[f"test_{k}"] = comparison["model"].map(lambda m: test_metrics[m][k])
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    metrics = {
        "best_model": best_name,
        "best_model_label": MODEL_LABELS[best_name],
        "selection": "lowest 5-fold cross-validation MAE on the training set",
        "test_metrics": test_metrics[best_name],
        "all_test_metrics": test_metrics,
        "cv": table.drop(columns="best_params").to_dict(orient="records"),
        "best_params": {r["model"]: r["best_params"] for r in table.to_dict(orient="records")},
        "grouped_check": grouped,
        "permutation_importance": imp.to_dict(orient="records"),
        "cleaning": clean_report,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "app": build_app_metadata(pd.concat([X_train, y_train], axis=1)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))

    pred = best_model.predict(X_test)
    plot_mae_comparison(table, FIGURES_DIR / "model_comparison_mae.png")
    plot_actual_vs_predicted(y_test.values, pred,
                             f"Actual vs predicted - {MODEL_LABELS[best_name]} (test set)",
                             FIGURES_DIR / "actual_vs_predicted.png")
    plot_residuals(y_test.values, pred, FIGURES_DIR / "residuals.png")
    plot_importance(imp, FIGURES_DIR / "permutation_importance.png")
    plt.close("all")
    print(f"\nSaved {MODEL_PATH.name}, metrics.json, model_comparison.csv and figures.")
    return metrics


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
