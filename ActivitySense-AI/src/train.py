"""End-to-end training: tune KNN/SVM/RF across 4 feature groups, evaluate on the
official test set, and save models, metrics, predictions and figures.

Run from the project root:  python -m src.train
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import warnings
warnings.filterwarnings("ignore")

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from .data_loader import load_train, load_test, load_features
from .feature_selection import get_feature_groups
from .preprocessing import make_pipeline, param_grid
from .evaluate import classification_metrics, confusion
from . import visualize as V


def tune_and_eval(model_name, Xtr, ytr, Xte, yte, cols):
    """GridSearchCV on train, clean single-fit timing, evaluate on test."""
    pipe = make_pipeline(model_name)
    if model_name == "svm":
        pipe.set_params(clf__probability=False)        # faster search
    gs = GridSearchCV(pipe, param_grid(model_name), cv=3,
                      scoring="accuracy", n_jobs=-1)
    gs.fit(Xtr[cols], ytr)

    est = clone(gs.best_estimator_)
    t0 = time.perf_counter()
    est.fit(Xtr[cols], ytr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = est.predict(Xte[cols])
    predict_time = time.perf_counter() - t0

    m = classification_metrics(yte, y_pred)
    m.update({"model": model_name, "n_features": len(cols),
              "train_time_s": round(train_time, 3),
              "predict_time_s": round(predict_time, 4),
              "best_params": str(gs.best_params_)})
    return est, m, y_pred


def main():
    print("Loading UCI HAR...")
    Xtr, ytr_i, ytr, subj_tr = load_train()
    Xte, yte_i, yte, subj_te = load_test()
    feature_names = load_features()
    labels = [C.ACTIVITY_MAP[i] for i in range(1, 7)]
    print(f"  train {Xtr.shape}, test {Xte.shape}")

    groups = get_feature_groups(Xtr, ytr, feature_names, k=C.SELECTK)
    print("  feature groups:", {k: len(v) for k, v in groups.items()})

    # ---- 3 models x 4 feature groups ---------------------------------------
    rows, full_models = [], {}
    for gname in C.FEATURE_GROUPS:
        cols = groups[gname]
        for mname in C.MODEL_NAMES:
            est, m, y_pred = tune_and_eval(mname, Xtr, ytr, Xte, yte, cols)
            m["feature_group"] = gname
            rows.append(m)
            if gname == "full":
                full_models[mname] = (est, y_pred)
            print(f"  {gname:14s} {mname:14s} acc={m['accuracy']:.3f} "
                  f"macroF1={m['macro_f1']:.3f} train={m['train_time_s']}s")

    metrics_df = pd.DataFrame(rows)[["model", "feature_group", "n_features",
                                     "accuracy", "macro_f1", "weighted_f1",
                                     "train_time_s", "predict_time_s", "best_params"]]
    metrics_df.to_csv(C.RESULTS_DIR / "metrics.csv", index=False)

    # ---- save full-feature models for the app ------------------------------
    for mname, (est, _) in full_models.items():
        if mname == "svm":
            # refit with probability enabled so the app can show confidence
            best = eval(metrics_df[(metrics_df.model == "svm") &
                                   (metrics_df.feature_group == "full")]
                        ["best_params"].iloc[0])
            est = Pipeline([("scaler", StandardScaler()),
                            ("clf", SVC(kernel="rbf", probability=True,
                                        random_state=C.RANDOM_SEED,
                                        C=best["clf__C"], gamma=best["clf__gamma"]))])
            est.fit(Xtr[groups["full"]], ytr)
        joblib.dump(est, C.MODELS_DIR / f"{mname}.joblib")

    # ---- manual-entry model (small interpretable subset) -------------------
    manual = RandomForestClassifier(n_estimators=200, random_state=C.RANDOM_SEED, n_jobs=-1)
    manual.fit(Xtr[C.MANUAL_FEATURES], ytr)
    joblib.dump(manual, C.MODELS_DIR / "manual_input_model.joblib")

    # ---- best full-feature model -> predictions.csv + confusion fig --------
    full_only = metrics_df[metrics_df.feature_group == "full"]
    best_model = full_only.sort_values("accuracy", ascending=False)["model"].iloc[0]
    best_est, best_pred = full_models[best_model]
    pred_df = pd.DataFrame({"y_true": yte.values, "y_pred": best_pred,
                            "subject": subj_te.values})
    pred_df["correct"] = pred_df.y_true == pred_df.y_pred
    pred_df.to_csv(C.RESULTS_DIR / "predictions.csv", index=False)

    # ---- metadata for the app ----------------------------------------------
    manual_ranges = {f: [float(Xtr[f].quantile(0.02)), float(Xtr[f].quantile(0.5)),
                         float(Xtr[f].quantile(0.98))] for f in C.MANUAL_FEATURES}
    meta = {"feature_cols_full": groups["full"], "feature_groups": groups,
            "manual_features": C.MANUAL_FEATURES, "manual_ranges": manual_ranges,
            "labels": labels, "best_model": best_model,
            "test_accuracy": float(full_only["accuracy"].max())}
    joblib.dump(meta, C.MODELS_DIR / "metadata.joblib")

    # ---- figures ------------------------------------------------------------
    V.plot_confusion(confusion(yte, best_pred, labels), labels,
                     C.FIGURES_DIR / "confusion_matrix.png",
                     title=f"Confusion matrix - {best_model} (full features)")
    V.plot_model_comparison(full_only, C.FIGURES_DIR / "model_comparison.png")
    V.plot_group_comparison(metrics_df, C.FIGURES_DIR / "feature_group_comparison.png")
    rf_full = full_models["random_forest"][0]
    imp = pd.Series(rf_full.feature_importances_, index=groups["full"])
    V.plot_feature_importance(imp, C.FIGURES_DIR / "feature_importance.png")
    V.plot_activity_distribution(ytr.value_counts(),
                                 C.FIGURES_DIR / "activity_distribution.png")

    print(f"\nBest full-feature model: {best_model} "
          f"(test accuracy {meta['test_accuracy']:.3f})")
    print("Saved models to", C.MODELS_DIR, "and results to", C.RESULTS_DIR)
    return metrics_df


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
