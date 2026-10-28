"""Reusable plotting helpers (backend not forced, so notebooks render inline)."""
import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = {"knn": "#4C72B0", "svm": "#DD8452", "random_forest": "#55A868"}


def _save(fig, path):
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_activity_distribution(counts, path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = counts.sort_values(ascending=False)
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_ylabel("Number of samples")
    ax.set_title("Activity class distribution (training set)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(cm, labels, path=None, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    thr = cm.max() / 2
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > thr else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(full_metrics, path=None):
    """Bar chart of accuracy & macro-F1 per model on the full feature set."""
    df = full_metrics.set_index("model")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df)); w = 0.35
    ax.bar(x - w / 2, df["accuracy"], w, label="Accuracy", color="#4C72B0")
    ax.bar(x + w / 2, df["macro_f1"], w, label="Macro-F1", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(df.index)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model comparison (full 561 features)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, (a, f) in enumerate(zip(df["accuracy"], df["macro_f1"])):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_group_comparison(metrics_df, path=None):
    """Accuracy per feature group, grouped by model."""
    pivot = metrics_df.pivot(index="feature_group", columns="model", values="accuracy")
    order = ["full", "accelerometer", "gyroscope", "reduced"]
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot)); w = 0.25
    for i, m in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * w, pivot[m], w, label=m, color=MODEL_COLORS.get(m))
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by feature group")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_importance(importances, path=None, top=15):
    imp = importances.sort_values(ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.values, color="teal")
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest feature importance (top {top})")
    fig.tight_layout()
    return _save(fig, path)
