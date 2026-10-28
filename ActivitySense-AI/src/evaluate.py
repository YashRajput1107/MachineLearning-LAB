"""Metrics and comparison helpers."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, classification_report)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def per_class_recall(y_true, y_pred, labels):
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return dict(zip(labels, rec.round(3)))


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)


def report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0)
