"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import CLASSES


def macro_f1(y_true, y_pred):
    labels = sorted(set(np.asarray(y_true)) | set(np.asarray(y_pred)))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def report(y_true, y_pred):
    labels = sorted(set(np.asarray(y_true)) | set(np.asarray(y_pred)))
    names = [CLASSES[i] for i in labels]
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=names,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": names,
    }
