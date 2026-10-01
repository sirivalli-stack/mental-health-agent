"""Classification metrics shared by every ML component (Phases 3-5).

Kept in one place so sentiment, emotion and risk report identical numbers:
same label ordering, same confusion-matrix orientation (rows = truth,
columns = prediction), same macro/weighted aggregation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def metrics_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Accuracy + per-class P/R/F1 + macro/weighted F1 + confusion matrix."""
    labels = list(labels)
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    support_arr = np.asarray(support, dtype=float)

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(labels)
    }
    total = float(support_arr.sum()) or 1.0

    return {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            np.average(f1, weights=support_arr) if support_arr.any() else 0.0
        ),
        "per_class": per_class,
        "label_order": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_share": {name: round(float(support_arr[i]) / total, 4)
                        for i, name in enumerate(labels)},
    }
