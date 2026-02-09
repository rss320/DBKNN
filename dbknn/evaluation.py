"""Evaluation metrics (numpy only)."""

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def evaluate_f1(
    predicted: NDArray[np.integer],
    true_labels: NDArray[np.integer],
) -> dict[str, float]:
    """Compute precision, recall, and F1 score.

    Parameters
    ----------
    predicted : array of int (1=cluster, 0=matrix)
    true_labels : array of int (1=cluster, 0=matrix)
    """
    pred_bool = predicted.astype(bool)
    true_bool = true_labels.astype(bool)
    tp = int((pred_bool & true_bool).sum())
    fp = int((pred_bool & ~true_bool).sum())
    fn = int((~pred_bool & true_bool).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    log.info("TP=%d, FP=%d, FN=%d", tp, fp, fn)
    log.info("Precision=%.4f, Recall=%.4f, F1=%.4f", precision, recall, f1)
    return {"precision": precision, "recall": recall, "f1": f1}
