"""
metrics.py

Streaming classification metrics for the NumCompute streaming framework.
"""

from collections import deque
import numpy as np


def _validate_labels(y_true, y_pred):
    """
    Validate and convert labels to NumPy arrays.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")

    if y_true.shape[0] == 0:
        raise ValueError("Metric update received an empty chunk.")

    return y_true, y_pred