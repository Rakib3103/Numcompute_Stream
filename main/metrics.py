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

def safe_divide(numerator, denominator):
    """
    Safely divide two numbers.

    Returns 0.0 when the denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


class StreamingAccuracy:
    """
    Streaming accuracy metric.

    Accuracy = correct predictions / total predictions.
    """

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        """
        Update accuracy using a new chunk.
        """
        y_true, y_pred = _validate_labels(y_true, y_pred)

        self.correct += int(np.sum(y_true == y_pred))
        self.total += int(y_true.shape[0])

        return self

    def result(self):
        """
        Return current cumulative accuracy.
        """
        return safe_divide(self.correct, self.total)

    def reset(self):
        """
        Reset stored metric state.
        """
        self.correct = 0
        self.total = 0
        return self


class StreamingPrecision:
    """
    Streaming precision metric.

    For binary classification:

    Precision = TP / (TP + FP)

    By default, positive_label=1.
    """

    def __init__(self, positive_label=1):
        self.positive_label = positive_label
        self.reset()

    def update(self, y_true, y_pred):
        """
        Update precision using a new chunk.
        """
        y_true, y_pred = _validate_labels(y_true, y_pred)

        positive_predictions = y_pred == self.positive_label
        true_positive = (y_true == self.positive_label) & positive_predictions
        false_positive = (y_true != self.positive_label) & positive_predictions

        self.tp += int(np.sum(true_positive))
        self.fp += int(np.sum(false_positive))

        return self

    def result(self):
        """
        Return current cumulative precision.
        """
        return safe_divide(self.tp, self.tp + self.fp)

    def reset(self):
        """
        Reset stored metric state.
        """
        self.tp = 0
        self.fp = 0
        return self