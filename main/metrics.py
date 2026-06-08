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
class StreamingRecall:
    """
    Streaming recall metric.

    For binary classification:

    Recall = TP / (TP + FN)

    By default, positive_label=1.
    """

    def __init__(self, positive_label=1):
        self.positive_label = positive_label
        self.reset()

    def update(self, y_true, y_pred):
        """
        Update recall using a new chunk.
        """
        y_true, y_pred = _validate_labels(y_true, y_pred)

        actual_positive = y_true == self.positive_label
        true_positive = actual_positive & (y_pred == self.positive_label)
        false_negative = actual_positive & (y_pred != self.positive_label)

        self.tp += int(np.sum(true_positive))
        self.fn += int(np.sum(false_negative))

        return self

    def result(self):
        """
        Return current cumulative recall.
        """
        return safe_divide(self.tp, self.tp + self.fn)

    def reset(self):
        """
        Reset stored metric state.
        """
        self.tp = 0
        self.fn = 0
        return self


class StreamingF1Score:
    """
    Streaming F1-score metric.

    F1 = 2 * precision * recall / (precision + recall)

    By default, positive_label=1.
    """

    def __init__(self, positive_label=1):
        self.positive_label = positive_label
        self.reset()

    def update(self, y_true, y_pred):
        """
        Update F1-score using a new chunk.
        """
        y_true, y_pred = _validate_labels(y_true, y_pred)

        actual_positive = y_true == self.positive_label
        predicted_positive = y_pred == self.positive_label

        self.tp += int(np.sum(actual_positive & predicted_positive))
        self.fp += int(np.sum((~actual_positive) & predicted_positive))
        self.fn += int(np.sum(actual_positive & (~predicted_positive)))

        return self

    def result(self):
        """
        Return current cumulative F1-score.
        """
        precision = safe_divide(self.tp, self.tp + self.fp)
        recall = safe_divide(self.tp, self.tp + self.fn)

        return safe_divide(2 * precision * recall, precision + recall)

    def reset(self):
        """
        Reset stored metric state.
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        return self


class StreamingConfusionMatrix:
    """
    Streaming confusion matrix.

    Supports binary and multiclass labels.
    """

    def __init__(self, labels=None):
        """
        Parameters
        ----------
        labels : array-like or None
            Fixed list of possible class labels.
            If None, labels are discovered from incoming chunks.
        """
        self.fixed_labels = labels is not None
        self.labels = None if labels is None else np.asarray(labels)
        self.reset()

    def update(self, y_true, y_pred):
        """
        Update confusion matrix using a new chunk.
        """
        y_true, y_pred = _validate_labels(y_true, y_pred)

        if self.labels is None:
            self.labels = np.unique(np.concatenate([y_true, y_pred]))
            self.matrix = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        elif not self.fixed_labels:
            new_labels = np.unique(np.concatenate([self.labels, y_true, y_pred]))

            if len(new_labels) != len(self.labels) or not np.all(new_labels == self.labels):
                old_labels = self.labels
                old_matrix = self.matrix

                self.labels = new_labels
                self.matrix = np.zeros((len(self.labels), len(self.labels)), dtype=int)

                for i, true_label in enumerate(old_labels):
                    for j, pred_label in enumerate(old_labels):
                        new_i = np.where(self.labels == true_label)[0][0]
                        new_j = np.where(self.labels == pred_label)[0][0]
                        self.matrix[new_i, new_j] = old_matrix[i, j]

        label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label not in label_to_index or pred_label not in label_to_index:
                raise ValueError("Found label not included in fixed labels.")

            i = label_to_index[true_label]
            j = label_to_index[pred_label]
            self.matrix[i, j] += 1

        return self

    def result(self):
        """
        Return current confusion matrix.
        """
        if self.matrix is None:
            if self.labels is None:
                return np.zeros((0, 0), dtype=int)
            return np.zeros((len(self.labels), len(self.labels)), dtype=int)

        return self.matrix.copy()

    def reset(self):
        """
        Reset stored confusion matrix.
        """
        if self.labels is None:
            self.matrix = None
        else:
            self.matrix = np.zeros((len(self.labels), len(self.labels)), dtype=int)

        return self


class RollingAccuracy:
    """
    Rolling-window accuracy.

    Only the most recent window_size predictions are used.
    """

    def __init__(self, window_size=100):
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero.")

        self.window_size = window_size
        self.reset()

    def update(self, y_true, y_pred):
        """
        Update rolling accuracy using a new chunk.
        """
        y_true, y_pred = _validate_labels(y_true, y_pred)

        for true_label, pred_label in zip(y_true, y_pred):
            self.window.append(int(true_label == pred_label))

        return self

    def result(self):
        """
        Return current rolling accuracy.
        """
        if len(self.window) == 0:
            return 0.0

        return float(np.mean(self.window))

    def reset(self):
        """
        Reset rolling window.
        """
        self.window = deque(maxlen=self.window_size)
        return self


def accuracy_score(y_true, y_pred):
    """
    Non-streaming accuracy helper.
    """
    y_true, y_pred = _validate_labels(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, positive_label=1):
    """
    Non-streaming binary precision helper.
    """
    metric = StreamingPrecision(positive_label=positive_label)
    metric.update(y_true, y_pred)
    return metric.result()


def recall_score(y_true, y_pred, positive_label=1):
    """
    Non-streaming binary recall helper.
    """
    metric = StreamingRecall(positive_label=positive_label)
    metric.update(y_true, y_pred)
    return metric.result()


def f1_score(y_true, y_pred, positive_label=1):
    """
    Non-streaming binary F1-score helper.
    """
    metric = StreamingF1Score(positive_label=positive_label)
    metric.update(y_true, y_pred)
    return metric.result()