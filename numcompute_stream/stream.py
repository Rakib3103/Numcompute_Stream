"""
stream.py

StreamTrainer for chunk-wise learning.

This module manages:
- fitting models/pipelines chunk by chunk
- scoring each chunk
- updating streaming metrics
- logging accuracy, error, memory use, and cumulative accuracy

The assignment specification requires stream.py to provide StreamTrainer
with fit_chunk(X, y), score_chunk(X, y), and logging support.
"""

import time
import numpy as np

from .metrics import StreamingAccuracy


class StreamTrainer:
    """
    Manage streaming training, prediction, evaluation, and logs.

    Parameters
    ----------
    model : object
        Any model or pipeline implementing:
        - partial_fit(X, y)
        - predict(X)

    metrics : dict or None
        Optional dictionary of metric objects.
        Each metric should implement:
        - update(y_true, y_pred)
        - result()
        - reset()

    evaluate_before_fit : bool, default=False
        If True, the trainer tries to score each chunk before training on it.
        If False, each chunk is fitted first and then scored.

    verbose : bool, default=False
        Whether to print per-chunk logs.
    """

    def __init__(self, model, metrics=None, evaluate_before_fit=False, verbose=False):
        if not hasattr(model, "partial_fit"):
            raise AttributeError("model must implement partial_fit(X, y).")

        if not hasattr(model, "predict"):
            raise AttributeError("model must implement predict(X).")

        self.model = model
        self.metrics = metrics if metrics is not None else {
            "accuracy": StreamingAccuracy()
        }

        self.evaluate_before_fit = evaluate_before_fit
        self.verbose = verbose

        self.chunk_index = 0
        self.total_correct = 0
        self.total_seen = 0

        self.logs = {
            "chunk": [],
            "accuracy": [],
            "error": [],
            "cumulative_accuracy": [],
            "memory_bytes": [],
            "fit_time": [],
            "score_time": [],
        }

    def fit_chunk(self, X_chunk, y_chunk):
        """
        Fit the model on a single incoming chunk and log metrics.

        Parameters
        ----------
        X_chunk : np.ndarray
            Feature chunk.

        y_chunk : np.ndarray
            Target chunk.

        Returns
        -------
        self
        """
        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        start_fit = time.perf_counter()
        self.model.partial_fit(X_chunk, y_chunk)
        fit_time = time.perf_counter() - start_fit

        start_score = time.perf_counter()
        y_pred = self.model.predict(X_chunk)
        score_time = time.perf_counter() - start_score

        self._update_logs(X_chunk, y_chunk, y_pred, fit_time, score_time)

        return self

    def score_chunk(self, X_chunk, y_chunk):
        """
        Score the current model on a chunk without fitting.

        Parameters
        ----------
        X_chunk : np.ndarray
            Feature chunk.

        y_chunk : np.ndarray
            Target chunk.

        Returns
        -------
        scores : dict
            Dictionary of metric results for this chunk.
        """
        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        start_score = time.perf_counter()
        y_pred = self.model.predict(X_chunk)
        score_time = time.perf_counter() - start_score

        chunk_accuracy = float(np.mean(y_pred == y_chunk))
        chunk_error = 1.0 - chunk_accuracy

        scores = {
            "accuracy": chunk_accuracy,
            "error": chunk_error,
            "score_time": score_time,
        }

        return scores

    def fit_stream(self, chunks):
        """
        Fit model over an iterable of chunks.

        Parameters
        ----------
        chunks : iterable
            Iterable containing (X_chunk, y_chunk) pairs.

        Returns
        -------
        self
        """
        for X_chunk, y_chunk in chunks:
            self.fit_chunk(X_chunk, y_chunk)

        return self

    def predict(self, X):
        """
        Predict using the managed model.
        """
        return self.model.predict(X)

    def get_logs(self):
        """
        Return training logs.

        Returns
        -------
        logs : dict
            Dictionary containing per-chunk values.
        """
        return {
            key: list(value)
            for key, value in self.logs.items()
        }

    def reset_logs(self):
        """
        Reset logs and metric states.
        """
        self.chunk_index = 0
        self.total_correct = 0
        self.total_seen = 0

        self.logs = {
            "chunk": [],
            "accuracy": [],
            "error": [],
            "cumulative_accuracy": [],
            "memory_bytes": [],
            "fit_time": [],
            "score_time": [],
        }

        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

        return self

    def _update_logs(self, X_chunk, y_chunk, y_pred, fit_time, score_time):
        """
        Update internal logs after processing a chunk.
        """
        chunk_accuracy = float(np.mean(y_pred == y_chunk))
        chunk_error = 1.0 - chunk_accuracy

        correct = int(np.sum(y_pred == y_chunk))
        total = int(y_chunk.shape[0])

        self.total_correct += correct
        self.total_seen += total

        cumulative_accuracy = (
            self.total_correct / self.total_seen
            if self.total_seen > 0
            else 0.0
        )

        memory_bytes = int(X_chunk.nbytes + y_chunk.nbytes + y_pred.nbytes)

        for metric in self.metrics.values():
            metric.update(y_chunk, y_pred)

        self.logs["chunk"].append(self.chunk_index)
        self.logs["accuracy"].append(chunk_accuracy)
        self.logs["error"].append(chunk_error)
        self.logs["cumulative_accuracy"].append(cumulative_accuracy)
        self.logs["memory_bytes"].append(memory_bytes)
        self.logs["fit_time"].append(float(fit_time))
        self.logs["score_time"].append(float(score_time))

        if self.verbose:
            print(
                f"Chunk {self.chunk_index}: "
                f"accuracy={chunk_accuracy:.4f}, "
                f"cumulative_accuracy={cumulative_accuracy:.4f}, "
                f"memory={memory_bytes} bytes"
            )

        self.chunk_index += 1

    def _validate_X_y(self, X, y):
        """
        Validate feature and target chunks.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X_chunk must not be empty.")

        y = np.asarray(y).ravel()

        if y.shape[0] == 0:
            raise ValueError("y_chunk must not be empty.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must contain the same number of samples.")

        return X, y