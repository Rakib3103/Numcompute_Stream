"""
Streaming statistical utilities for NumCompute Stream.
"""

import numpy as np


def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide numerator by denominator.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.full_like(numerator, fill_value=default, dtype=float)

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0
    )

    if result.shape == ():
        return float(result)

    return result


def nanmean(X, axis=0):
    """
    Compute NaN-safe mean.
    """
    X = np.asarray(X, dtype=float)

    valid = ~np.isnan(X)
    count = np.sum(valid, axis=axis)
    total = np.nansum(X, axis=axis)

    return safe_divide(total, count)


def nanvar(X, axis=0):
    """
    Compute NaN-safe variance.
    """
    X = np.asarray(X, dtype=float)
    mean = nanmean(X, axis=axis)

    if axis == 0:
        centred = X - mean
    elif axis == 1:
        centred = X - mean[:, None]
    else:
        raise ValueError("Only axis=0 and axis=1 are supported.")

    valid = ~np.isnan(X)
    squared = np.where(valid, centred ** 2, 0.0)

    count = np.sum(valid, axis=axis)
    total = np.sum(squared, axis=axis)

    return safe_divide(total, count)


class RunningMean:
    """
    Streaming running mean.
    """

    def __init__(self):
        self.count_ = None
        self.mean_ = None

    def update(self, X_chunk):
        """
        Update running mean from a new chunk.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)

        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk must not be empty.")

        valid = ~np.isnan(X_chunk)
        chunk_count = np.sum(valid, axis=0)
        chunk_sum = np.nansum(X_chunk, axis=0)

        if self.mean_ is None:
            self.count_ = chunk_count.astype(float)
            self.mean_ = safe_divide(chunk_sum, self.count_)
            return self

        if X_chunk.shape[1] != self.mean_.shape[0]:
            raise ValueError("X_chunk has a different number of features.")

        total_count = self.count_ + chunk_count
        total_sum = self.mean_ * self.count_ + chunk_sum

        self.mean_ = safe_divide(total_sum, total_count)
        self.count_ = total_count

        return self

    def result(self):
        """
        Return current running mean.
        """
        if self.mean_ is None:
            return None

        return self.mean_.copy()

    def reset(self):
        """
        Reset state.
        """
        self.count_ = None
        self.mean_ = None
        return self


class RunningVariance:
    """
    Streaming feature-wise variance.

    This implementation stores cumulative count, sum, and squared sum.
    NaN values are ignored.
    """

    def __init__(self):
        self.count_ = None
        self.sum_ = None
        self.sq_sum_ = None

    def update(self, X_chunk):
        """
        Update running variance from a new chunk.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)

        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk must not be empty.")

        valid = ~np.isnan(X_chunk)
        clean = np.where(valid, X_chunk, 0.0)

        chunk_count = np.sum(valid, axis=0).astype(float)
        chunk_sum = np.sum(clean, axis=0)
        chunk_sq_sum = np.sum(clean ** 2, axis=0)

        if self.count_ is None:
            self.count_ = chunk_count
            self.sum_ = chunk_sum
            self.sq_sum_ = chunk_sq_sum
            return self

        if X_chunk.shape[1] != self.sum_.shape[0]:
            raise ValueError("X_chunk has a different number of features.")

        self.count_ += chunk_count
        self.sum_ += chunk_sum
        self.sq_sum_ += chunk_sq_sum

        return self

    def result(self):
        """
        Return current variance.
        """
        if self.count_ is None:
            return None

        mean = safe_divide(self.sum_, self.count_)
        mean_square = safe_divide(self.sq_sum_, self.count_)
        variance = mean_square - mean ** 2

        # Avoid tiny negative values caused by floating point precision.
        variance = np.maximum(variance, 0.0)

        return variance

    def reset(self):
        """
        Reset state.
        """
        self.count_ = None
        self.sum_ = None
        self.sq_sum_ = None
        return self


class RunningHistogram:
    """
    Streaming histogram.
    """

    def __init__(self, bins=10, value_range=None):
        if bins <= 0:
            raise ValueError("bins must be greater than zero.")

        self.bins = bins
        self.value_range = value_range
        self.counts_ = None
        self.edges_ = None

    def update(self, X_chunk):
        """
        Update histogram using a new chunk.
        """
        X_chunk = np.asarray(X_chunk, dtype=float).ravel()
        X_chunk = X_chunk[~np.isnan(X_chunk)]

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk contains no valid numeric values.")

        counts, edges = np.histogram(
            X_chunk,
            bins=self.bins,
            range=self.value_range
        )

        if self.counts_ is None:
            self.counts_ = counts.astype(int)
            self.edges_ = edges
        else:
            if not np.allclose(edges, self.edges_):
                raise ValueError(
                    "Histogram bin edges changed. Use a fixed value_range for streaming histograms."
                )
            self.counts_ += counts

        return self

    def result(self):
        """
        Return histogram counts and bin edges.
        """
        if self.counts_ is None:
            return None, None

        return self.counts_.copy(), self.edges_.copy()

    def reset(self):
        """
        Reset histogram state.
        """
        self.counts_ = None
        self.edges_ = None
        return self


class RunningQuantile:
    """
    Simple streaming quantile estimator.
    """

    def __init__(self, q=0.5, window_size=1000):
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1.")

        if window_size <= 0:
            raise ValueError("window_size must be greater than zero.")

        self.q = q
        self.window_size = window_size
        self.values_ = np.array([], dtype=float)

    def update(self, X_chunk):
        """
        Update quantile window using a new chunk.
        """
        X_chunk = np.asarray(X_chunk, dtype=float).ravel()
        X_chunk = X_chunk[~np.isnan(X_chunk)]

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk contains no valid numeric values.")

        self.values_ = np.concatenate([self.values_, X_chunk])

        if self.values_.shape[0] > self.window_size:
            self.values_ = self.values_[-self.window_size:]

        return self

    def result(self):
        """
        Return current quantile estimate.
        """
        if self.values_.shape[0] == 0:
            return None

        return float(np.quantile(self.values_, self.q))

    def reset(self):
        """
        Reset stored values.
        """
        self.values_ = np.array([], dtype=float)
        return self


class StreamingStats:
    """
    Combined streaming statistics manager.
    """

    def __init__(self, histogram_bins=10, histogram_range=None):
        self.mean = RunningMean()
        self.variance = RunningVariance()
        self.histogram = RunningHistogram(
            bins=histogram_bins,
            value_range=histogram_range
        )

    def update_stats(self, X_chunk):
        """
        Update all statistics from one chunk.
        """
        self.mean.update(X_chunk)
        self.variance.update(X_chunk)
        self.histogram.update(X_chunk)

        return self

    def result(self):
        """
        Return all current statistics in a dictionary.
        """
        hist_counts, hist_edges = self.histogram.result()

        return {
            "mean": self.mean.result(),
            "variance": self.variance.result(),
            "histogram_counts": hist_counts,
            "histogram_edges": hist_edges,
        }

    def reset(self):
        """
        Reset all statistics.
        """
        self.mean.reset()
        self.variance.reset()
        self.histogram.reset()
        return self