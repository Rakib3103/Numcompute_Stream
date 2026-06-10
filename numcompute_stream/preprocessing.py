"""

Streaming preprocessing tools for NumCompute Stream.

This module contains:
- SimpleImputer
- StandardScaler
- MinMaxScaler
- OneHotEncoder

All main transformers support:
- partial_fit(X)
- transform(X)
- fit_transform(X)

"""

import numpy as np


def _as_2d_float_array(X, name="X"):
    """
    Convert input into a 2D float NumPy array.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")

    if X.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")

    return X


class SimpleImputer:
    """
    Streaming numeric imputer.

    Missing values are replaced with the running column mean.

    Parameters
    ----------
    strategy : str, default="mean"
        Currently only "mean" is supported.

    fill_value : float, default=0.0
        Fallback value used when a column has not seen any valid values.
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy != "mean":
            raise ValueError("Only strategy='mean' is supported.")

        self.strategy = strategy
        self.fill_value = fill_value

        self.count_ = None
        self.sum_ = None
        self.statistics_ = None

    def partial_fit(self, X):
        """
        Update imputation statistics from a new chunk.

        Parameters
        ----------
        X : np.ndarray
            Input chunk with shape (n_samples, n_features).

        Returns
        -------
        self
        """
        X = _as_2d_float_array(X)

        valid = ~np.isnan(X)
        chunk_count = np.sum(valid, axis=0).astype(float)
        chunk_sum = np.nansum(X, axis=0)

        if self.count_ is None:
            self.count_ = chunk_count
            self.sum_ = chunk_sum
        else:
            if X.shape[1] != self.count_.shape[0]:
                raise ValueError("X has a different number of features than previous chunks.")

            self.count_ += chunk_count
            self.sum_ += chunk_sum

        self.statistics_ = np.full_like(self.sum_, self.fill_value, dtype=float)

        np.divide(
            self.sum_,
            self.count_,
            out=self.statistics_,
            where=self.count_ != 0
        )

        return self

    def transform(self, X):
        """
        Replace NaN values with learned column means.
        """
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform().")

        X = _as_2d_float_array(X)

        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError("X has a different number of features than fitted data.")

        X_out = X.copy()
        nan_rows, nan_cols = np.where(np.isnan(X_out))
        X_out[nan_rows, nan_cols] = self.statistics_[nan_cols]

        return X_out

    def fit_transform(self, X):
        """
        Fit imputer and transform the same chunk.
        """
        return self.partial_fit(X).transform(X)

    def reset(self):
        """
        Reset learned statistics.
        """
        self.count_ = None
        self.sum_ = None
        self.statistics_ = None
        return self


class StandardScaler:
    """
    Streaming standard scaler.

    Transforms data using:

        z = (x - mean) / std

    Running mean and variance are updated chunk by chunk.
    NaN values are ignored during fitting and preserved during transform unless
    an imputer is used before this scaler.

    Parameters
    ----------
    eps : float, default=1e-12
        Small value used to avoid division by zero.
    """

    def __init__(self, eps=1e-12):
        self.eps = eps

        self.count_ = None
        self.sum_ = None
        self.sq_sum_ = None

        self.mean_ = None
        self.var_ = None
        self.scale_ = None

    def partial_fit(self, X):
        """
        Update scaler statistics from a new chunk.
        """
        X = _as_2d_float_array(X)

        valid = ~np.isnan(X)
        clean = np.where(valid, X, 0.0)

        chunk_count = np.sum(valid, axis=0).astype(float)
        chunk_sum = np.sum(clean, axis=0)
        chunk_sq_sum = np.sum(clean ** 2, axis=0)

        if self.count_ is None:
            self.count_ = chunk_count
            self.sum_ = chunk_sum
            self.sq_sum_ = chunk_sq_sum
        else:
            if X.shape[1] != self.count_.shape[0]:
                raise ValueError("X has a different number of features than previous chunks.")

            self.count_ += chunk_count
            self.sum_ += chunk_sum
            self.sq_sum_ += chunk_sq_sum

        self.mean_ = np.zeros_like(self.sum_, dtype=float)

        np.divide(
            self.sum_,
            self.count_,
            out=self.mean_,
            where=self.count_ != 0
        )

        mean_square = np.zeros_like(self.sq_sum_, dtype=float)

        np.divide(
            self.sq_sum_,
            self.count_,
            out=mean_square,
            where=self.count_ != 0
        )

        self.var_ = mean_square - self.mean_ ** 2
        self.var_ = np.maximum(self.var_, 0.0)

        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)

        return self

    def transform(self, X):
        """
        Standardise a chunk using current running statistics.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")

        X = _as_2d_float_array(X)

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("X has a different number of features than fitted data.")

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        Fit scaler and transform the same chunk.
        """
        return self.partial_fit(X).transform(X)

    def reset(self):
        """
        Reset scaler state.
        """
        self.count_ = None
        self.sum_ = None
        self.sq_sum_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        return self


class MinMaxScaler:
    """
    Streaming min-max scaler.

    Transforms data into a chosen range, usually [0, 1].

    Parameters
    ----------
    feature_range : tuple, default=(0, 1)
        Desired output range.
    """

    def __init__(self, feature_range=(0, 1)):
        min_range, max_range = feature_range

        if min_range >= max_range:
            raise ValueError("feature_range minimum must be smaller than maximum.")

        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def partial_fit(self, X):
        """
        Update min and max values from a new chunk.
        """
        X = _as_2d_float_array(X)

        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if self.data_min_ is None:
            self.data_min_ = chunk_min
            self.data_max_ = chunk_max
        else:
            if X.shape[1] != self.data_min_.shape[0]:
                raise ValueError("X has a different number of features than previous chunks.")

            self.data_min_ = np.minimum(self.data_min_, chunk_min)
            self.data_max_ = np.maximum(self.data_max_, chunk_max)

        return self

    def transform(self, X):
        """
        Scale a chunk using running min and max.
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform().")

        X = _as_2d_float_array(X)

        if X.shape[1] != self.data_min_.shape[0]:
            raise ValueError("X has a different number of features than fitted data.")

        out_min, out_max = self.feature_range

        denominator = self.data_max_ - self.data_min_
        denominator = np.where(denominator == 0, 1.0, denominator)

        X_scaled = (X - self.data_min_) / denominator
        X_scaled = X_scaled * (out_max - out_min) + out_min

        return X_scaled

    def fit_transform(self, X):
        """
        Fit scaler and transform the same chunk.
        """
        return self.partial_fit(X).transform(X)

    def reset(self):
        """
        Reset scaler state.
        """
        self.data_min_ = None
        self.data_max_ = None
        return self


class OneHotEncoder:
    """
    Simple streaming one-hot encoder for categorical integer/string data.

    Categories are expanded incrementally as new chunks arrive.

    Notes
    -----
    This encoder returns a dense NumPy matrix for simplicity.
    """

    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be either 'ignore' or 'error'.")

        self.handle_unknown = handle_unknown
        self.categories_ = None

    def partial_fit(self, X):
        """
        Update known categories from a new chunk.
        """
        X = np.asarray(X, dtype=object)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")

        if self.categories_ is None:
            self.categories_ = []

            for col in range(X.shape[1]):
                values = np.unique(X[:, col])
                self.categories_.append(values)
        else:
            if X.shape[1] != len(self.categories_):
                raise ValueError("X has a different number of features than previous chunks.")

            for col in range(X.shape[1]):
                values = np.unique(X[:, col])
                self.categories_[col] = np.unique(
                    np.concatenate([self.categories_[col], values])
                )

        return self

    def transform(self, X):
        """
        Convert categorical values to one-hot encoded matrix.
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform().")

        X = np.asarray(X, dtype=object)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[1] != len(self.categories_):
            raise ValueError("X has a different number of features than fitted data.")

        encoded_columns = []

        for col in range(X.shape[1]):
            categories = self.categories_[col]
            col_values = X[:, col]

            if self.handle_unknown == "error":
                unknown = np.setdiff1d(col_values, categories)
                if unknown.size > 0:
                    raise ValueError(f"Unknown categories found: {unknown}")

            encoded = np.zeros((X.shape[0], len(categories)), dtype=float)

            for idx, category in enumerate(categories):
                encoded[:, idx] = (col_values == category).astype(float)

            encoded_columns.append(encoded)

        return np.concatenate(encoded_columns, axis=1)

    def fit_transform(self, X):
        """
        Fit encoder and transform the same chunk.
        """
        return self.partial_fit(X).transform(X)

    def reset(self):
        """
        Reset categories.
        """
        self.categories_ = None
        return self