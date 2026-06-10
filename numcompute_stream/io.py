"""
io.py

Custom input/output utilities for the NumCompute streaming framework.

This module avoids external data-processing libraries such as pandas.
It provides simple CSV loading, train-test splitting, and chunk creation
for streaming machine learning experiments.
"""

import csv
import numpy as np


def _convert_cell(value):
    """
    Convert a CSV cell to float.

    Empty cells are converted to np.nan so that SimpleImputer can handle them.
    """
    value = value.strip()

    if value == "":
        return np.nan

    return float(value)


def load_csv(path, target_column=-1, skip_header=True):
    """
    Load a CSV file into NumPy arrays.

    Empty cells are treated as np.nan.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    target_column : int, default=-1
        Index of the target column.

    skip_header : bool, default=True
        Whether to skip the first row.

    Returns
    -------
    X : np.ndarray
        Feature matrix with shape (n_samples, n_features).

    y : np.ndarray
        Target vector with shape (n_samples,).
    """
    rows = []

    with open(path, "r", newline="") as file:
        reader = csv.reader(file)

        if skip_header:
            next(reader, None)

        for row in reader:
            if len(row) == 0:
                continue

            converted_row = [_convert_cell(cell) for cell in row]
            rows.append(converted_row)

    if len(rows) == 0:
        raise ValueError("CSV file is empty or contains no valid rows.")

    data = np.array(rows, dtype=float)

    if target_column < 0:
        target_column = data.shape[1] + target_column

    if target_column < 0 or target_column >= data.shape[1]:
        raise ValueError("target_column is out of range.")

    y = data[:, target_column]
    X = np.delete(data, target_column, axis=1)

    return X, y


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split arrays into training and testing sets.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_count = int(n_samples * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def make_chunks(X, y, chunk_size=32):
    """
    Split data into chunks to simulate a streaming setting.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    chunks = []

    for start in range(0, X.shape[0], chunk_size):
        end = start + chunk_size
        chunks.append((X[start:end], y[start:end]))

    return chunks