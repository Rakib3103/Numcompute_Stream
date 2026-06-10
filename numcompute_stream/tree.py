"""
tree.py

Decision tree classifier built from scratch using NumPy only.

This classifier supports:
- Gini impurity
- Entropy impurity
- max_depth
- min_samples_split
- max_features
- partial_fit() for streaming-style updates

Note:
Here, partial_fit() stores incoming chunks and rebuilds the tree using all data seen so far.
This provides a clean stream-compatible API while keeping the algorithm
stable and understandable.
"""

import numpy as np


class _TreeNode:
    """
    Internal decision tree node.
    """

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        prediction=None,
        depth=0,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.depth = depth

    def is_leaf(self):
        return self.prediction is not None


class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.

    min_samples_split : int, default=2
        Minimum number of samples required to split a node.

    criterion : str, default="gini"
        Split criterion. Supported values are "gini" and "entropy".

    max_features : int, str, or None, default=None
        Number of features considered at each split.
        - None: use all features
        - int: use that many features
        - "sqrt": use sqrt(n_features)
        - "log2": use log2(n_features)

    random_state : int or None
        Random seed for feature sampling.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero.")

        if min_samples_split <= 1:
            raise ValueError("min_samples_split must be greater than one.")

        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be either 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root_ = None
        self.classes_ = None
        self.n_features_ = None

        self._X_seen = None
        self._y_seen = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit a decision tree from complete data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features).

        y : np.ndarray
            Target vector with shape (n_samples,).

        Returns
        -------
        self
        """
        X, y = self._validate_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        X = self._replace_nan_with_column_mean(X)

        self.root_ = self._build_tree(X, y, depth=0)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Incrementally update the tree from a new chunk.

        This implementation stores all previous chunks and rebuilds the tree
        whenever new data arrives.
        """
        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        if self._X_seen is None:
            self._X_seen = X_chunk.copy()
            self._y_seen = y_chunk.copy()
        else:
            if X_chunk.shape[1] != self._X_seen.shape[1]:
                raise ValueError("X_chunk has a different number of features.")

            self._X_seen = np.vstack([self._X_seen, X_chunk])
            self._y_seen = np.concatenate([self._y_seen, y_chunk])

        return self.fit(self._X_seen, self._y_seen)

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels with shape (n_samples,).
        """
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict().")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different number of features than fitted data.")

        X = self._replace_nan_with_column_mean(X)

        predictions = np.array([self._predict_one(row, self.root_) for row in X])

        return predictions

    def score(self, X, y):
        """
        Return accuracy on given data.
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        return float(np.mean(y == y_pred))

    def _validate_X_y(self, X, y):
        """
        Validate feature and target arrays.
        """
        X = self._validate_X(X)
        y = np.asarray(y).ravel()

        if y.shape[0] == 0:
            raise ValueError("y must not be empty.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        return X, y

    def _validate_X(self, X):
        """
        Validate feature matrix.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")

        return X

    def _replace_nan_with_column_mean(self, X):
        """
        Replace NaN values using column means.

        If a whole column is NaN, replace with 0.
        """
        X = X.copy()

        if not np.isnan(X).any():
            return X

        means = np.nanmean(X, axis=0)
        means = np.where(np.isnan(means), 0.0, means)

        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = means[cols]

        return X

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        prediction = self._majority_class(y)

        if (
            depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or np.unique(y).shape[0] == 1
        ):
            return _TreeNode(prediction=prediction, depth=depth)

        split = self._best_split(X, y)

        if split is None:
            return _TreeNode(prediction=prediction, depth=depth)

        feature_index, threshold, left_mask, right_mask = split

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return _TreeNode(prediction=prediction, depth=depth)

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child,
            depth=depth,
        )

    def _best_split(self, X, y):
        """
        Find the best feature and threshold split.
        """
        n_samples, n_features = X.shape
        feature_indices = self._select_feature_indices(n_features)

        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_split = None

        for feature_index in feature_indices:
            values = X[:, feature_index]
            unique_values = np.unique(values)

            if unique_values.shape[0] <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = values > threshold

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left == 0 or n_right == 0:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                weighted_impurity = (
                    (n_left / n_samples) * left_impurity
                    + (n_right / n_samples) * right_impurity
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (
                        feature_index,
                        threshold,
                        left_mask,
                        right_mask,
                    )

        return best_split

    def _select_feature_indices(self, n_features):
        """
        Select feature indices based on max_features.
        """
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, int):
            k = self.max_features
        elif self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            k = int(np.log2(n_features))
        else:
            raise ValueError("max_features must be None, int, 'sqrt', or 'log2'.")

        k = max(1, min(k, n_features))

        return self._rng.choice(n_features, size=k, replace=False)

    def _impurity(self, y):
        """
        Calculate impurity using selected criterion.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities ** 2)

        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def _majority_class(self, y):
        """
        Return majority class.

        Ties are resolved by choosing the smallest class label.
        """
        values, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)

        tied_values = values[counts == max_count]

        return np.min(tied_values)

    def _predict_one(self, row, node):
        """
        Predict one sample by traversing the tree.
        """
        if node.is_leaf():
            return node.prediction

        if row[node.feature_index] <= node.threshold:
            return self._predict_one(row, node.left)

        return self._predict_one(row, node.right)

    def get_depth(self):
        """
        Return maximum depth of the fitted tree.
        """
        if self.root_ is None:
            return 0

        return self._node_depth(self.root_)

    def _node_depth(self, node):
        if node is None or node.is_leaf():
            return node.depth

        return max(self._node_depth(node.left), self._node_depth(node.right))