"""
ensemble.py

Tree-based ensemble models for NumCompute Stream.

This module implements:
- BaggingClassifier
- RandomForestClassifier
- EnsembleClassifier alias/wrapper

Only NumPy is used.

Streaming design note:
The ensemble supports partial_fit() by storing incoming chunks and rebuilding
the ensemble from all data seen so far. Each tree is trained on a bootstrap
sample, and RandomForestClassifier also uses random feature subsets.
"""

import numpy as np

from .tree import DecisionTreeClassifier


def _validate_X_y(X, y):
    """
    Validate feature matrix and target vector.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")

    if X.shape[0] == 0:
        raise ValueError("X must not be empty.")

    y = np.asarray(y).ravel()

    if y.shape[0] == 0:
        raise ValueError("y must not be empty.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    return X, y


def _validate_X(X):
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


class BaggingClassifier:
    """
    Bagging ensemble using multiple decision trees.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.

    max_depth : int, default=5
        Maximum depth for each tree.

    min_samples_split : int, default=2
        Minimum samples required to split a tree node.

    criterion : str, default="gini"
        Tree split criterion. Supported: "gini", "entropy".

    bootstrap : bool, default=True
        Whether to train each tree on a bootstrap sample.

    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        bootstrap=True,
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_ = []
        self.classes_ = None
        self.n_features_ = None

        self._X_seen = None
        self._y_seen = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit the bagging ensemble.
        """
        X, y = _validate_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.estimators_ = []

        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = self._rng.integers(0, n_samples, size=n_samples)
            else:
                indices = np.arange(n_samples)

            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=None,
                random_state=None if self.random_state is None else self.random_state + i,
            )

            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Update ensemble using a new streaming chunk.
        """
        X_chunk, y_chunk = _validate_X_y(X_chunk, y_chunk)

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
        Predict labels using majority voting.
        """
        if len(self.estimators_) == 0:
            raise RuntimeError("BaggingClassifier must be fitted before predict().")

        X = _validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different number of features than fitted data.")

        all_predictions = np.array([tree.predict(X) for tree in self.estimators_])

        return self._majority_vote(all_predictions)

    def score(self, X, y):
        """
        Return accuracy on given data.
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        return float(np.mean(y == y_pred))

    def _majority_vote(self, all_predictions):
        """
        Majority vote across trees.

        Ties are resolved by choosing the smallest class label.
        """
        n_samples = all_predictions.shape[1]
        final_predictions = []

        for sample_index in range(n_samples):
            votes = all_predictions[:, sample_index]
            values, counts = np.unique(votes, return_counts=True)
            max_count = np.max(counts)
            tied_values = values[counts == max_count]
            final_predictions.append(np.min(tied_values))

        return np.array(final_predictions)


class RandomForestClassifier:
    """
    Random Forest classifier using decision trees.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.

    max_depth : int, default=5
        Maximum depth of each tree.

    min_samples_split : int, default=2
        Minimum samples needed to split a tree node.

    criterion : str, default="gini"
        Tree split criterion. Supported: "gini", "entropy".

    max_features : int, str, or None, default="sqrt"
        Number of features considered at each split.

    bootstrap : bool, default=True
        Whether to train trees using bootstrap samples.

    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_ = []
        self.classes_ = None
        self.n_features_ = None

        self._X_seen = None
        self._y_seen = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit the random forest.
        """
        X, y = _validate_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.estimators_ = []

        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = self._rng.integers(0, n_samples, size=n_samples)
            else:
                indices = np.arange(n_samples)

            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + i,
            )

            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Incrementally update the random forest using a new data chunk.

        Internally, the model stores all seen data and rebuilds the forest.
        """
        X_chunk, y_chunk = _validate_X_y(X_chunk, y_chunk)

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
        Predict labels using majority voting from all trees.
        """
        if len(self.estimators_) == 0:
            raise RuntimeError("RandomForestClassifier must be fitted before predict().")

        X = _validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different number of features than fitted data.")

        all_predictions = np.array([tree.predict(X) for tree in self.estimators_])

        return self._majority_vote(all_predictions)

    def score(self, X, y):
        """
        Return accuracy on given data.
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        return float(np.mean(y == y_pred))

    def _majority_vote(self, all_predictions):
        """
        Majority vote across trees.

        Ties are resolved by choosing the smallest class label.
        """
        n_samples = all_predictions.shape[1]
        final_predictions = []

        for sample_index in range(n_samples):
            votes = all_predictions[:, sample_index]
            values, counts = np.unique(votes, return_counts=True)
            max_count = np.max(counts)
            tied_values = values[counts == max_count]
            final_predictions.append(np.min(tied_values))

        return np.array(final_predictions)


class EnsembleClassifier:
    """
    General ensemble wrapper.

    This class allows switching between ensemble methods using a shared API.

    Parameters
    ----------
    method : str, default="random_forest"
        Supported values:
        - "random_forest"
        - "bagging"

    **kwargs
        Parameters passed to the selected ensemble model.
    """

    def __init__(self, method="random_forest", **kwargs):
        if method == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        elif method == "bagging":
            self.model = BaggingClassifier(**kwargs)
        else:
            raise ValueError("method must be either 'random_forest' or 'bagging'.")

        self.method = method

    def fit(self, X, y):
        """
        Fit selected ensemble model.
        """
        self.model.fit(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Stream update selected ensemble model.
        """
        self.model.partial_fit(X_chunk, y_chunk)
        return self

    def predict(self, X):
        """
        Predict using selected ensemble model.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Return accuracy.
        """
        return self.model.score(X, y)