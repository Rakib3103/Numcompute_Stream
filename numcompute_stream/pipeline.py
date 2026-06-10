"""
pipeline.py

Streaming pipeline implementation for NumCompute Stream.

The Pipeline class chains preprocessing transformers and a final model.

Example
-------
pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("scale", StandardScaler()),
    ("model", RandomForestClassifier())
])

pipe.partial_fit(X_chunk, y_chunk)
predictions = pipe.predict(X_test)
"""

import numpy as np


class Pipeline:
    """
    A simple stream-compatible machine learning pipeline.

    Parameters
    ----------
    steps : list of tuple
        List of (name, component) pairs.

        All intermediate components should implement:
        - partial_fit(X)
        - transform(X)

        The final component should implement:
        - partial_fit(X, y)
        - predict(X)
    """

    def __init__(self, steps):
        if not isinstance(steps, list):
            raise ValueError("steps must be a list of (name, component) tuples.")

        if len(steps) == 0:
            raise ValueError("Pipeline must contain at least one step.")

        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        """
        Validate pipeline step format.
        """
        seen_names = set()

        for step in self.steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError("Each pipeline step must be a (name, component) tuple.")

            name, component = step

            if not isinstance(name, str):
                raise ValueError("Pipeline step names must be strings.")

            if name in seen_names:
                raise ValueError(f"Duplicate pipeline step name found: {name}")

            if component is None:
                raise ValueError(f"Pipeline step '{name}' cannot be None.")

            seen_names.add(name)

    @property
    def named_steps(self):
        """
        Return pipeline steps as a dictionary.
        """
        return dict(self.steps)

    @property
    def model(self):
        """
        Return the final model in the pipeline.
        """
        return self.steps[-1][1]

    def partial_fit(self, X, y):
        """
        Incrementally fit the pipeline on a new chunk.

        For each transformer:
            partial_fit(X)
            transform(X)

        For final model:
            partial_fit(X_transformed, y)
        """
        X = self._validate_X(X)
        y = np.asarray(y).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        X_transformed = X

        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "partial_fit"):
                raise AttributeError(f"Step '{name}' does not implement partial_fit().")

            if not hasattr(transformer, "transform"):
                raise AttributeError(f"Step '{name}' does not implement transform().")

            transformer.partial_fit(X_transformed)
            X_transformed = transformer.transform(X_transformed)

        model_name, model = self.steps[-1]

        if not hasattr(model, "partial_fit"):
            raise AttributeError(f"Final step '{model_name}' does not implement partial_fit().")

        model.partial_fit(X_transformed, y)

        return self

    def fit(self, X, y):
        """
        Fit the pipeline on complete data.

        This method uses partial_fit internally for interface consistency.
        """
        return self.partial_fit(X, y)

    def transform(self, X):
        """
        Apply all transformer steps except the final model.

        Returns
        -------
        X_transformed : np.ndarray
            Transformed feature matrix.
        """
        X = self._validate_X(X)
        X_transformed = X

        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "transform"):
                raise AttributeError(f"Step '{name}' does not implement transform().")

            X_transformed = transformer.transform(X_transformed)

        return X_transformed

    def predict(self, X):
        """
        Transform X and predict using the final model.
        """
        X_transformed = self.transform(X)

        model_name, model = self.steps[-1]

        if not hasattr(model, "predict"):
            raise AttributeError(f"Final step '{model_name}' does not implement predict().")

        return model.predict(X_transformed)

    def score(self, X, y):
        """
        Return accuracy score for the pipeline.
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        return float(np.mean(y == y_pred))

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