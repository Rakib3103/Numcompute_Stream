import unittest
import numpy as np

from numcompute_stream.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    EnsembleClassifier,
)


class TestEnsembleModels(unittest.TestCase):

    def test_bagging_fit_predict_shape(self):
        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = BaggingClassifier(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        self.assertEqual(preds.shape, y.shape)

    def test_bagging_score(self):
        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = BaggingClassifier(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X, y)

        score = model.score(X, y)
        self.assertGreaterEqual(score, 0.8)

    def test_bagging_partial_fit(self):
        model = BaggingClassifier(n_estimators=5, max_depth=2, random_state=42)

        model.partial_fit(np.array([[0], [1], [2]]), np.array([0, 0, 0]))
        model.partial_fit(np.array([[3], [4], [5]]), np.array([1, 1, 1]))

        preds = model.predict(np.array([[0], [5]]))
        expected = np.array([0, 1])

        np.testing.assert_array_equal(preds, expected)

    def test_random_forest_fit_predict_shape(self):
        X = np.array([
            [0, 0],
            [1, 0],
            [2, 1],
            [3, 1],
            [4, 1],
            [5, 1],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = RandomForestClassifier(
            n_estimators=5,
            max_depth=3,
            random_state=42
        )
        model.fit(X, y)

        preds = model.predict(X)
        self.assertEqual(preds.shape, y.shape)

    def test_random_forest_score(self):
        X = np.array([
            [0, 0],
            [1, 0],
            [2, 1],
            [3, 1],
            [4, 1],
            [5, 1],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = RandomForestClassifier(
            n_estimators=7,
            max_depth=3,
            random_state=42
        )
        model.fit(X, y)

        score = model.score(X, y)
        self.assertGreaterEqual(score, 0.8)

    def test_random_forest_partial_fit(self):
        model = RandomForestClassifier(
            n_estimators=5,
            max_depth=3,
            random_state=42
        )

        model.partial_fit(
            np.array([[0, 0], [1, 0], [2, 1]]),
            np.array([0, 0, 0])
        )

        model.partial_fit(
            np.array([[3, 1], [4, 1], [5, 1]]),
            np.array([1, 1, 1])
        )

        preds = model.predict(np.array([[0, 0], [5, 1]]))
        expected = np.array([0, 1])

        np.testing.assert_array_equal(preds, expected)

    def test_ensemble_classifier_random_forest(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        model = EnsembleClassifier(
            method="random_forest",
            n_estimators=5,
            max_depth=2,
            random_state=42
        )

        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(preds.shape, y.shape)

    def test_ensemble_classifier_bagging(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        model = EnsembleClassifier(
            method="bagging",
            n_estimators=5,
            max_depth=2,
            random_state=42
        )

        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(preds.shape, y.shape)

    def test_invalid_ensemble_method(self):
        with self.assertRaises(ValueError):
            EnsembleClassifier(method="wrong")

    def test_predict_before_fit_raises_error(self):
        model = RandomForestClassifier(n_estimators=3)

        with self.assertRaises(RuntimeError):
            model.predict(np.array([[1.0]]))

    def test_invalid_n_estimators(self):
        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=0)

    def test_shape_mismatch_raises_error(self):
        X = np.array([[0], [1]])
        y = np.array([0, 1])

        model = RandomForestClassifier(n_estimators=3)
        model.fit(X, y)

        with self.assertRaises(ValueError):
            model.predict(np.array([[1, 2]]))


if __name__ == "__main__":
    unittest.main()