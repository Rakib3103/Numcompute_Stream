import unittest
import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier


class TestPipeline(unittest.TestCase):

    def test_pipeline_partial_fit_predict_tree(self):
        X = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ])
        y = np.array([0, 0, 1, 1])

        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2))
        ])

        pipe.partial_fit(X, y)
        preds = pipe.predict(X)

        self.assertEqual(preds.shape, y.shape)

    def test_pipeline_partial_fit_predict_forest(self):
        X = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 3.0],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42))
        ])

        pipe.partial_fit(X, y)
        preds = pipe.predict(X)

        self.assertEqual(preds.shape, y.shape)

    def test_pipeline_streaming_two_chunks(self):
        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3))
        ])

        pipe.partial_fit(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([0, 0, 0])
        )

        pipe.partial_fit(
            np.array([[3.0], [4.0], [5.0]]),
            np.array([1, 1, 1])
        )

        preds = pipe.predict(np.array([[0.0], [5.0]]))
        expected = np.array([0, 1])

        np.testing.assert_array_equal(preds, expected)

    def test_pipeline_score(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])

        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2))
        ])

        pipe.partial_fit(X, y)
        score = pipe.score(X, y)

        self.assertGreaterEqual(score, 0.75)

    def test_pipeline_handles_nan(self):
        X = np.array([
            [0.0, np.nan],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, np.nan],
        ])
        y = np.array([0, 0, 1, 1])

        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2))
        ])

        pipe.partial_fit(X, y)
        preds = pipe.predict(X)

        self.assertEqual(preds.shape, y.shape)

    def test_named_steps(self):
        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("model", DecisionTreeClassifier())
        ])

        self.assertIn("imputer", pipe.named_steps)
        self.assertIn("model", pipe.named_steps)

    def test_empty_pipeline_raises_error(self):
        with self.assertRaises(ValueError):
            Pipeline([])

    def test_duplicate_step_name_raises_error(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ("step", SimpleImputer()),
                ("step", StandardScaler()),
                ("model", DecisionTreeClassifier())
            ])

    def test_invalid_step_format_raises_error(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ("imputer", SimpleImputer()),
                "bad_step"
            ])

    def test_shape_mismatch_raises_error(self):
        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("model", DecisionTreeClassifier())
        ])

        with self.assertRaises(ValueError):
            pipe.partial_fit(np.array([[1.0], [2.0]]), np.array([1]))


if __name__ == "__main__":
    unittest.main()