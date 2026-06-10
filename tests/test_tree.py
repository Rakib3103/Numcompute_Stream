import unittest
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):

    def test_tree_fit_predict_simple(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_tree_predict_shape(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=2)
        model.fit(X, y)

        preds = model.predict(np.array([[1], [2]]))

        self.assertEqual(preds.shape, (2,))

    def test_tree_score(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=2)
        model.fit(X, y)

        self.assertAlmostEqual(model.score(X, y), 1.0)

    def test_tree_partial_fit(self):
        model = DecisionTreeClassifier(max_depth=2)

        model.partial_fit(np.array([[0], [1]]), np.array([0, 0]))
        model.partial_fit(np.array([[2], [3]]), np.array([1, 1]))

        preds = model.predict(np.array([[0], [3]]))
        expected = np.array([0, 1])

        np.testing.assert_array_equal(preds, expected)

    def test_tree_handles_nan(self):
        X = np.array([[0.0], [1.0], [np.nan], [3.0]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=2)
        model.fit(X, y)

        preds = model.predict(np.array([[np.nan], [3.0]]))

        self.assertEqual(preds.shape, (2,))

    def test_tree_entropy_criterion(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=2, criterion="entropy")
        model.fit(X, y)

        preds = model.predict(X)

        np.testing.assert_array_equal(preds, y)

    def test_tree_invalid_criterion(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="wrong")

    def test_tree_max_depth(self):
        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 0, 0])

        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y)

        self.assertLessEqual(model.get_depth(), 1)

    def test_tree_tie_break_smallest_label(self):
        X = np.array([[0], [1]])
        y = np.array([1, 0])

        model = DecisionTreeClassifier(max_depth=1, min_samples_split=3)
        model.fit(X, y)

        preds = model.predict(np.array([[0.5]]))

        self.assertEqual(preds[0], 0)

    def test_predict_before_fit_raises_error(self):
        model = DecisionTreeClassifier()

        with self.assertRaises(RuntimeError):
            model.predict(np.array([[1.0]]))

    def test_shape_mismatch_raises_error(self):
        X = np.array([[0], [1]])
        y = np.array([0, 1])

        model = DecisionTreeClassifier()
        model.fit(X, y)

        with self.assertRaises(ValueError):
            model.predict(np.array([[1, 2]]))

    def test_max_features_sqrt(self):
        X = np.array([
            [0, 1, 2, 3],
            [1, 1, 2, 3],
            [2, 1, 2, 3],
            [3, 1, 2, 3],
        ])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=2, max_features="sqrt", random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        self.assertEqual(preds.shape, y.shape)


if __name__ == "__main__":
    unittest.main()