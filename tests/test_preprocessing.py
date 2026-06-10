import unittest
import numpy as np

from numcompute_stream.preprocessing import (
    SimpleImputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)


class TestPreprocessing(unittest.TestCase):

    def test_simple_imputer_replaces_nan(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        imputer = SimpleImputer()
        X_out = imputer.fit_transform(X)

        expected = np.array([[1.0, 4.0], [3.0, 4.0]])
        np.testing.assert_allclose(X_out, expected)

    def test_simple_imputer_streaming_mean(self):
        imputer = SimpleImputer()
        imputer.partial_fit(np.array([[1.0], [3.0]]))
        imputer.partial_fit(np.array([[5.0], [np.nan]]))

        self.assertAlmostEqual(imputer.statistics_[0], 3.0)

    def test_simple_imputer_all_nan_column(self):
        X = np.array([[np.nan], [np.nan]])
        imputer = SimpleImputer(fill_value=-1.0)
        X_out = imputer.fit_transform(X)

        expected = np.array([[-1.0], [-1.0]])
        np.testing.assert_allclose(X_out, expected)

    def test_standard_scaler_mean(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler = StandardScaler()
        scaler.partial_fit(X)

        expected = np.array([2.0, 3.0])
        np.testing.assert_allclose(scaler.mean_, expected)

    def test_standard_scaler_variance(self):
        X = np.array([[1.0], [3.0], [5.0]])
        scaler = StandardScaler()
        scaler.partial_fit(X)

        expected = np.array([8 / 3])
        np.testing.assert_allclose(scaler.var_, expected)

    def test_standard_scaler_transform(self):
        X = np.array([[1.0], [3.0], [5.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.assertAlmostEqual(float(np.mean(X_scaled)), 0.0, places=7)
        self.assertAlmostEqual(float(np.var(X_scaled)), 1.0, places=7)

    def test_standard_scaler_zero_variance(self):
        X = np.array([[2.0], [2.0], [2.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        expected = np.array([[0.0], [0.0], [0.0]])
        np.testing.assert_allclose(X_scaled, expected)

    def test_standard_scaler_streaming(self):
        scaler = StandardScaler()
        scaler.partial_fit(np.array([[1.0], [3.0]]))
        scaler.partial_fit(np.array([[5.0]]))

        self.assertAlmostEqual(scaler.mean_[0], 3.0)
        self.assertAlmostEqual(scaler.var_[0], 8 / 3)

    def test_minmax_scaler(self):
        X = np.array([[1.0], [3.0], [5.0]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        expected = np.array([[0.0], [0.5], [1.0]])
        np.testing.assert_allclose(X_scaled, expected)

    def test_minmax_scaler_streaming(self):
        scaler = MinMaxScaler()
        scaler.partial_fit(np.array([[1.0], [3.0]]))
        scaler.partial_fit(np.array([[5.0]]))

        X_scaled = scaler.transform(np.array([[3.0]]))
        self.assertAlmostEqual(X_scaled[0, 0], 0.5)

    def test_onehot_encoder_basic(self):
        X = np.array([["red"], ["blue"], ["red"]])
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)

        self.assertEqual(X_encoded.shape, (3, 2))

    def test_onehot_encoder_streaming_expands_categories(self):
        encoder = OneHotEncoder()
        encoder.partial_fit(np.array([["red"], ["blue"]]))
        encoder.partial_fit(np.array([["green"]]))

        X_encoded = encoder.transform(np.array([["green"]]))
        self.assertEqual(X_encoded.shape, (1, 3))

    def test_shape_mismatch_raises_error(self):
        scaler = StandardScaler()
        scaler.partial_fit(np.array([[1.0, 2.0]]))

        with self.assertRaises(ValueError):
            scaler.partial_fit(np.array([[1.0, 2.0, 3.0]]))

    def test_transform_before_fit_raises_error(self):
        scaler = StandardScaler()

        with self.assertRaises(RuntimeError):
            scaler.transform(np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()