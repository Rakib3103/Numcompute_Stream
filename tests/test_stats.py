import unittest
import numpy as np

from numcompute_stream.stats import (
    safe_divide,
    nanmean,
    nanvar,
    RunningMean,
    RunningVariance,
    RunningHistogram,
    RunningQuantile,
    StreamingStats,
)


class TestStreamingStats(unittest.TestCase):

    def test_safe_divide_normal(self):
        self.assertEqual(safe_divide(10, 2), 5.0)

    def test_safe_divide_zero(self):
        self.assertEqual(safe_divide(10, 0), 0.0)

    def test_nanmean_ignores_nan(self):
        X = np.array([[1.0, np.nan], [3.0, 5.0]])
        result = nanmean(X, axis=0)
        expected = np.array([2.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_nanvar_ignores_nan(self):
        X = np.array([[1.0, np.nan], [3.0, 5.0]])
        result = nanvar(X, axis=0)
        expected = np.array([1.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_running_mean_single_chunk(self):
        rm = RunningMean()
        rm.update(np.array([[1, 2], [3, 4]]))
        expected = np.array([2.0, 3.0])
        np.testing.assert_allclose(rm.result(), expected)

    def test_running_mean_multiple_chunks(self):
        rm = RunningMean()
        rm.update(np.array([[1, 2], [3, 4]]))
        rm.update(np.array([[5, 6]]))
        expected = np.array([3.0, 4.0])
        np.testing.assert_allclose(rm.result(), expected)

    def test_running_mean_nan(self):
        rm = RunningMean()
        rm.update(np.array([[1.0, np.nan], [3.0, 5.0]]))
        expected = np.array([2.0, 5.0])
        np.testing.assert_allclose(rm.result(), expected)

    def test_running_variance_single_chunk(self):
        rv = RunningVariance()
        rv.update(np.array([[1, 2], [3, 4]]))
        expected = np.array([1.0, 1.0])
        np.testing.assert_allclose(rv.result(), expected)

    def test_running_variance_multiple_chunks(self):
        rv = RunningVariance()
        rv.update(np.array([[1], [3]]))
        rv.update(np.array([[5]]))
        expected = np.array([8 / 3])
        np.testing.assert_allclose(rv.result(), expected)

    def test_running_variance_zero_variance(self):
        rv = RunningVariance()
        rv.update(np.array([[2], [2], [2]]))
        expected = np.array([0.0])
        np.testing.assert_allclose(rv.result(), expected)

    def test_running_histogram(self):
        hist = RunningHistogram(bins=2, value_range=(0, 4))
        hist.update(np.array([0, 1, 2, 3]))
        counts, edges = hist.result()

        expected_counts = np.array([2, 2])
        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(len(edges), 3)

    def test_running_histogram_multiple_chunks(self):
        hist = RunningHistogram(bins=2, value_range=(0, 4))
        hist.update(np.array([0, 1]))
        hist.update(np.array([2, 3]))
        counts, _ = hist.result()

        expected_counts = np.array([2, 2])
        np.testing.assert_array_equal(counts, expected_counts)

    def test_running_quantile_median(self):
        rq = RunningQuantile(q=0.5)
        rq.update(np.array([1, 2, 3, 4, 5]))
        self.assertEqual(rq.result(), 3.0)

    def test_streaming_stats_result(self):
        stats = StreamingStats(histogram_bins=2, histogram_range=(0, 4))
        stats.update_stats(np.array([[0, 1], [2, 3]]))
        result = stats.result()

        self.assertIn("mean", result)
        self.assertIn("variance", result)
        self.assertIn("histogram_counts", result)
        self.assertIn("histogram_edges", result)

    def test_running_mean_shape_mismatch(self):
        rm = RunningMean()
        rm.update(np.array([[1, 2]]))

        with self.assertRaises(ValueError):
            rm.update(np.array([[1, 2, 3]]))

    def test_empty_chunk_raises_error(self):
        rm = RunningMean()

        with self.assertRaises(ValueError):
            rm.update(np.array([]))


if __name__ == "__main__":
    unittest.main()