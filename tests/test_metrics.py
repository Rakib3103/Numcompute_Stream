import unittest
import numpy as np

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1Score,
    StreamingConfusionMatrix,
    RollingAccuracy,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class TestStreamingMetrics(unittest.TestCase):

    def test_accuracy_single_chunk(self):
        metric = StreamingAccuracy()
        metric.update([1, 0, 1, 1], [1, 0, 0, 1])
        self.assertAlmostEqual(metric.result(), 0.75)

    def test_accuracy_multiple_chunks(self):
        metric = StreamingAccuracy()
        metric.update([1, 0], [1, 1])
        metric.update([1, 1], [1, 1])
        self.assertAlmostEqual(metric.result(), 0.75)

    def test_accuracy_reset(self):
        metric = StreamingAccuracy()
        metric.update([1, 0], [1, 0])
        metric.reset()
        self.assertEqual(metric.result(), 0.0)

    def test_precision(self):
        metric = StreamingPrecision(positive_label=1)
        metric.update([1, 0, 1, 0], [1, 1, 1, 0])
        self.assertAlmostEqual(metric.result(), 2 / 3)

    def test_recall(self):
        metric = StreamingRecall(positive_label=1)
        metric.update([1, 0, 1, 1], [1, 0, 0, 1])
        self.assertAlmostEqual(metric.result(), 2 / 3)

    def test_f1_score(self):
        metric = StreamingF1Score(positive_label=1)
        metric.update([1, 0, 1, 0], [1, 1, 1, 0])
        self.assertAlmostEqual(metric.result(), 0.8)

    def test_confusion_matrix_binary(self):
        metric = StreamingConfusionMatrix(labels=[0, 1])
        metric.update([0, 0, 1, 1], [0, 1, 0, 1])
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(metric.result(), expected)

    def test_confusion_matrix_multiclass(self):
        metric = StreamingConfusionMatrix(labels=[0, 1, 2])
        metric.update([0, 1, 2, 2], [0, 2, 2, 1])

        expected = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1]
        ])

        np.testing.assert_array_equal(metric.result(), expected)

    def test_rolling_accuracy(self):
        metric = RollingAccuracy(window_size=3)
        metric.update([1, 1, 1], [1, 0, 1])
        self.assertAlmostEqual(metric.result(), 2 / 3)

        metric.update([0], [0])
        self.assertAlmostEqual(metric.result(), 2 / 3)

    def test_helper_accuracy_score(self):
        result = accuracy_score([1, 0, 1], [1, 1, 1])
        self.assertAlmostEqual(result, 2 / 3)

    def test_helper_precision_score(self):
        result = precision_score([1, 0, 1, 0], [1, 1, 1, 0])
        self.assertAlmostEqual(result, 2 / 3)

    def test_helper_recall_score(self):
        result = recall_score([1, 0, 1, 1], [1, 0, 0, 1])
        self.assertAlmostEqual(result, 2 / 3)

    def test_helper_f1_score(self):
        result = f1_score([1, 0, 1, 0], [1, 1, 1, 0])
        self.assertAlmostEqual(result, 0.8)

    def test_shape_mismatch_raises_error(self):
        metric = StreamingAccuracy()
        with self.assertRaises(ValueError):
            metric.update([1, 0], [1])


if __name__ == "__main__":
    unittest.main()