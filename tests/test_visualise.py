import unittest
import numpy as np
import matplotlib

matplotlib.use("Agg")

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_error_over_time,
    plot_memory_usage,
)


class TestVisualise(unittest.TestCase):

    def test_plot_metric_over_time(self):
        fig, ax = plot_metric_over_time(
            [0.5, 0.6, 0.7],
            title="Accuracy",
            ylabel="Accuracy",
            show=False
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_compare_models(self):
        fig, ax = compare_models(
            [0.5, 0.6, 0.7],
            [0.4, 0.65, 0.75],
            labels=("Tree", "Forest"),
            show=False
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_predictions_vs_ground_truth(self):
        fig, ax = plot_predictions_vs_ground_truth(
            np.array([0, 1, 1]),
            np.array([0, 0, 1]),
            show=False
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_error_over_time(self):
        fig, ax = plot_error_over_time(
            [0.5, 0.4, 0.3],
            show=False
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_memory_usage(self):
        fig, ax = plot_memory_usage(
            [100, 200, 300],
            show=False
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_empty_metric_raises_error(self):
        with self.assertRaises(ValueError):
            plot_metric_over_time([], show=False)

    def test_prediction_shape_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            plot_predictions_vs_ground_truth([0, 1], [0], show=False)


if __name__ == "__main__":
    unittest.main()