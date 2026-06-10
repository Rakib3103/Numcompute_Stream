import unittest
import numpy as np

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.metrics import StreamingAccuracy, RollingAccuracy


class TestStreamTrainer(unittest.TestCase):

    def test_stream_trainer_fit_chunk_tree(self):
        model = DecisionTreeClassifier(max_depth=2)
        trainer = StreamTrainer(model)

        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        trainer.fit_chunk(X, y)
        logs = trainer.get_logs()

        self.assertEqual(len(logs["accuracy"]), 1)
        self.assertEqual(logs["chunk"][0], 0)

    def test_stream_trainer_fit_multiple_chunks(self):
        model = DecisionTreeClassifier(max_depth=3)
        trainer = StreamTrainer(model)

        trainer.fit_chunk(np.array([[0], [1]]), np.array([0, 0]))
        trainer.fit_chunk(np.array([[2], [3]]), np.array([1, 1]))

        logs = trainer.get_logs()

        self.assertEqual(len(logs["accuracy"]), 2)
        self.assertEqual(logs["chunk"], [0, 1])

    def test_stream_trainer_pipeline(self):
        pipe = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2))
        ])

        trainer = StreamTrainer(pipe)

        X = np.array([
            [0.0, 1.0],
            [1.0, np.nan],
            [2.0, 2.0],
            [3.0, 2.0],
        ])
        y = np.array([0, 0, 1, 1])

        trainer.fit_chunk(X, y)
        logs = trainer.get_logs()

        self.assertEqual(len(logs["accuracy"]), 1)

    def test_stream_trainer_random_forest(self):
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        trainer = StreamTrainer(model)

        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([0, 0, 0, 1, 1, 1])

        trainer.fit_chunk(X, y)
        logs = trainer.get_logs()

        self.assertEqual(len(logs["accuracy"]), 1)

    def test_score_chunk(self):
        model = DecisionTreeClassifier(max_depth=2)
        trainer = StreamTrainer(model)

        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        trainer.fit_chunk(X, y)
        scores = trainer.score_chunk(X, y)

        self.assertIn("accuracy", scores)
        self.assertIn("error", scores)

    def test_fit_stream(self):
        model = DecisionTreeClassifier(max_depth=3)
        trainer = StreamTrainer(model)

        chunks = [
            (np.array([[0], [1]]), np.array([0, 0])),
            (np.array([[2], [3]]), np.array([1, 1])),
        ]

        trainer.fit_stream(chunks)
        logs = trainer.get_logs()

        self.assertEqual(len(logs["accuracy"]), 2)

    def test_memory_logged(self):
        model = DecisionTreeClassifier(max_depth=2)
        trainer = StreamTrainer(model)

        X = np.array([[0], [1]])
        y = np.array([0, 1])

        trainer.fit_chunk(X, y)
        logs = trainer.get_logs()

        self.assertGreater(logs["memory_bytes"][0], 0)

    def test_cumulative_accuracy_logged(self):
        model = DecisionTreeClassifier(max_depth=2)
        trainer = StreamTrainer(model)

        trainer.fit_chunk(np.array([[0], [1]]), np.array([0, 0]))
        trainer.fit_chunk(np.array([[2], [3]]), np.array([1, 1]))

        logs = trainer.get_logs()

        self.assertEqual(len(logs["cumulative_accuracy"]), 2)

    def test_custom_metrics(self):
        model = DecisionTreeClassifier(max_depth=2)

        metrics = {
            "accuracy": StreamingAccuracy(),
            "rolling_accuracy": RollingAccuracy(window_size=3)
        }

        trainer = StreamTrainer(model, metrics=metrics)

        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        trainer.fit_chunk(X, y)

        self.assertIn("accuracy", trainer.metrics)
        self.assertIn("rolling_accuracy", trainer.metrics)

    def test_reset_logs(self):
        model = DecisionTreeClassifier(max_depth=2)
        trainer = StreamTrainer(model)

        trainer.fit_chunk(np.array([[0], [1]]), np.array([0, 1]))
        trainer.reset_logs()

        logs = trainer.get_logs()
        self.assertEqual(len(logs["accuracy"]), 0)

    def test_invalid_model_raises_error(self):
        class BadModel:
            pass

        with self.assertRaises(AttributeError):
            StreamTrainer(BadModel())

    def test_shape_mismatch_raises_error(self):
        model = DecisionTreeClassifier(max_depth=2)
        trainer = StreamTrainer(model)

        with self.assertRaises(ValueError):
            trainer.fit_chunk(np.array([[0], [1]]), np.array([0]))


if __name__ == "__main__":
    unittest.main()