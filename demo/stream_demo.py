"""

This script:
1. Loads a CSV dataset using custom io.py
2. Splits data into streaming chunks
3. Trains a Decision Tree pipeline incrementally
4. Trains a Random Forest pipeline incrementally
5. Logs accuracy/error/memory over chunks
6. Saves visualisations using visualise.py

Run from project root:

    python demo/stream_demo.py
"""

import os
import sys

# Allows running this file directly from the demo folder or project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from numcompute_stream.io import load_csv, make_chunks
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_error_over_time,
    plot_memory_usage,
)


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "sample_dataset.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    X, y = load_csv(dataset_path, target_column=-1, skip_header=True)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    chunks = make_chunks(X, y, chunk_size=8)

    print(f"Number of streaming chunks: {len(chunks)}")

    tree_pipeline = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=2,
            criterion="gini"
        )),
    ])

    forest_pipeline = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=7,
            max_depth=4,
            min_samples_split=2,
            criterion="gini",
            random_state=42
        )),
    ])

    tree_trainer = StreamTrainer(tree_pipeline, verbose=True)
    forest_trainer = StreamTrainer(forest_pipeline, verbose=True)

    print("\nTraining Decision Tree stream...")
    for X_chunk, y_chunk in chunks:
        tree_trainer.fit_chunk(X_chunk, y_chunk)

    print("\nTraining Random Forest stream...")
    for X_chunk, y_chunk in chunks:
        forest_trainer.fit_chunk(X_chunk, y_chunk)

    tree_logs = tree_trainer.get_logs()
    forest_logs = forest_trainer.get_logs()

    print("\nFinal Results")
    print("-------------")
    print(f"Decision Tree final cumulative accuracy: {tree_logs['cumulative_accuracy'][-1]:.4f}")
    print(f"Random Forest final cumulative accuracy: {forest_logs['cumulative_accuracy'][-1]:.4f}")

    print("\nSaving visualisations...")

    plot_metric_over_time(
        tree_logs["accuracy"],
        title="Decision Tree Accuracy Over Streaming Chunks",
        ylabel="Accuracy",
        save_path=os.path.join(output_dir, "tree_accuracy.png"),
        show=False,
    )

    plot_metric_over_time(
        forest_logs["accuracy"],
        title="Random Forest Accuracy Over Streaming Chunks",
        ylabel="Accuracy",
        save_path=os.path.join(output_dir, "forest_accuracy.png"),
        show=False,
    )

    compare_models(
        tree_logs["accuracy"],
        forest_logs["accuracy"],
        labels=("Decision Tree", "Random Forest"),
        title="Streaming Accuracy Comparison",
        ylabel="Accuracy",
        save_path=os.path.join(output_dir, "model_comparison.png"),
        show=False,
    )

    plot_error_over_time(
        forest_logs["error"],
        title="Random Forest Error Over Streaming Chunks",
        save_path=os.path.join(output_dir, "forest_error.png"),
        show=False,
    )

    plot_memory_usage(
        forest_logs["memory_bytes"],
        title="Random Forest Memory Footprint Per Chunk",
        save_path=os.path.join(output_dir, "forest_memory.png"),
        show=False,
    )

    latest_X, latest_y = chunks[-1]
    latest_pred = forest_trainer.predict(latest_X)

    plot_predictions_vs_ground_truth(
        latest_y,
        latest_pred,
        title="Latest Chunk: Predictions vs Ground Truth",
        save_path=os.path.join(output_dir, "predictions_vs_ground_truth.png"),
        show=False,
    )

    print(f"Visualisations saved in: {output_dir}")

    print("\nLatest chunk predictions:")
    print("Ground truth:", latest_y.astype(int))
    print("Predictions :", latest_pred.astype(int))


if __name__ == "__main__":
    main()