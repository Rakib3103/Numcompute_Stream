"""
benchmark_streaming.py

Benchmark script for NumCompute Stream.

This script compares:
1. DecisionTreeClassifier
2. RandomForestClassifier

under a streaming scenario.

It measures:
- total fit time
- average chunk fit time
- prediction time
- final cumulative accuracy

Run from project root:

    python benchmark/benchmark_streaming.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from numcompute_stream.io import load_csv, make_chunks
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer


def run_benchmark(model_name, pipeline, chunks):
    """
    Run streaming benchmark for one model.
    """
    trainer = StreamTrainer(pipeline)

    start_total = time.perf_counter()

    for X_chunk, y_chunk in chunks:
        trainer.fit_chunk(X_chunk, y_chunk)

    total_fit_time = time.perf_counter() - start_total

    logs = trainer.get_logs()

    latest_X, latest_y = chunks[-1]

    start_pred = time.perf_counter()
    latest_pred = trainer.predict(latest_X)
    prediction_time = time.perf_counter() - start_pred

    latest_accuracy = float(np.mean(latest_pred == latest_y))

    result = {
        "model": model_name,
        "chunks": len(chunks),
        "total_fit_time": total_fit_time,
        "avg_chunk_fit_time": float(np.mean(logs["fit_time"])),
        "avg_score_time": float(np.mean(logs["score_time"])),
        "prediction_time_latest_chunk": prediction_time,
        "final_cumulative_accuracy": logs["cumulative_accuracy"][-1],
        "latest_chunk_accuracy": latest_accuracy,
        "avg_memory_bytes": float(np.mean(logs["memory_bytes"])),
    }

    return result


def print_results(results):
    """
    Print benchmark results in a readable table.
    """
    print("\nBenchmark Results")
    print("-----------------")

    header = (
        f"{'Model':<20}"
        f"{'Chunks':<10}"
        f"{'Total Fit(s)':<15}"
        f"{'Avg Fit(s)':<15}"
        f"{'Pred(s)':<15}"
        f"{'Final Acc':<12}"
        f"{'Avg Memory':<12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        row = (
            f"{result['model']:<20}"
            f"{result['chunks']:<10}"
            f"{result['total_fit_time']:<15.6f}"
            f"{result['avg_chunk_fit_time']:<15.6f}"
            f"{result['prediction_time_latest_chunk']:<15.6f}"
            f"{result['final_cumulative_accuracy']:<12.4f}"
            f"{result['avg_memory_bytes']:<12.2f}"
        )

        print(row)


def save_results(results, output_path):
    """
    Save benchmark results to a text file.
    """
    with open(output_path, "w") as file:
        file.write("Benchmark Results\n")
        file.write("=================\n\n")

        for result in results:
            file.write(f"Model: {result['model']}\n")
            file.write(f"Chunks: {result['chunks']}\n")
            file.write(f"Total fit time: {result['total_fit_time']:.6f} seconds\n")
            file.write(f"Average chunk fit time: {result['avg_chunk_fit_time']:.6f} seconds\n")
            file.write(f"Average score time: {result['avg_score_time']:.6f} seconds\n")
            file.write(
                f"Prediction time latest chunk: "
                f"{result['prediction_time_latest_chunk']:.6f} seconds\n"
            )
            file.write(
                f"Final cumulative accuracy: "
                f"{result['final_cumulative_accuracy']:.4f}\n"
            )
            file.write(f"Latest chunk accuracy: {result['latest_chunk_accuracy']:.4f}\n")
            file.write(f"Average memory bytes: {result['avg_memory_bytes']:.2f}\n")
            file.write("\n")


def main():
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "demo",
        "sample_dataset.csv"
    )

    output_path = os.path.join(os.path.dirname(__file__), "results.txt")

    X, y = load_csv(dataset_path, target_column=-1, skip_header=True)

    chunks = make_chunks(X, y, chunk_size=8)

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

    results = []

    results.append(
        run_benchmark(
            "Decision Tree",
            tree_pipeline,
            chunks
        )
    )

    results.append(
        run_benchmark(
            "Random Forest",
            forest_pipeline,
            chunks
        )
    )

    print_results(results)
    save_results(results, output_path)

    print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()