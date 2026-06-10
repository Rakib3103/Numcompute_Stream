# NumCompute Stream

NumCompute Stream is a lightweight machine learning framework built using only **Python, NumPy, and matplotlib**.

This project extends the original NumCompute package into a streaming, decision tree–based machine learning framework. It supports incremental preprocessing, streaming metrics, decision tree learning, tree-based ensembles, pipeline training, benchmarking, and visualisation.

---

## Features

* Custom CSV loading without pandas
* Streaming-compatible preprocessing:

  * `SimpleImputer`
  * `StandardScaler`
  * `MinMaxScaler`
  * `OneHotEncoder`
* Streaming statistics:

  * Running mean
  * Running variance
  * Running histogram
  * Running quantile
* Streaming metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion matrix
  * Rolling accuracy
* Decision tree classifier from scratch:

  * Gini impurity
  * Entropy impurity
  * Maximum depth
  * Minimum samples split
  * NaN handling
  * Tie handling
* Ensemble models:

  * Bagging classifier
  * Random forest classifier
* Stream-compatible pipeline
* Stream trainer with per-chunk logs
* Matplotlib visualisation module
* Benchmark script
* Unit tests covering standard and edge cases

---

## Project Structure

```text
Assignment/
│
├── numcompute_stream/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── stats.py
│   ├── metrics.py
│   ├── tree.py
│   ├── ensemble.py
│   ├── pipeline.py
│   ├── stream.py
│   └── visualise.py
│
├── tests/
│   ├── test_metrics.py
│   ├── test_stats.py
│   ├── test_preprocessing.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_pipeline.py
│   ├── test_stream.py
│   └── test_visualise.py
│
├── demo/
│   ├── sample_dataset.csv
│   ├── stream_demo.py
│   └── outputs/
│
├── benchmark/
│   ├── benchmark_streaming.py
│   └── results.txt
│
└── README.md
```

---

## Requirements

Only the following external libraries are used:

```text
numpy
matplotlib
```

No scikit-learn, pandas, PyTorch, TensorFlow, or other machine learning/data-processing libraries are used.

---

## How to Run Tests

From the project root folder, run:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests/test_tree.py
```

The project includes tests for:

* preprocessing
* statistics
* metrics
* decision tree
* ensemble models
* pipeline
* stream trainer
* visualisation

---

## How to Run the Demo

From the project root folder, run:

```bash
python demo/stream_demo.py
```

The demo will:

1. Load a CSV dataset using `io.py`
2. Split the dataset into chunks
3. Train a decision tree pipeline using `.partial_fit()`
4. Train a random forest pipeline using `.partial_fit()`
5. Log per-chunk accuracy, error, memory usage, and cumulative accuracy
6. Save visualisations into `demo/outputs/`

Example output:

```text
Loading dataset...
Dataset shape: X=(44, 3), y=(44,)
Number of streaming chunks: 6

Training Decision Tree stream...
Chunk 0: accuracy=1.0000, cumulative_accuracy=1.0000

Training Random Forest stream...
Chunk 0: accuracy=1.0000, cumulative_accuracy=1.0000

Final Results
-------------
Decision Tree final cumulative accuracy: 0.9545
Random Forest final cumulative accuracy: 0.9091
```

---

## Visualisations

The demo generates plots such as:

* Decision tree accuracy over chunks
* Random forest accuracy over chunks
* Model comparison over chunks
* Error over time
* Memory usage over time
* Predictions vs ground truth

These are saved in:

```text
demo/outputs/
```

---

## How to Run Benchmark

From the project root folder, run:

```bash
python benchmark/benchmark_streaming.py
```

The benchmark compares:

* `DecisionTreeClassifier`
* `RandomForestClassifier`

It measures:

* total fit time
* average chunk fit time
* score time
* prediction time
* final cumulative accuracy
* average memory usage

The results are saved to:

```text
benchmark/results.txt
```

---

## Basic Usage Example

```python
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline

pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=4))
])

pipe.partial_fit(X_chunk, y_chunk)
predictions = pipe.predict(X_test)
```

---

## Random Forest Streaming Example

```python
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler

pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=7,
        max_depth=4,
        random_state=42
    ))
])

pipe.partial_fit(X_chunk, y_chunk)
y_pred = pipe.predict(X_test)
```

---

## Design Notes

### Streaming Compatibility

All important components use a streaming-style API:

```python
partial_fit(X_chunk, y_chunk)
update(y_true_chunk, y_pred_chunk)
update_stats(X_chunk)
```

Preprocessing and metric modules naturally support incremental updates.

For decision trees and ensembles, this project uses a rebuild-on-update strategy. Each call to `.partial_fit()` stores the new chunk and rebuilds the model using all data seen so far. This keeps the implementation stable, understandable, and suitable for demonstrating streaming behaviour within the assignment constraints.

---

### Decision Tree

The decision tree is implemented from scratch using NumPy. It supports:

* Gini impurity
* Entropy impurity
* Recursive binary splitting
* Maximum depth
* Minimum split size
* NaN replacement using column means
* Tie-breaking using the smallest class label

At each node, the algorithm evaluates possible feature thresholds and selects the split that gives the highest impurity reduction. The tree stops growing when the maximum depth is reached, when the node has too few samples, or when all samples belong to the same class.

---

### Ensemble Learning

The ensemble module implements:

* Bagging
* Random Forest

Random forest uses:

* Bootstrap sampling
* Random feature subsets
* Majority voting

Each tree is trained on a different bootstrap sample. During prediction, all trees vote and the final class is selected by majority vote. Ties are resolved by selecting the smallest class label.

---

### Numerical Stability

The framework handles:

* NaN values
* Empty chunks
* Zero variance columns
* Zero division in metrics
* Shape mismatch errors
* Tie resolution
* All-NaN imputation columns

---

## Benchmark Summary

The benchmark compares a single decision tree with a random forest under the same streaming setup.

The decision tree is faster because only one tree is rebuilt per chunk. The random forest is slower because it trains multiple trees using bootstrap samples and random feature subsets. However, the random forest provides a stronger ensemble design because predictions are based on majority voting across several trees.

This demonstrates the trade-off between computational cost and predictive robustness.

---

## Demo Summary

The demo shows a full streaming workflow:

1. A CSV file is loaded using the custom `io.py` module.
2. The data is split into six chunks to simulate streaming input.
3. A decision tree pipeline and a random forest pipeline are trained chunk by chunk.
4. Accuracy, error, memory footprint, and cumulative accuracy are logged.
5. Matplotlib visualisations are saved into the `demo/outputs/` folder.
6. The latest chunk predictions are compared against the ground truth labels.

---

## Author

Mazharul Islam Rakib

Master of Computer Science
Programming for AI
