"""
visualise.py

Lightweight visualisation utilities for NumCompute Stream.

Only matplotlib is used.

Required assignment functions:
- plot_metric_over_time(metric_values, title, ylabel)
- compare_models(metric1, metric2, labels)
- plot_predictions_vs_ground_truth(y_true, y_pred)

Each function supports:
- save_path for saving figures
- show=True/False for inline display control
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(
    metric_values,
    title="Metric over Time",
    ylabel="Metric",
    xlabel="Chunk",
    save_path=None,
    show=True,
):
    """
    Plot one metric across streaming chunks.

    Parameters
    ----------
    metric_values : array-like
        Metric values recorded over chunks.

    title : str
        Plot title.

    ylabel : str
        Y-axis label.

    xlabel : str
        X-axis label.

    save_path : str or None
        If provided, save the plot to this path.

    show : bool
        Whether to display the plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    values = np.asarray(metric_values, dtype=float)

    if values.ndim != 1:
        raise ValueError("metric_values must be a 1D sequence.")

    if values.shape[0] == 0:
        raise ValueError("metric_values must not be empty.")

    chunks = np.arange(values.shape[0])

    fig, ax = plt.subplots()
    ax.plot(chunks, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model Comparison Over Time",
    ylabel="Metric",
    xlabel="Chunk",
    save_path=None,
    show=True,
):
    """
    Compare two models using metric values over chunks.

    Parameters
    ----------
    metric1 : array-like
        Metric values for first model.

    metric2 : array-like
        Metric values for second model.

    labels : tuple
        Names of the two models.

    title : str
        Plot title.

    ylabel : str
        Y-axis label.

    xlabel : str
        X-axis label.

    save_path : str or None
        If provided, save plot to file.

    show : bool
        Whether to display the plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    metric1 = np.asarray(metric1, dtype=float)
    metric2 = np.asarray(metric2, dtype=float)

    if metric1.ndim != 1 or metric2.ndim != 1:
        raise ValueError("metric1 and metric2 must be 1D sequences.")

    if metric1.shape[0] == 0 or metric2.shape[0] == 0:
        raise ValueError("metric values must not be empty.")

    min_len = min(metric1.shape[0], metric2.shape[0])

    metric1 = metric1[:min_len]
    metric2 = metric2[:min_len]

    chunks = np.arange(min_len)

    fig, ax = plt.subplots()
    ax.plot(chunks, metric1, marker="o", label=labels[0])
    ax.plot(chunks, metric2, marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs Ground Truth",
    save_path=None,
    show=True,
):
    """
    Visualise predicted labels against actual labels.

    Parameters
    ----------
    y_true : array-like
        Actual labels.

    y_pred : array-like
        Predicted labels.

    title : str
        Plot title.

    save_path : str or None
        If provided, save plot to file.

    show : bool
        Whether to display the plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of samples.")

    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    sample_index = np.arange(y_true.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(sample_index, y_true, marker="o", label="Ground Truth")
    ax.scatter(sample_index, y_pred, marker="x", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class Label")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_error_over_time(
    error_values,
    title="Error Over Time",
    save_path=None,
    show=True,
):
    """
    Convenience function for plotting error across chunks.
    """
    return plot_metric_over_time(
        error_values,
        title=title,
        ylabel="Error",
        xlabel="Chunk",
        save_path=save_path,
        show=show,
    )


def plot_memory_usage(
    memory_values,
    title="Memory Usage Over Time",
    save_path=None,
    show=True,
):
    """
    Plot memory usage across chunks.
    """
    return plot_metric_over_time(
        memory_values,
        title=title,
        ylabel="Memory Bytes",
        xlabel="Chunk",
        save_path=save_path,
        show=show,
    )