# pylint: skip-file
"""Module for plotting reliability diagrams to visualize the calibration of probabilistic predictions."""

from pathlib import Path
from typing import Dict, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# pylint: disable=useless-param-doc
def human_format(num: int, _: None) -> str:
    """
    Format large numbers into human-readable strings.
    Args:
        num (int or float): The number to format.
        _: Unused parameter for compatibility with matplotlib's FuncFormatter.
    Returns:
        str: Formatted string representation of the number.
    """

    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    if num >= 1e6:
        return f"{num/1e6:.1f}M"
    if num >= 1e3:
        return f"{num/1e3:.0f}k"
    return str(int(num))


def _create_bin_data(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """Create bin data for calibration plot.

    Args:
        y_pred: Predicted probabilities
        y_true: True labels
        n_bins: Number of bins

    Returns:
        tuple: (bin_centers, freq_obs, freq_pred, bin_edges)
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges, right=True)

    freq_obs = np.full((n_bins,), float("nan"))
    freq_pred = np.full((n_bins,), float("nan"))
    bin_counts = np.zeros(n_bins)

    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if np.any(mask):
            bin_probs = y_pred[mask]
            bin_true = y_true[mask]
            freq_pred[i - 1] = np.mean(bin_probs).item()
            freq_obs[i - 1] = np.mean(bin_true).item()
            bin_counts[i - 1] = len(bin_probs)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, freq_obs, freq_pred, bin_edges, bin_counts


def _create_adaptive_bin_data(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """Create bin data for calibration plot using adaptive binning (empirical quantiles).

    Args:
        y_pred: Predicted probabilities
        y_true: True labels
        n_bins: Number of bins

    Returns:
        tuple: (bin_centers, freq_obs, freq_pred, bin_edges, bin_counts)
    """

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    # Sort predictions to easily determine quantiles
    sorted_indices = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_indices]
    y_true[sorted_indices]

    # Define bin edges using empirical quantiles
    bin_edges = [0.0]
    for i in range(1, n_bins):
        quantile = i / n_bins
        bin_edges.append(np.percentile(y_pred_sorted, quantile * 100))
    bin_edges.append(1.0)
    bin_edges = np.array(bin_edges)

    freq_obs = np.full((n_bins,), float("nan"))
    freq_pred = np.full((n_bins,), float("nan"))
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        # Handle cases where quantile edges might be identical if many predictions are the same
        # This prevents empty bins when predictions are highly concentrated
        if i == n_bins - 1:  # Last bin includes 1.0
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        else:  # Other bins exclude the upper edge
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])

        if np.any(mask):
            bin_probs = y_pred[mask]
            bin_true = y_true[mask]
            freq_pred[i] = np.mean(bin_probs).item()
            freq_obs[i] = np.mean(bin_true).item()
            bin_counts[i] = len(bin_probs)
        else:  # If a bin is empty due to repeated prediction values, set count to 0
            bin_counts[i] = 0

    # For plotting, use the mean of the predictions within each bin as the center
    # This better represents the adaptive nature.
    # If a bin is empty, its center will be nan, which matplotlib handles gracefully.

    actual_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    actual_bin_centers = np.array(actual_bin_centers)

    # Filter out NaN values for plotting
    valid_bins = ~np.isnan(freq_obs) & ~np.isnan(freq_pred)
    freq_obs = freq_obs[valid_bins]
    freq_pred = freq_pred[valid_bins]
    actual_bin_centers = actual_bin_centers[valid_bins]

    return actual_bin_centers, freq_obs, freq_pred, bin_edges, bin_counts


def _calculate_adaptive_ece(
    freq_obs: np.ndarray, freq_pred: np.ndarray, bin_counts: np.ndarray
) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        freq_obs: Conditional observed probabilities for each bin.
        freq_pred: Average predicted probabilities for each bin.
        bin_counts: Number of samples in each bin.

    Returns:
        float: The ECE value.
    """
    total_samples = np.sum(bin_counts)
    if total_samples == 0:
        return 0.0

    ece = 0.0
    for i in range(len(freq_obs)):
        if bin_counts[i] > 0:
            ece += (np.abs(freq_obs[i] - freq_pred[i]) * bin_counts[i]) / total_samples
    return ece


def calculate_adaptive_ece(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """Calculate Expected Calibration Error (ECE).

    Args:
        y_pred: Predicted probabilities
        y_true: True labels
        n_bins: Number of bins

    Returns:
        float: The ECE value.
    """
    bin_centers, freq_obs, freq_pred, bin_edges, bin_counts = _create_adaptive_bin_data(
        y_pred, y_true, n_bins
    )

    total_samples = np.sum(bin_counts)
    if total_samples == 0:
        return 0.0

    ece = 0.0
    for i in range(len(freq_obs)):
        if bin_counts[i] > 0:
            ece += (np.abs(freq_obs[i] - freq_pred[i]) * bin_counts[i]) / total_samples
    return ece


def plot_calibration(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    save_path: Path,
    tag: str,
    sample_dictionary: Dict,
    curve_label: Union[str, None] = None,
    binning: str = "uniform",
    n_bins: int = 10,
) -> matplotlib.axes.Axes:
    """Plot the observed proportion vs prediction proportion of outputs falling into a
    range of intervals, and display miscalibration area.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        save_path: Path: The path to save the plot.
        tag: Tag for saving the plot file.
        sample_dictionary: Dictionary containing sample information including ECE value.
        curve_label: legend label str for calibration curve.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    _, ax = plt.subplots(figsize=(5, 5))

    if curve_label is None:
        curve_label = "Predictor"

    # Create bin data
    if binning == "uniform":
        bin_centers, freq_obs, freq_pred, bin_edges, bin_counts = _create_bin_data(y_pred, y_true)
        bar_widths = np.diff(bin_edges) * 0.9
        ECE = sample_dictionary["ECE"]
        suffix = "uniform"
    elif binning == "adaptive":
        bin_centers, freq_obs, freq_pred, bin_edges, bin_counts = _create_adaptive_bin_data(
            y_pred, y_true
        )
        ECE = sample_dictionary["ECE_adaptive"]
        suffix = "adaptive"
        valid_bin_edges = bin_edges[
            [np.where(~np.isnan(freq_obs))[0][0], *np.where(~np.isnan(freq_obs))[0] + 1]
        ]
        bar_widths = (
            np.diff(valid_bin_edges) if len(valid_bin_edges) > 1 else 0.1
        )  # Default width if only one bin

    # Plot background barplot
    ax_twin = ax.twinx()
    ax_twin.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    ax_twin.bar(
        bin_centers,
        freq_obs,
        width=bar_widths,
        align="center",
        color="lightgray",
        edgecolor=None,
        alpha=0.5,
    )
    ax_twin.set_ylim([0, 1])
    ax_twin.set_ylabel("Count", color="gray")
    ax_twin.tick_params(axis="y", labelcolor="gray")
    # ax_twin.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))

    # Plot
    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")
    ax.plot(freq_pred, freq_obs, label=curve_label, c="#1f77b4")
    ax.fill_between(freq_pred, freq_pred, freq_obs, alpha=0.2)
    # ax.plot(bin_centers, freq_obs, label=curve_label, c="#1f77b4")
    # ax.fill_between(bin_centers, bin_centers, freq_obs, alpha=0.2)

    # Format plot
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Conditional Observed Probability")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title(f"Reliability Diagram - ECE {ECE:.5f} ({suffix})")

    # Add tiny frequency plot
    ax.set_adjustable("datalim")
    axins = inset_axes(
        ax,
        width="75%",
        height="75%",
        loc="upper left",
        bbox_to_anchor=(0.11, 0.7, 0.3, 0.25),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    # Keep the histogram with uniform bins
    bin_centers, freq_obs, freq_pred, bin_edges, bin_counts = _create_bin_data(y_pred, y_true)
    axins.hist(y_pred, bins=bin_edges, rwidth=0.8, color="lightcoral", edgecolor="black")
    axins.set_xticks([0, 1])
    axins.yaxis.set_major_formatter(mtick.FuncFormatter(human_format))
    axins.tick_params(axis="both", labelsize=8)
    axins.set_title("Pred Freq", fontdict={"fontsize": 8})

    save_path = save_path / f"reliability_plots_{suffix}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"{tag}.png")
    plt.close("all")

    return ax


if __name__ == "__main__":
    # Example usage with more data points to better demonstrate adaptive binning
    np.random.seed(42)
    num_samples = 1000

    # Generate some predictions with a bias (e.g., more low predictions)
    preds_biased = np.random.beta(0.5, 5, num_samples)  # Creates a distribution skewed towards 0
    targets_biased = (preds_biased + np.random.uniform(-0.1, 0.1, num_samples) > 0.5).astype(
        float
    )  # Simulate true labels
    targets_biased = np.clip(targets_biased, 0, 1)  # Ensure targets are 0 or 1

    # Ensure y_pred is a tensor of float32 for consistency with original script
    y_pred_tensor = torch.tensor(preds_biased, dtype=torch.float32)
    y_true_tensor = torch.tensor(targets_biased, dtype=torch.float32)

    # Create a dummy save path
    current_dir = Path.cwd()
    dummy_save_path = current_dir / "reliability_plots_output"
    dummy_save_path.mkdir(exist_ok=True)

    # Create sample dictionary for testing
    sample_dict = {"ECE": 0.0}  # ECE will be updated by the plotting function

    plot_calibration(
        y_pred_tensor,
        y_true_tensor,
        dummy_save_path,
        "adaptive_bins_test",
        sample_dict,
        n_bins=10,
        binning="adaptive",
    )
