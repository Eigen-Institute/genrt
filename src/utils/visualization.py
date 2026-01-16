"""
Visualization utilities for RT-TDDFT ML training and evaluation.

Provides plotting functions for:
- Training curves (loss, learning rate)
- Error accumulation over rollout steps
- Physics constraint violations
- Absorption spectrum comparisons
- Density matrix visualizations
- Dipole moment trajectories
"""

import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass

# Try to import matplotlib, provide fallback message if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = Any
    Axes = Any


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


@dataclass
class PlotStyle:
    """Configuration for plot styling."""
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 100
    font_size: int = 12
    line_width: float = 1.5
    marker_size: float = 4
    grid: bool = True
    tight_layout: bool = True


DEFAULT_STYLE = PlotStyle()


def setup_style(style: Optional[PlotStyle] = None):
    """Apply plot styling."""
    _check_matplotlib()
    style = style or DEFAULT_STYLE

    plt.rcParams.update({
        'font.size': style.font_size,
        'lines.linewidth': style.line_width,
        'lines.markersize': style.marker_size,
        'figure.figsize': style.figsize,
        'figure.dpi': style.dpi,
        'axes.grid': style.grid,
    })


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Progress",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    log_scale: bool = True,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Optional validation losses per epoch
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic y-scale
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    fig, ax = plt.subplots()

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", marker='o', markersize=3)

    if val_losses is not None:
        val_epochs = range(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, label="Validation", marker='s', markersize=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if log_scale:
        ax.set_yscale('log')

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=style.dpi if style else 100, bbox_inches='tight')

    return fig


def plot_loss_components(
    loss_history: Dict[str, List[float]],
    title: str = "Loss Components",
    log_scale: bool = True,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot individual loss components over training.

    Args:
        loss_history: Dict mapping loss name to list of values
        title: Plot title
        log_scale: Use logarithmic y-scale
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    fig, ax = plt.subplots()

    for name, values in loss_history.items():
        steps = range(1, len(values) + 1)
        ax.plot(steps, values, label=name, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc='upper right')

    if log_scale:
        ax.set_yscale('log')

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_learning_rate(
    lr_history: List[float],
    title: str = "Learning Rate Schedule",
    log_scale: bool = True,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot learning rate over training steps.

    Args:
        lr_history: Learning rate values per step
        title: Plot title
        log_scale: Use logarithmic y-scale
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    fig, ax = plt.subplots()

    steps = range(1, len(lr_history) + 1)
    ax.plot(steps, lr_history, color='tab:green')

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale('log')

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_error_accumulation(
    step_errors: Union[Tensor, np.ndarray, List[float]],
    cumulative_errors: Optional[Union[Tensor, np.ndarray, List[float]]] = None,
    dt: Optional[float] = None,
    title: str = "Error Accumulation",
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot error accumulation over rollout steps.

    Args:
        step_errors: Per-step errors
        cumulative_errors: Optional cumulative errors
        dt: Time step (for time axis instead of step axis)
        title: Plot title
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    if isinstance(step_errors, Tensor):
        step_errors = step_errors.detach().cpu().numpy()
    step_errors = np.array(step_errors)

    n_steps = len(step_errors)
    if dt is not None:
        x = np.arange(n_steps) * dt
        xlabel = "Time (a.u.)"
    else:
        x = np.arange(n_steps)
        xlabel = "Step"

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Step Error", color=color1)
    ax1.plot(x, step_errors, color=color1, label="Step Error")
    ax1.tick_params(axis='y', labelcolor=color1)

    if cumulative_errors is not None:
        if isinstance(cumulative_errors, Tensor):
            cumulative_errors = cumulative_errors.detach().cpu().numpy()
        cumulative_errors = np.array(cumulative_errors)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel("Cumulative Error", color=color2)
        ax2.plot(x, cumulative_errors, color=color2, linestyle='--', label="Cumulative")
        ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title(title)

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_physics_violations(
    trace_violations: Union[Tensor, np.ndarray, List[float]],
    hermiticity_violations: Optional[Union[Tensor, np.ndarray, List[float]]] = None,
    idempotency_violations: Optional[Union[Tensor, np.ndarray, List[float]]] = None,
    dt: Optional[float] = None,
    title: str = "Physics Constraint Violations",
    log_scale: bool = True,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot physics constraint violations over trajectory.

    Args:
        trace_violations: Trace conservation violations
        hermiticity_violations: Optional Hermiticity violations
        idempotency_violations: Optional idempotency violations
        dt: Time step for x-axis
        title: Plot title
        log_scale: Use logarithmic y-scale
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    def to_numpy(x):
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    trace_violations = to_numpy(trace_violations)
    n_steps = len(trace_violations)

    if dt is not None:
        x = np.arange(n_steps) * dt
        xlabel = "Time (a.u.)"
    else:
        x = np.arange(n_steps)
        xlabel = "Step"

    fig, ax = plt.subplots()

    ax.plot(x, trace_violations, label="Trace", color='tab:blue')

    if hermiticity_violations is not None:
        hermiticity_violations = to_numpy(hermiticity_violations)
        ax.plot(x, hermiticity_violations, label="Hermiticity", color='tab:orange')

    if idempotency_violations is not None:
        idempotency_violations = to_numpy(idempotency_violations)
        ax.plot(x, idempotency_violations, label="Idempotency", color='tab:green')

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Violation")
    ax.set_title(title)
    ax.legend()

    if log_scale:
        ax.set_yscale('log')

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_absorption_spectrum(
    freqs: Union[Tensor, np.ndarray],
    spectrum_pred: Union[Tensor, np.ndarray],
    spectrum_true: Optional[Union[Tensor, np.ndarray]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    title: str = "Absorption Spectrum",
    normalize: bool = True,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot absorption spectrum comparison.

    Args:
        freqs: Frequency values (in eV)
        spectrum_pred: Predicted absorption spectrum
        spectrum_true: Optional true absorption spectrum
        freq_range: Optional (min, max) frequency range to display
        title: Plot title
        normalize: Normalize spectra to unit area
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    def to_numpy(x):
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    freqs = to_numpy(freqs)
    spectrum_pred = to_numpy(spectrum_pred)

    if spectrum_true is not None:
        spectrum_true = to_numpy(spectrum_true)

    # Apply frequency range filter
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        spectrum_pred = spectrum_pred[mask]
        if spectrum_true is not None:
            spectrum_true = spectrum_true[mask]

    # Normalize
    if normalize:
        spectrum_pred = spectrum_pred / (spectrum_pred.sum() + 1e-10)
        if spectrum_true is not None:
            spectrum_true = spectrum_true / (spectrum_true.sum() + 1e-10)

    fig, ax = plt.subplots()

    ax.plot(freqs, spectrum_pred, label="Predicted", color='tab:blue')

    if spectrum_true is not None:
        ax.plot(freqs, spectrum_true, label="True", color='tab:orange', linestyle='--')
        ax.fill_between(freqs, spectrum_pred, spectrum_true, alpha=0.2, color='gray')

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption (a.u.)")
    ax.set_title(title)
    ax.legend()

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_density_matrix(
    rho: Union[Tensor, np.ndarray],
    component: str = "real",
    title: Optional[str] = None,
    colorbar: bool = True,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot density matrix as heatmap.

    Args:
        rho: Density matrix (n, n)
        component: "real", "imag", "abs", or "phase"
        title: Plot title
        colorbar: Show colorbar
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    if isinstance(rho, Tensor):
        rho = rho.detach().cpu().numpy()

    if component == "real":
        data = rho.real
        default_title = "Density Matrix (Real)"
        cmap = "RdBu_r"
    elif component == "imag":
        data = rho.imag
        default_title = "Density Matrix (Imaginary)"
        cmap = "RdBu_r"
    elif component == "abs":
        data = np.abs(rho)
        default_title = "Density Matrix (Magnitude)"
        cmap = "viridis"
    elif component == "phase":
        data = np.angle(rho)
        default_title = "Density Matrix (Phase)"
        cmap = "twilight"
    else:
        raise ValueError(f"Unknown component: {component}")

    title = title or default_title

    fig, ax = plt.subplots()

    # Symmetric colormap for real/imag
    if component in ["real", "imag"]:
        vmax = np.abs(data).max()
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    else:
        im = ax.imshow(data, cmap=cmap, aspect='equal')

    ax.set_xlabel("Basis Index")
    ax.set_ylabel("Basis Index")
    ax.set_title(title)

    if colorbar:
        plt.colorbar(im, ax=ax)

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_density_comparison(
    rho_pred: Union[Tensor, np.ndarray],
    rho_true: Union[Tensor, np.ndarray],
    component: str = "abs",
    title: str = "Density Matrix Comparison",
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot predicted vs true density matrices side by side.

    Args:
        rho_pred: Predicted density matrix
        rho_true: True density matrix
        component: "real", "imag", "abs", or "phase"
        title: Plot title
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    def to_numpy(x):
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    rho_pred = to_numpy(rho_pred)
    rho_true = to_numpy(rho_true)

    if component == "real":
        pred_data = rho_pred.real
        true_data = rho_true.real
        cmap = "RdBu_r"
    elif component == "imag":
        pred_data = rho_pred.imag
        true_data = rho_true.imag
        cmap = "RdBu_r"
    elif component == "abs":
        pred_data = np.abs(rho_pred)
        true_data = np.abs(rho_true)
        cmap = "viridis"
    elif component == "phase":
        pred_data = np.angle(rho_pred)
        true_data = np.angle(rho_true)
        cmap = "twilight"
    else:
        raise ValueError(f"Unknown component: {component}")

    diff = pred_data - true_data

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Common color scale for pred/true
    vmin = min(pred_data.min(), true_data.min())
    vmax = max(pred_data.max(), true_data.max())

    if component in ["real", "imag"]:
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs

    im0 = axes[0].imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title("Predicted")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(true_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title("True")
    plt.colorbar(im1, ax=axes[1])

    # Difference with symmetric colormap
    diff_max = np.abs(diff).max()
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max, aspect='equal')
    axes[2].set_title("Difference")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title)

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_dipole_trajectory(
    dipole_pred: Union[Tensor, np.ndarray],
    dipole_true: Optional[Union[Tensor, np.ndarray]] = None,
    dt: Optional[float] = None,
    components: List[str] = ["x", "y", "z"],
    title: str = "Dipole Moment Trajectory",
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot dipole moment trajectory over time.

    Args:
        dipole_pred: Predicted dipole moments (n_steps, 3)
        dipole_true: Optional true dipole moments
        dt: Time step for x-axis
        components: Which components to plot ("x", "y", "z")
        title: Plot title
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    def to_numpy(x):
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    dipole_pred = to_numpy(dipole_pred)
    n_steps = dipole_pred.shape[0]

    if dipole_true is not None:
        dipole_true = to_numpy(dipole_true)

    if dt is not None:
        x = np.arange(n_steps) * dt
        xlabel = "Time (a.u.)"
    else:
        x = np.arange(n_steps)
        xlabel = "Step"

    component_map = {"x": 0, "y": 1, "z": 2}
    colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}

    fig, ax = plt.subplots()

    for comp in components:
        idx = component_map[comp]
        color = colors[comp]

        ax.plot(x, dipole_pred[:, idx], label=f"Pred {comp}", color=color)

        if dipole_true is not None:
            ax.plot(x, dipole_true[:, idx], label=f"True {comp}",
                   color=color, linestyle='--', alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dipole Moment (a.u.)")
    ax.set_title(title)
    ax.legend()

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    title: str = "Metrics Comparison",
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot bar chart comparing metrics across experiments.

    Args:
        metrics_dict: Dict mapping experiment name to dict of metrics
        metric_names: Which metrics to plot (None for all)
        title: Plot title
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    experiment_names = list(metrics_dict.keys())
    first_metrics = metrics_dict[experiment_names[0]]

    if metric_names is None:
        metric_names = list(first_metrics.keys())

    n_metrics = len(metric_names)
    n_experiments = len(experiment_names)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2), 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_experiments

    for i, exp_name in enumerate(experiment_names):
        values = [metrics_dict[exp_name].get(m, 0) for m in metric_names]
        offset = (i - n_experiments / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=exp_name)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_curriculum_progress(
    stage_epochs: List[int],
    stage_horizons: List[int],
    current_epoch: int,
    title: str = "Curriculum Progress",
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Plot curriculum learning progress.

    Args:
        stage_epochs: Epoch when each stage starts
        stage_horizons: Horizon for each stage
        current_epoch: Current training epoch
        title: Plot title
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    fig, ax = plt.subplots()

    # Create step function for curriculum
    epochs = []
    horizons = []
    for i, (start, horizon) in enumerate(zip(stage_epochs, stage_horizons)):
        epochs.append(start)
        horizons.append(horizon)
        # Add end of stage
        if i < len(stage_epochs) - 1:
            epochs.append(stage_epochs[i + 1])
            horizons.append(horizon)

    # Extend to current epoch
    if current_epoch > epochs[-1]:
        epochs.append(current_epoch)
        horizons.append(horizons[-1])

    ax.step(epochs, horizons, where='post', color='tab:blue', linewidth=2)
    ax.fill_between(epochs, horizons, step='post', alpha=0.3, color='tab:blue')

    # Mark current position
    current_horizon = stage_horizons[-1]
    for i, start in enumerate(stage_epochs):
        if current_epoch < start:
            current_horizon = stage_horizons[max(0, i - 1)]
            break

    ax.axvline(current_epoch, color='tab:red', linestyle='--', label=f'Current: {current_epoch}')
    ax.scatter([current_epoch], [current_horizon], color='tab:red', s=100, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Horizon (steps)")
    ax.set_title(title)
    ax.legend()

    if style and style.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def create_training_dashboard(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    loss_components: Optional[Dict[str, List[float]]] = None,
    lr_history: Optional[List[float]] = None,
    physics_violations: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Figure:
    """
    Create comprehensive training dashboard with multiple subplots.

    Args:
        train_losses: Training losses per epoch
        val_losses: Optional validation losses
        loss_components: Optional dict of loss component histories
        lr_history: Optional learning rate history
        physics_violations: Optional dict of physics violation histories
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    setup_style(style)

    # Determine number of subplots needed
    n_plots = 1  # Always have main loss
    if loss_components:
        n_plots += 1
    if lr_history:
        n_plots += 1
    if physics_violations:
        n_plots += 1

    n_cols = min(2, n_plots)
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0

    # Main loss plot
    ax = axes[plot_idx]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", marker='o', markersize=2)
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val", marker='s', markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    plot_idx += 1

    # Loss components
    if loss_components and plot_idx < len(axes):
        ax = axes[plot_idx]
        for name, values in loss_components.items():
            ax.plot(range(1, len(values) + 1), values, label=name, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Components")
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)
        plot_idx += 1

    # Learning rate
    if lr_history and plot_idx < len(axes):
        ax = axes[plot_idx]
        ax.plot(range(1, len(lr_history) + 1), lr_history, color='tab:green')
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale('log')
        ax.grid(True)
        plot_idx += 1

    # Physics violations
    if physics_violations and plot_idx < len(axes):
        ax = axes[plot_idx]
        for name, values in physics_violations.items():
            ax.plot(range(1, len(values) + 1), values, label=name, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Violation")
        ax.set_title("Physics Violations")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_figure(fig: Figure, path: str, dpi: int = 150, formats: List[str] = ["png"]):
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        dpi: Resolution
        formats: List of formats ("png", "pdf", "svg")
    """
    _check_matplotlib()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi, bbox_inches='tight', format=fmt)


def close_all():
    """Close all open figures."""
    if HAS_MATPLOTLIB:
        plt.close('all')
