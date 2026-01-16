"""
Evaluation metrics for RT-TDDFT model performance.

Provides metrics for:
- Density matrix accuracy (Frobenius norm, relative error)
- Physical observable errors (dipole, populations)
- Physics constraint violations
- Spectral analysis (absorption spectrum overlap)
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryMetrics:
    """Metrics computed over a trajectory."""
    mse: float
    mae: float
    relative_error: float
    max_error: float
    dipole_error: Optional[float] = None
    trace_violation: Optional[float] = None
    hermiticity_violation: Optional[float] = None
    spectrum_overlap: Optional[float] = None


def frobenius_error(
    pred: Tensor,
    true: Tensor,
    normalize: bool = True,
) -> Tensor:
    """
    Compute Frobenius norm error between density matrices.

    Args:
        pred: Predicted density, shape (..., n, n)
        true: True density, shape (..., n, n)
        normalize: If True, normalize by true norm

    Returns:
        Frobenius error
    """
    diff = pred - true
    error = diff.abs().pow(2).sum(dim=(-2, -1)).sqrt()

    if normalize:
        norm = true.abs().pow(2).sum(dim=(-2, -1)).sqrt() + 1e-10
        error = error / norm

    return error


def mean_absolute_error(pred: Tensor, true: Tensor) -> Tensor:
    """Compute mean absolute error."""
    return (pred - true).abs().mean()


def max_absolute_error(pred: Tensor, true: Tensor) -> Tensor:
    """Compute maximum absolute error."""
    return (pred - true).abs().max()


def relative_error(pred: Tensor, true: Tensor, eps: float = 1e-10) -> Tensor:
    """
    Compute relative error.

    Args:
        pred: Predicted values
        true: True values
        eps: Small constant for stability

    Returns:
        Relative error
    """
    return (pred - true).abs().sum() / (true.abs().sum() + eps)


def dipole_error(
    rho_pred: Tensor,
    rho_true: Tensor,
    dipole_integrals: Tensor,
    relative: bool = True,
) -> Tensor:
    """
    Compute dipole moment error.

    Args:
        rho_pred: Predicted density, shape (..., n, n)
        rho_true: True density, shape (..., n, n)
        dipole_integrals: Dipole matrices, shape (3, n, n)
        relative: If True, return relative error

    Returns:
        Dipole error (scalar or per-component)
    """
    # Compute dipole moments: μ = Tr(ρ·D)
    dipole_pred = torch.einsum("...ij,cji->...c", rho_pred, dipole_integrals).real
    dipole_true = torch.einsum("...ij,cji->...c", rho_true, dipole_integrals).real

    diff = (dipole_pred - dipole_true).norm(dim=-1)

    if relative:
        norm = dipole_true.norm(dim=-1) + 1e-10
        return diff / norm

    return diff


def trace_violation(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
) -> Tensor:
    """
    Compute trace conservation violation.

    Args:
        rho: Density matrix, shape (..., n, n)
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Expected number of electrons

    Returns:
        Absolute trace violation
    """
    trace = torch.einsum("...ij,ji->...", rho, overlap).real
    return (trace - n_electrons).abs()


def hermiticity_violation(rho: Tensor) -> Tensor:
    """
    Compute Hermiticity violation.

    Args:
        rho: Density matrix, shape (..., n, n)

    Returns:
        Maximum Hermiticity violation
    """
    rho_dag = rho.conj().transpose(-2, -1)
    return (rho - rho_dag).abs().amax(dim=(-2, -1))


def idempotency_violation(rho: Tensor, overlap: Tensor) -> Tensor:
    """
    Compute idempotency violation: ||ρSρ - ρ||.

    Args:
        rho: Density matrix, shape (..., n, n)
        overlap: Overlap matrix, shape (n, n)

    Returns:
        Idempotency violation
    """
    rho_S_rho = rho @ overlap @ rho
    return (rho_S_rho - rho).abs().amax(dim=(-2, -1))


def compute_absorption_spectrum(
    dipole_trajectory: Tensor,
    dt: float,
    window: str = "hann",
    padding_factor: int = 4,
) -> Tuple[Tensor, Tensor]:
    """
    Compute absorption spectrum from dipole moment trajectory.

    Args:
        dipole_trajectory: Dipole moments, shape (n_steps, 3)
        dt: Time step in atomic units
        window: Window function ("hann", "hamming", "none")
        padding_factor: Zero-padding factor for FFT

    Returns:
        Tuple of (frequencies, absorption_strength)
    """
    n_steps = dipole_trajectory.shape[0]

    # Apply window function
    if window == "hann":
        win = torch.hann_window(n_steps, device=dipole_trajectory.device)
    elif window == "hamming":
        win = torch.hamming_window(n_steps, device=dipole_trajectory.device)
    else:
        win = torch.ones(n_steps, device=dipole_trajectory.device)

    # Apply window to each component
    windowed = dipole_trajectory * win.unsqueeze(-1)

    # Zero-pad for better frequency resolution
    n_fft = n_steps * padding_factor
    padded = torch.zeros(n_fft, 3, device=dipole_trajectory.device)
    padded[:n_steps] = windowed

    # FFT
    spectrum = torch.fft.rfft(padded, dim=0)

    # Frequencies
    freqs = torch.fft.rfftfreq(n_fft, dt, device=dipole_trajectory.device)

    # Absorption strength (sum over components)
    absorption = (spectrum.abs().pow(2)).sum(dim=-1)

    # Convert to eV (1 a.u. = 27.2114 eV)
    freqs_ev = freqs * 27.2114

    return freqs_ev, absorption


def spectrum_overlap(
    spectrum_pred: Tensor,
    spectrum_true: Tensor,
    freqs: Tensor,
    freq_range: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Compute overlap between predicted and true absorption spectra.

    Args:
        spectrum_pred: Predicted spectrum
        spectrum_true: True spectrum
        freqs: Frequency values
        freq_range: Optional (min, max) frequency range in eV

    Returns:
        Overlap coefficient (0 to 1)
    """
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        spectrum_pred = spectrum_pred[mask]
        spectrum_true = spectrum_true[mask]

    # Normalize
    pred_norm = spectrum_pred / (spectrum_pred.sum() + 1e-10)
    true_norm = spectrum_true / (spectrum_true.sum() + 1e-10)

    # Compute overlap (Bhattacharyya coefficient)
    overlap = torch.sqrt(pred_norm * true_norm).sum()

    return overlap.item()


def compute_trajectory_metrics(
    trajectory_pred: Tensor,
    trajectory_true: Tensor,
    overlap: Tensor,
    n_electrons: int,
    dipole_integrals: Optional[Tensor] = None,
    dt: Optional[float] = None,
) -> TrajectoryMetrics:
    """
    Compute comprehensive metrics over a trajectory.

    Args:
        trajectory_pred: Predicted trajectory, shape (n_steps, n, n)
        trajectory_true: True trajectory, shape (n_steps, n, n)
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Number of electrons
        dipole_integrals: Optional dipole matrices for dipole error
        dt: Time step for spectrum computation

    Returns:
        TrajectoryMetrics dataclass
    """
    # Basic errors
    mse = (trajectory_pred - trajectory_true).abs().pow(2).mean().item()
    mae = (trajectory_pred - trajectory_true).abs().mean().item()
    rel_err = relative_error(trajectory_pred, trajectory_true).item()
    max_err = max_absolute_error(trajectory_pred, trajectory_true).item()

    # Physics violations (on predicted trajectory)
    trace_viol = trace_violation(trajectory_pred, overlap, n_electrons).mean().item()
    herm_viol = hermiticity_violation(trajectory_pred).mean().item()

    # Dipole error
    dip_err = None
    spec_overlap = None

    if dipole_integrals is not None:
        dip_err = dipole_error(trajectory_pred, trajectory_true, dipole_integrals).mean().item()

        if dt is not None:
            # Compute spectra
            dipole_pred = torch.einsum("tij,cji->tc", trajectory_pred, dipole_integrals).real
            dipole_true = torch.einsum("tij,cji->tc", trajectory_true, dipole_integrals).real

            freqs, spec_pred = compute_absorption_spectrum(dipole_pred, dt)
            _, spec_true = compute_absorption_spectrum(dipole_true, dt)

            spec_overlap = spectrum_overlap(spec_pred, spec_true, freqs)

    return TrajectoryMetrics(
        mse=mse,
        mae=mae,
        relative_error=rel_err,
        max_error=max_err,
        dipole_error=dip_err,
        trace_violation=trace_viol,
        hermiticity_violation=herm_viol,
        spectrum_overlap=spec_overlap,
    )


def compute_rollout_metrics(
    model,
    initial_density: Tensor,
    field_sequence: Tensor,
    geometry: Dict,
    ground_truth: Tensor,
    overlap: Tensor,
    n_electrons: int,
    dipole_integrals: Optional[Tensor] = None,
    dt: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute metrics for a model rollout vs ground truth.

    Args:
        model: Trained model
        initial_density: Starting density
        field_sequence: External field sequence
        geometry: Molecular geometry dict
        ground_truth: Ground truth trajectory
        overlap: Overlap matrix
        n_electrons: Number of electrons
        dipole_integrals: Optional dipole matrices
        dt: Time step

    Returns:
        Dictionary of metrics
    """
    from src.inference.predictor import Predictor, RolloutConfig

    # Create predictor and run rollout
    predictor = Predictor(model)
    config = RolloutConfig(
        max_steps=field_sequence.shape[0],
        apply_physics_projection=False,
    )

    result = predictor.rollout(
        initial_density=initial_density,
        geometry=geometry,
        field_sequence=field_sequence,
        overlap=overlap,
        n_electrons=n_electrons,
        config=config,
    )

    # Compute metrics
    trajectory_metrics = compute_trajectory_metrics(
        trajectory_pred=result.densities,
        trajectory_true=ground_truth,
        overlap=overlap,
        n_electrons=n_electrons,
        dipole_integrals=dipole_integrals,
        dt=dt,
    )

    return {
        'mse': trajectory_metrics.mse,
        'mae': trajectory_metrics.mae,
        'relative_error': trajectory_metrics.relative_error,
        'max_error': trajectory_metrics.max_error,
        'dipole_error': trajectory_metrics.dipole_error,
        'trace_violation': trajectory_metrics.trace_violation,
        'hermiticity_violation': trajectory_metrics.hermiticity_violation,
        'spectrum_overlap': trajectory_metrics.spectrum_overlap,
    }


def compute_step_errors(
    trajectory_pred: Tensor,
    trajectory_true: Tensor,
) -> Dict[str, Tensor]:
    """
    Compute per-step errors for error accumulation analysis.

    Args:
        trajectory_pred: Predicted trajectory
        trajectory_true: True trajectory

    Returns:
        Dict with per-step error tensors
    """
    n_steps = trajectory_pred.shape[0]

    step_mse = []
    step_rel_err = []
    cumulative_err = []

    for t in range(n_steps):
        mse_t = (trajectory_pred[t] - trajectory_true[t]).abs().pow(2).mean()
        rel_t = relative_error(trajectory_pred[t], trajectory_true[t])

        step_mse.append(mse_t)
        step_rel_err.append(rel_t)

        # Cumulative from start
        cum_err = relative_error(trajectory_pred[:t+1], trajectory_true[:t+1])
        cumulative_err.append(cum_err)

    return {
        'step_mse': torch.stack(step_mse),
        'step_relative_error': torch.stack(step_rel_err),
        'cumulative_error': torch.stack(cumulative_err),
    }


class MetricsAccumulator:
    """Accumulate metrics across multiple trajectories."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def add(self, metrics: Dict[str, float]):
        """Add metrics from one trajectory."""
        for key, value in metrics.items():
            if value is not None:
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)

    def compute_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics."""
        summary = {}
        for key, values in self.metrics.items():
            arr = np.array(values)
            summary[key] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'median': float(np.median(arr)),
            }
        return summary

    def __len__(self) -> int:
        if not self.metrics:
            return 0
        return len(next(iter(self.metrics.values())))
