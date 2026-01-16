"""
Tests for evaluation metrics module.

Tests cover:
- Basic error metrics (MSE, MAE, relative error)
- Physics constraint metrics (trace, Hermiticity, idempotency)
- Spectral analysis (absorption spectrum, overlap)
- Trajectory metrics computation
- Metrics accumulation
"""

import pytest
import torch
import numpy as np

from src.utils.metrics import (
    TrajectoryMetrics,
    frobenius_error,
    mean_absolute_error,
    max_absolute_error,
    relative_error,
    dipole_error,
    trace_violation,
    hermiticity_violation,
    idempotency_violation,
    compute_absorption_spectrum,
    spectrum_overlap,
    compute_trajectory_metrics,
    compute_step_errors,
    MetricsAccumulator,
)


class TestBasicErrorMetrics:
    """Tests for basic error metrics."""

    def test_frobenius_error_identical(self):
        """Frobenius error is zero for identical matrices."""
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        error = frobenius_error(rho, rho, normalize=False)
        assert error.item() < 1e-6

    def test_frobenius_error_normalized(self):
        """Normalized Frobenius error is relative."""
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64) + 1.0
        rho_noisy = rho + 0.1 * torch.randn(n, n, dtype=torch.complex64)

        error_unnorm = frobenius_error(rho_noisy, rho, normalize=False)
        error_norm = frobenius_error(rho_noisy, rho, normalize=True)

        # Normalized error should be smaller if norm > 1
        rho_norm = rho.abs().pow(2).sum().sqrt()
        if rho_norm > 1:
            assert error_norm < error_unnorm

    def test_frobenius_error_batched(self):
        """Frobenius error works with batched inputs."""
        batch = 5
        n = 4
        pred = torch.randn(batch, n, n, dtype=torch.complex64)
        true = torch.randn(batch, n, n, dtype=torch.complex64)

        error = frobenius_error(pred, true)
        assert error.shape == (batch,)

    def test_mean_absolute_error(self):
        """MAE computes correctly."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.0, 2.0, 4.0])
        mae = mean_absolute_error(pred, true)
        assert abs(mae.item() - 1.0 / 3) < 1e-6

    def test_max_absolute_error(self):
        """Max error computes correctly."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.0, 3.0, 3.0])
        max_err = max_absolute_error(pred, true)
        assert abs(max_err.item() - 1.0) < 1e-6

    def test_relative_error(self):
        """Relative error computes correctly."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.0, 2.0, 4.0])
        rel_err = relative_error(pred, true)
        expected = 1.0 / (1.0 + 2.0 + 4.0)
        assert abs(rel_err.item() - expected) < 1e-6


class TestPhysicsConstraintMetrics:
    """Tests for physics constraint metrics."""

    def test_trace_violation_conserved(self):
        """Trace violation is zero for properly normalized density."""
        n = 4
        n_electrons = 2

        # Create density with correct trace
        rho = torch.zeros(n, n, dtype=torch.complex64)
        rho[0, 0] = 1.0
        rho[1, 1] = 1.0

        overlap = torch.eye(n, dtype=torch.complex64)
        violation = trace_violation(rho, overlap, n_electrons)

        assert violation.item() < 1e-6

    def test_trace_violation_detected(self):
        """Trace violation is detected for wrong trace."""
        n = 4
        n_electrons = 2

        # Create density with wrong trace
        rho = torch.eye(n, dtype=torch.complex64)  # Trace = 4, not 2

        overlap = torch.eye(n, dtype=torch.complex64)
        violation = trace_violation(rho, overlap, n_electrons)

        assert abs(violation.item() - 2.0) < 1e-6

    def test_trace_violation_batched(self):
        """Trace violation works with batched inputs."""
        batch = 3
        n = 4
        n_electrons = 2

        rho = torch.randn(batch, n, n, dtype=torch.complex64)
        overlap = torch.eye(n, dtype=torch.complex64)
        violation = trace_violation(rho, overlap, n_electrons)

        assert violation.shape == (batch,)

    def test_hermiticity_violation_hermitian(self):
        """Hermiticity violation is zero for Hermitian matrix."""
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)  # Make Hermitian

        violation = hermiticity_violation(rho)
        assert violation.item() < 1e-6

    def test_hermiticity_violation_detected(self):
        """Hermiticity violation is detected for non-Hermitian matrix."""
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        # Add non-Hermitian part
        rho[0, 1] = 1.0 + 1.0j
        rho[1, 0] = 0.0  # Not conjugate

        violation = hermiticity_violation(rho)
        assert violation.item() > 0.5

    def test_hermiticity_violation_batched(self):
        """Hermiticity violation works with batched inputs."""
        batch = 3
        n = 4
        rho = torch.randn(batch, n, n, dtype=torch.complex64)

        violation = hermiticity_violation(rho)
        assert violation.shape == (batch,)

    def test_idempotency_violation_idempotent(self):
        """Idempotency violation is zero for projection matrix."""
        n = 4
        # Projection matrix is idempotent: P^2 = P
        rho = torch.zeros(n, n, dtype=torch.complex64)
        rho[0, 0] = 1.0
        rho[1, 1] = 1.0

        overlap = torch.eye(n, dtype=torch.complex64)
        violation = idempotency_violation(rho, overlap)

        assert violation.item() < 1e-6

    def test_idempotency_violation_detected(self):
        """Idempotency violation is detected for non-idempotent matrix."""
        n = 4
        # Not idempotent
        rho = torch.randn(n, n, dtype=torch.complex64)

        overlap = torch.eye(n, dtype=torch.complex64)
        violation = idempotency_violation(rho, overlap)

        assert violation.item() > 0


class TestDipoleError:
    """Tests for dipole moment error computation."""

    def test_dipole_error_identical(self):
        """Dipole error is zero for identical densities."""
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)

        dipole_integrals = torch.randn(3, n, n, dtype=torch.complex64)
        dipole_integrals = 0.5 * (dipole_integrals + dipole_integrals.conj().transpose(-2, -1))

        error = dipole_error(rho, rho, dipole_integrals)
        assert error.item() < 1e-6

    def test_dipole_error_batched(self):
        """Dipole error works with batched inputs."""
        batch = 5
        n = 4

        rho_pred = torch.randn(batch, n, n, dtype=torch.complex64)
        rho_true = torch.randn(batch, n, n, dtype=torch.complex64)
        dipole_integrals = torch.randn(3, n, n, dtype=torch.complex64)

        error = dipole_error(rho_pred, rho_true, dipole_integrals)
        assert error.shape == (batch,)

    def test_dipole_error_relative(self):
        """Relative dipole error scales correctly."""
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)

        # Small perturbation
        rho_perturbed = rho + 0.01 * torch.randn(n, n, dtype=torch.complex64)
        rho_perturbed = 0.5 * (rho_perturbed + rho_perturbed.conj().T)

        dipole_integrals = torch.randn(3, n, n, dtype=torch.complex64)

        error_rel = dipole_error(rho_perturbed, rho, dipole_integrals, relative=True)
        error_abs = dipole_error(rho_perturbed, rho, dipole_integrals, relative=False)

        # Relative error should exist
        assert error_rel.item() > 0
        assert error_abs.item() > 0


class TestSpectralAnalysis:
    """Tests for spectral analysis functions."""

    def test_compute_absorption_spectrum_output_shape(self):
        """Absorption spectrum has correct output shape."""
        n_steps = 100
        dt = 0.1
        dipole = torch.randn(n_steps, 3)

        freqs, absorption = compute_absorption_spectrum(dipole, dt)

        # FFT output size with padding
        expected_n_freq = (n_steps * 4) // 2 + 1
        assert freqs.shape[0] == expected_n_freq
        assert absorption.shape[0] == expected_n_freq

    def test_compute_absorption_spectrum_positive(self):
        """Absorption spectrum is non-negative."""
        n_steps = 100
        dt = 0.1
        dipole = torch.randn(n_steps, 3)

        _, absorption = compute_absorption_spectrum(dipole, dt)

        assert (absorption >= 0).all()

    def test_compute_absorption_spectrum_frequencies(self):
        """Frequencies are in correct range."""
        n_steps = 100
        dt = 0.1  # atomic units
        dipole = torch.randn(n_steps, 3)

        freqs, _ = compute_absorption_spectrum(dipole, dt)

        # Should be in eV
        assert freqs[0].item() == 0.0
        assert freqs[-1].item() > 0

    def test_spectrum_overlap_identical(self):
        """Spectrum overlap is 1 for identical spectra."""
        n_freq = 50
        spectrum = torch.rand(n_freq).abs() + 0.1
        freqs = torch.linspace(0, 10, n_freq)

        overlap = spectrum_overlap(spectrum, spectrum, freqs)
        assert abs(overlap - 1.0) < 1e-6

    def test_spectrum_overlap_range(self):
        """Spectrum overlap with frequency range filtering."""
        n_freq = 50
        spectrum1 = torch.rand(n_freq).abs() + 0.1
        spectrum2 = torch.rand(n_freq).abs() + 0.1
        freqs = torch.linspace(0, 10, n_freq)

        overlap = spectrum_overlap(spectrum1, spectrum2, freqs, freq_range=(2.0, 8.0))

        # Should be between 0 and 1
        assert 0 <= overlap <= 1.0


class TestTrajectoryMetrics:
    """Tests for trajectory-level metrics computation."""

    @pytest.fixture
    def sample_trajectory(self):
        """Create sample trajectory data."""
        n_steps = 20
        n = 4
        n_electrons = 2

        # Create Hermitian densities with correct trace
        trajectory = []
        for _ in range(n_steps):
            rho = torch.randn(n, n, dtype=torch.complex64)
            rho = 0.5 * (rho + rho.conj().T)
            # Normalize trace
            trace = rho.diagonal().sum().real
            rho = rho * (n_electrons / trace)
            trajectory.append(rho)

        trajectory = torch.stack(trajectory)
        overlap = torch.eye(n, dtype=torch.complex64)

        return trajectory, overlap, n_electrons

    def test_compute_trajectory_metrics_output(self, sample_trajectory):
        """Trajectory metrics returns TrajectoryMetrics dataclass."""
        trajectory, overlap, n_electrons = sample_trajectory
        # Create slightly different prediction
        pred = trajectory + 0.01 * torch.randn_like(trajectory)

        metrics = compute_trajectory_metrics(
            trajectory_pred=pred,
            trajectory_true=trajectory,
            overlap=overlap,
            n_electrons=n_electrons,
        )

        assert isinstance(metrics, TrajectoryMetrics)
        assert metrics.mse > 0
        assert metrics.mae > 0
        assert metrics.relative_error > 0

    def test_compute_trajectory_metrics_with_dipole(self, sample_trajectory):
        """Trajectory metrics includes dipole error when integrals provided."""
        trajectory, overlap, n_electrons = sample_trajectory
        n = trajectory.shape[1]
        pred = trajectory + 0.01 * torch.randn_like(trajectory)

        dipole_integrals = torch.randn(3, n, n, dtype=torch.complex64)

        metrics = compute_trajectory_metrics(
            trajectory_pred=pred,
            trajectory_true=trajectory,
            overlap=overlap,
            n_electrons=n_electrons,
            dipole_integrals=dipole_integrals,
        )

        assert metrics.dipole_error is not None

    def test_compute_step_errors_output(self, sample_trajectory):
        """Step errors returns per-step error tensors."""
        trajectory, _, _ = sample_trajectory
        pred = trajectory + 0.01 * torch.randn_like(trajectory)

        step_errors = compute_step_errors(pred, trajectory)

        assert "step_mse" in step_errors
        assert "step_relative_error" in step_errors
        assert "cumulative_error" in step_errors
        assert step_errors["step_mse"].shape[0] == len(trajectory)

    def test_compute_step_errors_accumulation(self, sample_trajectory):
        """Cumulative error generally increases over steps."""
        trajectory, _, _ = sample_trajectory
        # Add growing noise to simulate error accumulation
        noise_scale = torch.linspace(0, 0.1, len(trajectory))
        pred = trajectory + noise_scale.view(-1, 1, 1) * torch.randn_like(trajectory)

        step_errors = compute_step_errors(pred, trajectory)

        # Later steps should generally have higher cumulative error
        cumulative = step_errors["cumulative_error"]
        assert cumulative[-1] >= cumulative[0]


class TestMetricsAccumulator:
    """Tests for MetricsAccumulator class."""

    def test_accumulator_empty(self):
        """Empty accumulator has length 0."""
        acc = MetricsAccumulator()
        assert len(acc) == 0

    def test_accumulator_add(self):
        """Accumulator stores metrics correctly."""
        acc = MetricsAccumulator()

        acc.add({"mse": 0.1, "mae": 0.05})
        acc.add({"mse": 0.2, "mae": 0.08})

        assert len(acc) == 2
        assert len(acc.metrics["mse"]) == 2
        assert len(acc.metrics["mae"]) == 2

    def test_accumulator_skips_none(self):
        """Accumulator skips None values."""
        acc = MetricsAccumulator()

        acc.add({"mse": 0.1, "dipole": None})
        acc.add({"mse": 0.2, "dipole": 0.01})

        assert len(acc.metrics["mse"]) == 2
        assert len(acc.metrics["dipole"]) == 1

    def test_accumulator_summary(self):
        """Summary computes correct statistics."""
        acc = MetricsAccumulator()

        acc.add({"mse": 1.0})
        acc.add({"mse": 2.0})
        acc.add({"mse": 3.0})

        summary = acc.compute_summary()

        assert "mse" in summary
        assert abs(summary["mse"]["mean"] - 2.0) < 1e-6
        assert abs(summary["mse"]["min"] - 1.0) < 1e-6
        assert abs(summary["mse"]["max"] - 3.0) < 1e-6
        assert abs(summary["mse"]["median"] - 2.0) < 1e-6

    def test_accumulator_std(self):
        """Summary computes standard deviation correctly."""
        acc = MetricsAccumulator()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            acc.add({"metric": v})

        summary = acc.compute_summary()

        expected_std = np.std(values)
        assert abs(summary["metric"]["std"] - expected_std) < 1e-6


class TestTrajectoryMetricsDataclass:
    """Tests for TrajectoryMetrics dataclass."""

    def test_dataclass_creation(self):
        """TrajectoryMetrics can be created with required fields."""
        metrics = TrajectoryMetrics(
            mse=0.1,
            mae=0.05,
            relative_error=0.02,
            max_error=0.2,
        )

        assert metrics.mse == 0.1
        assert metrics.dipole_error is None

    def test_dataclass_optional_fields(self):
        """TrajectoryMetrics optional fields work correctly."""
        metrics = TrajectoryMetrics(
            mse=0.1,
            mae=0.05,
            relative_error=0.02,
            max_error=0.2,
            dipole_error=0.03,
            spectrum_overlap=0.95,
        )

        assert metrics.dipole_error == 0.03
        assert metrics.spectrum_overlap == 0.95
