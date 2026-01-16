"""
Tests for visualization utilities.

Tests cover:
- Training curve plots
- Error accumulation plots
- Physics violation plots
- Spectrum and density visualizations
- Dashboard creation
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.visualization import (
    PlotStyle,
    DEFAULT_STYLE,
    setup_style,
    plot_training_curves,
    plot_loss_components,
    plot_learning_rate,
    plot_error_accumulation,
    plot_physics_violations,
    plot_absorption_spectrum,
    plot_density_matrix,
    plot_density_comparison,
    plot_dipole_trajectory,
    plot_metrics_comparison,
    plot_curriculum_progress,
    create_training_dashboard,
    save_figure,
    close_all,
)


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Close all plots after each test."""
    yield
    close_all()


class TestPlotStyle:
    """Tests for plot styling."""

    def test_default_style(self):
        """Default style has expected values."""
        assert DEFAULT_STYLE.figsize == (10, 6)
        assert DEFAULT_STYLE.dpi == 100
        assert DEFAULT_STYLE.grid is True

    def test_custom_style(self):
        """Custom style can be created."""
        style = PlotStyle(figsize=(8, 4), dpi=150, grid=False)
        assert style.figsize == (8, 4)
        assert style.dpi == 150
        assert style.grid is False

    def test_setup_style(self):
        """Setup style applies to matplotlib."""
        style = PlotStyle(font_size=14)
        setup_style(style)
        assert plt.rcParams['font.size'] == 14


class TestTrainingCurvePlots:
    """Tests for training curve visualizations."""

    def test_plot_training_curves_basic(self):
        """Basic training curve plot works."""
        train_losses = [1.0, 0.5, 0.3, 0.2, 0.15]
        fig = plot_training_curves(train_losses)

        assert fig is not None
        assert len(fig.axes) == 1

    def test_plot_training_curves_with_validation(self):
        """Training curves with validation data."""
        train_losses = [1.0, 0.5, 0.3, 0.2, 0.15]
        val_losses = [1.1, 0.6, 0.35, 0.25, 0.18]

        fig = plot_training_curves(train_losses, val_losses)

        assert fig is not None
        # Check that legend has both lines
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 2

    def test_plot_training_curves_linear_scale(self):
        """Training curves with linear y-scale."""
        train_losses = [1.0, 0.5, 0.3, 0.2, 0.15]
        fig = plot_training_curves(train_losses, log_scale=False)

        ax = fig.axes[0]
        assert ax.get_yscale() == 'linear'

    def test_plot_training_curves_save(self):
        """Training curves can be saved."""
        train_losses = [1.0, 0.5, 0.3]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_curves.png"
            fig = plot_training_curves(train_losses, save_path=str(path))

            assert path.exists()
            assert path.stat().st_size > 0

    def test_plot_loss_components(self):
        """Loss components plot works."""
        loss_history = {
            'reconstruction': [1.0, 0.5, 0.3],
            'gradient': [0.5, 0.3, 0.2],
            'trace': [0.1, 0.08, 0.05],
        }

        fig = plot_loss_components(loss_history)

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 3

    def test_plot_learning_rate(self):
        """Learning rate plot works."""
        lr_history = [1e-3, 1e-3, 5e-4, 2e-4, 1e-4]
        fig = plot_learning_rate(lr_history)

        assert fig is not None


class TestErrorPlots:
    """Tests for error visualization."""

    def test_plot_error_accumulation_basic(self):
        """Error accumulation plot works."""
        step_errors = [0.01, 0.02, 0.03, 0.05, 0.08]
        fig = plot_error_accumulation(step_errors)

        assert fig is not None

    def test_plot_error_accumulation_with_cumulative(self):
        """Error accumulation with cumulative errors."""
        step_errors = [0.01, 0.02, 0.03, 0.05, 0.08]
        cumulative_errors = [0.01, 0.03, 0.06, 0.11, 0.19]

        fig = plot_error_accumulation(step_errors, cumulative_errors)

        assert fig is not None
        # Should have two y-axes
        assert len(fig.axes) == 2

    def test_plot_error_accumulation_with_time(self):
        """Error accumulation with time axis."""
        step_errors = [0.01, 0.02, 0.03, 0.05, 0.08]
        fig = plot_error_accumulation(step_errors, dt=0.1)

        ax = fig.axes[0]
        assert "Time" in ax.get_xlabel()

    def test_plot_error_accumulation_tensor_input(self):
        """Error accumulation accepts tensor input."""
        step_errors = torch.tensor([0.01, 0.02, 0.03, 0.05, 0.08])
        fig = plot_error_accumulation(step_errors)

        assert fig is not None


class TestPhysicsViolationPlots:
    """Tests for physics violation visualizations."""

    def test_plot_physics_violations_trace_only(self):
        """Physics violations plot with trace only."""
        trace_violations = [0.01, 0.02, 0.015, 0.018, 0.02]
        fig = plot_physics_violations(trace_violations)

        assert fig is not None

    def test_plot_physics_violations_all(self):
        """Physics violations plot with all metrics."""
        n_steps = 10
        trace = np.random.uniform(0.01, 0.03, n_steps)
        herm = np.random.uniform(1e-6, 1e-5, n_steps)
        idem = np.random.uniform(0.1, 0.2, n_steps)

        fig = plot_physics_violations(
            trace_violations=trace,
            hermiticity_violations=herm,
            idempotency_violations=idem,
        )

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 3

    def test_plot_physics_violations_linear_scale(self):
        """Physics violations with linear scale."""
        trace_violations = [0.01, 0.02, 0.015, 0.018, 0.02]
        fig = plot_physics_violations(trace_violations, log_scale=False)

        ax = fig.axes[0]
        assert ax.get_yscale() == 'linear'


class TestSpectrumPlots:
    """Tests for spectrum visualizations."""

    def test_plot_absorption_spectrum_basic(self):
        """Absorption spectrum plot works."""
        n_freq = 100
        freqs = np.linspace(0, 10, n_freq)
        spectrum = np.exp(-((freqs - 5) ** 2) / 2)

        fig = plot_absorption_spectrum(freqs, spectrum)

        assert fig is not None

    def test_plot_absorption_spectrum_comparison(self):
        """Absorption spectrum comparison works."""
        n_freq = 100
        freqs = np.linspace(0, 10, n_freq)
        spectrum_pred = np.exp(-((freqs - 5) ** 2) / 2)
        spectrum_true = np.exp(-((freqs - 5.2) ** 2) / 2)

        fig = plot_absorption_spectrum(freqs, spectrum_pred, spectrum_true)

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 2

    def test_plot_absorption_spectrum_freq_range(self):
        """Absorption spectrum with frequency range filter."""
        n_freq = 100
        freqs = np.linspace(0, 10, n_freq)
        spectrum = np.exp(-((freqs - 5) ** 2) / 2)

        fig = plot_absorption_spectrum(freqs, spectrum, freq_range=(3, 7))

        assert fig is not None

    def test_plot_absorption_spectrum_tensor_input(self):
        """Absorption spectrum accepts tensor input."""
        n_freq = 100
        freqs = torch.linspace(0, 10, n_freq)
        spectrum = torch.exp(-((freqs - 5) ** 2) / 2)

        fig = plot_absorption_spectrum(freqs, spectrum)

        assert fig is not None


class TestDensityMatrixPlots:
    """Tests for density matrix visualizations."""

    def test_plot_density_matrix_real(self):
        """Density matrix real part plot works."""
        n = 8
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)

        fig = plot_density_matrix(rho, component="real")

        assert fig is not None

    def test_plot_density_matrix_imag(self):
        """Density matrix imaginary part plot works."""
        n = 8
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)

        fig = plot_density_matrix(rho, component="imag")

        assert fig is not None

    def test_plot_density_matrix_abs(self):
        """Density matrix magnitude plot works."""
        n = 8
        rho = torch.randn(n, n, dtype=torch.complex64)

        fig = plot_density_matrix(rho, component="abs")

        assert fig is not None

    def test_plot_density_matrix_phase(self):
        """Density matrix phase plot works."""
        n = 8
        rho = torch.randn(n, n, dtype=torch.complex64)

        fig = plot_density_matrix(rho, component="phase")

        assert fig is not None

    def test_plot_density_matrix_numpy(self):
        """Density matrix plot accepts numpy input."""
        n = 8
        rho = np.random.randn(n, n) + 1j * np.random.randn(n, n)

        fig = plot_density_matrix(rho)

        assert fig is not None

    def test_plot_density_comparison(self):
        """Density comparison plot works."""
        n = 8
        rho_pred = torch.randn(n, n, dtype=torch.complex64)
        rho_true = torch.randn(n, n, dtype=torch.complex64)

        fig = plot_density_comparison(rho_pred, rho_true)

        assert fig is not None
        # Should have 3 subplots (pred, true, diff)
        assert len(fig.axes) >= 3


class TestDipolePlots:
    """Tests for dipole trajectory visualizations."""

    def test_plot_dipole_trajectory_basic(self):
        """Dipole trajectory plot works."""
        n_steps = 50
        dipole = np.random.randn(n_steps, 3) * 0.1

        fig = plot_dipole_trajectory(dipole)

        assert fig is not None

    def test_plot_dipole_trajectory_comparison(self):
        """Dipole trajectory comparison works."""
        n_steps = 50
        dipole_pred = np.random.randn(n_steps, 3) * 0.1
        dipole_true = dipole_pred + np.random.randn(n_steps, 3) * 0.01

        fig = plot_dipole_trajectory(dipole_pred, dipole_true)

        assert fig is not None
        ax = fig.axes[0]
        # Should have 6 lines (3 components x 2 trajectories)
        assert len(ax.get_lines()) == 6

    def test_plot_dipole_trajectory_single_component(self):
        """Dipole trajectory with single component."""
        n_steps = 50
        dipole = np.random.randn(n_steps, 3) * 0.1

        fig = plot_dipole_trajectory(dipole, components=["z"])

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1

    def test_plot_dipole_trajectory_with_time(self):
        """Dipole trajectory with time axis."""
        n_steps = 50
        dipole = np.random.randn(n_steps, 3) * 0.1

        fig = plot_dipole_trajectory(dipole, dt=0.1)

        ax = fig.axes[0]
        assert "Time" in ax.get_xlabel()


class TestMetricsComparisonPlots:
    """Tests for metrics comparison visualizations."""

    def test_plot_metrics_comparison(self):
        """Metrics comparison bar chart works."""
        metrics_dict = {
            'Model A': {'mse': 0.01, 'mae': 0.05, 'dipole_error': 0.02},
            'Model B': {'mse': 0.02, 'mae': 0.08, 'dipole_error': 0.03},
        }

        fig = plot_metrics_comparison(metrics_dict)

        assert fig is not None

    def test_plot_metrics_comparison_selected(self):
        """Metrics comparison with selected metrics."""
        metrics_dict = {
            'Model A': {'mse': 0.01, 'mae': 0.05, 'dipole_error': 0.02},
            'Model B': {'mse': 0.02, 'mae': 0.08, 'dipole_error': 0.03},
        }

        fig = plot_metrics_comparison(metrics_dict, metric_names=['mse', 'mae'])

        assert fig is not None


class TestCurriculumPlots:
    """Tests for curriculum progress visualizations."""

    def test_plot_curriculum_progress(self):
        """Curriculum progress plot works."""
        stage_epochs = [0, 25, 50, 75]
        stage_horizons = [16, 32, 48, 64]
        current_epoch = 40

        fig = plot_curriculum_progress(stage_epochs, stage_horizons, current_epoch)

        assert fig is not None

    def test_plot_curriculum_progress_early(self):
        """Curriculum progress with early epoch."""
        stage_epochs = [0, 25, 50, 75]
        stage_horizons = [16, 32, 48, 64]
        current_epoch = 10

        fig = plot_curriculum_progress(stage_epochs, stage_horizons, current_epoch)

        assert fig is not None

    def test_plot_curriculum_progress_late(self):
        """Curriculum progress with late epoch."""
        stage_epochs = [0, 25, 50, 75]
        stage_horizons = [16, 32, 48, 64]
        current_epoch = 100

        fig = plot_curriculum_progress(stage_epochs, stage_horizons, current_epoch)

        assert fig is not None


class TestDashboard:
    """Tests for training dashboard."""

    def test_create_training_dashboard_basic(self):
        """Basic dashboard creation works."""
        train_losses = [1.0, 0.5, 0.3, 0.2, 0.15]

        fig = create_training_dashboard(train_losses)

        assert fig is not None

    def test_create_training_dashboard_full(self):
        """Full dashboard with all components."""
        train_losses = [1.0, 0.5, 0.3, 0.2, 0.15]
        val_losses = [1.1, 0.6, 0.35, 0.25, 0.18]
        loss_components = {
            'recon': [0.8, 0.4, 0.2, 0.15, 0.1],
            'grad': [0.2, 0.1, 0.1, 0.05, 0.05],
        }
        lr_history = [1e-3, 1e-3, 5e-4, 2e-4, 1e-4]
        physics_violations = {
            'trace': [0.1, 0.08, 0.05, 0.03, 0.02],
            'herm': [1e-5, 1e-5, 1e-6, 1e-6, 1e-7],
        }

        fig = create_training_dashboard(
            train_losses=train_losses,
            val_losses=val_losses,
            loss_components=loss_components,
            lr_history=lr_history,
            physics_violations=physics_violations,
        )

        assert fig is not None
        # Should have 4 subplots
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 4

    def test_create_training_dashboard_save(self):
        """Dashboard can be saved."""
        train_losses = [1.0, 0.5, 0.3]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dashboard.png"
            fig = create_training_dashboard(train_losses, save_path=str(path))

            assert path.exists()


class TestSaveFigure:
    """Tests for figure saving utilities."""

    def test_save_figure_png(self):
        """Save figure as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            save_figure(fig, str(path), formats=["png"])

            assert (path.parent / "test.png").exists()

    def test_save_figure_multiple_formats(self):
        """Save figure in multiple formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            save_figure(fig, str(path), formats=["png", "pdf"])

            assert (path.parent / "test.png").exists()
            assert (path.parent / "test.pdf").exists()

    def test_save_figure_creates_directory(self):
        """Save figure creates parent directories."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "test"
            save_figure(fig, str(path), formats=["png"])

            assert (path.parent / "test.png").exists()


class TestCloseAll:
    """Tests for figure cleanup."""

    def test_close_all(self):
        """Close all figures works."""
        # Create some figures
        for _ in range(5):
            plt.figure()

        assert len(plt.get_fignums()) == 5

        close_all()

        assert len(plt.get_fignums()) == 0
