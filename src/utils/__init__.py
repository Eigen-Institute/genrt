"""
Utility functions for the RT-TDDFT ML accelerator.

This module provides:
- Complex tensor operations
- Molecular graph construction
- Visualization utilities
- Evaluation metrics
"""

from .complex_tensor import (
    real_to_complex,
    complex_to_real,
    hermitianize,
    trace_normalize,
    check_hermiticity,
    check_trace,
    check_idempotency,
    mcweeney_purification,
    density_eigenvalues,
)
from .graph import (
    build_molecular_graph,
    compute_edge_vectors,
    MolecularGraph,
    compute_edge_spherical,
    radial_basis_bessel,
    smooth_cutoff,
    get_edge_attributes,
    batch_graphs,
)
from .metrics import (
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
    compute_rollout_metrics,
    compute_step_errors,
    MetricsAccumulator,
)
from .visualization import (
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

__all__ = [
    # Complex tensor utilities
    "real_to_complex",
    "complex_to_real",
    "hermitianize",
    "trace_normalize",
    "check_hermiticity",
    "check_trace",
    "check_idempotency",
    "mcweeney_purification",
    "density_eigenvalues",
    # Graph utilities
    "build_molecular_graph",
    "compute_edge_vectors",
    "MolecularGraph",
    "compute_edge_spherical",
    "radial_basis_bessel",
    "smooth_cutoff",
    "get_edge_attributes",
    "batch_graphs",
    # Metrics
    "TrajectoryMetrics",
    "frobenius_error",
    "mean_absolute_error",
    "max_absolute_error",
    "relative_error",
    "dipole_error",
    "trace_violation",
    "hermiticity_violation",
    "idempotency_violation",
    "compute_absorption_spectrum",
    "spectrum_overlap",
    "compute_trajectory_metrics",
    "compute_rollout_metrics",
    "compute_step_errors",
    "MetricsAccumulator",
    # Visualization
    "PlotStyle",
    "DEFAULT_STYLE",
    "setup_style",
    "plot_training_curves",
    "plot_loss_components",
    "plot_learning_rate",
    "plot_error_accumulation",
    "plot_physics_violations",
    "plot_absorption_spectrum",
    "plot_density_matrix",
    "plot_density_comparison",
    "plot_dipole_trajectory",
    "plot_metrics_comparison",
    "plot_curriculum_progress",
    "create_training_dashboard",
    "save_figure",
    "close_all",
]
