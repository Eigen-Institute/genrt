"""
Physics constraints and projections for density matrices.

This module provides:
- Constraint checking (Hermiticity, trace, idempotency)
- Constraint projections (Hermitianize, trace normalize, McWeeney)
- Observable calculations (dipole moment, populations, bond orders)
"""

from .constraints import (
    check_hermiticity,
    check_trace,
    check_cauchy_schwarz,
    check_positive_semidefinite,
    check_idempotency,
    check_positive_occupations,
    check_occupation_bounds,  # Deprecated alias
    check_all_constraints,
    constraint_violation_loss,
)
from .projections import (
    hermitianize,
    trace_normalize,
    mcweeney_purification,
    diis_purification,
    project_positive_semidefinite,
    PhysicsProjection,
    SoftConstraintProjection,
)
from .observables import (
    compute_dipole_moment,
    compute_mulliken_populations,
    compute_lowdin_populations,
    compute_natural_orbital_occupations,
    compute_energy_components,
    compute_bond_order,
    compute_electron_density_at_point,
    compute_current_density,
    ObservableCalculator,
)

__all__ = [
    # Constraint checking
    "check_hermiticity",
    "check_trace",
    "check_cauchy_schwarz",
    "check_positive_semidefinite",
    "check_idempotency",
    "check_positive_occupations",
    "check_occupation_bounds",  # Deprecated alias
    "check_all_constraints",
    "constraint_violation_loss",
    # Projections
    "hermitianize",
    "trace_normalize",
    "mcweeney_purification",
    "diis_purification",
    "project_positive_semidefinite",
    "PhysicsProjection",
    "SoftConstraintProjection",
    # Observables
    "compute_dipole_moment",
    "compute_mulliken_populations",
    "compute_lowdin_populations",
    "compute_natural_orbital_occupations",
    "compute_energy_components",
    "compute_bond_order",
    "compute_electron_density_at_point",
    "compute_current_density",
    "ObservableCalculator",
]
