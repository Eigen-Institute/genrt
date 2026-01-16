"""
Physics constraint checking functions for density matrices.

These functions check whether density matrices satisfy the required
physics constraints without modifying them. For constraint enforcement,
see projections.py.
"""

import torch
from torch import Tensor
from typing import Tuple, Dict


def check_hermiticity(rho: Tensor, atol: float = 1e-6) -> Tuple[bool, float]:
    """
    Check if density matrix is Hermitian: ρ = ρ†.

    Args:
        rho: Density matrix, shape (..., n, n) complex
        atol: Absolute tolerance

    Returns:
        Tuple of (is_hermitian, max_error)
    """
    rho_dagger = rho.conj().transpose(-2, -1)
    error = (rho - rho_dagger).abs().max().item()
    return error < atol, error


def check_trace(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
    atol: float = 1e-4,
) -> Tuple[bool, float]:
    """
    Check if density matrix has correct trace: Tr(ρS) = n_electrons.

    Args:
        rho: Density matrix, shape (..., n, n) complex
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Expected number of electrons
        atol: Absolute tolerance

    Returns:
        Tuple of (is_correct, trace_error)
    """
    # Compute trace for each batch element
    trace = torch.einsum("...ij,ji->...", rho, overlap).real

    # Handle batched case
    if trace.dim() == 0:
        error = abs(trace.item() - n_electrons)
    else:
        error = (trace - n_electrons).abs().max().item()

    return error < atol, error


def check_cauchy_schwarz(
    rho: Tensor,
    atol: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Fast O(n²) check using Cauchy-Schwarz inequality.

    For a PSD matrix: |ρ_ij|² ≤ ρ_ii · ρ_jj for all i,j

    This is a NECESSARY but NOT SUFFICIENT condition for PSD.
    If this fails, the matrix is definitely not PSD.
    If this passes, the matrix might still not be PSD.

    Use this for fast screening during training; use check_positive_semidefinite
    for rigorous verification.

    Args:
        rho: Density matrix, shape (n, n) complex
        atol: Absolute tolerance

    Returns:
        Tuple of (satisfies_cs, max_violation)
        where max_violation = max(|ρ_ij|² - ρ_ii·ρ_jj) over all i,j
    """
    # Get diagonal elements (should be real for Hermitian matrix)
    diag = rho.diagonal().real

    # Check diagonal is non-negative (necessary for PSD)
    if (diag < -atol).any():
        min_diag = diag.min().item()
        return False, -min_diag

    # Compute |ρ_ij|² for all pairs
    rho_sq = (rho * rho.conj()).real  # |ρ_ij|²

    # Compute ρ_ii · ρ_jj for all pairs
    diag_product = diag.unsqueeze(0) * diag.unsqueeze(1)  # outer product

    # Cauchy-Schwarz violation: |ρ_ij|² - ρ_ii·ρ_jj > 0 means violation
    violation = rho_sq - diag_product
    max_violation = violation.max().item()

    return max_violation < atol, max(0.0, max_violation)


def check_positive_semidefinite(
    rho: Tensor,
    overlap: Tensor,
    atol: float = 1e-6,
    fast: bool = False,
) -> Tuple[bool, float]:
    """
    Check if density matrix is positive semi-definite.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        atol: Absolute tolerance for negative eigenvalues
        fast: If True, use O(n²) Cauchy-Schwarz check (necessary but not sufficient).
              If False, use O(n³) eigenvalue check (necessary and sufficient).

    Returns:
        Tuple of (is_psd, error_metric)
        For fast=True: error is max Cauchy-Schwarz violation
        For fast=False: error is most negative eigenvalue
    """
    if fast:
        # Transform to orthogonal basis first for proper check
        n = overlap.shape[0]
        device = overlap.device
        S_real = overlap.real if overlap.is_complex() else overlap
        L = torch.linalg.cholesky(S_real + 1e-6 * torch.eye(n, device=device))
        L_inv = torch.linalg.inv(L)
        if rho.is_complex():
            L_inv = L_inv.to(rho.dtype)
        rho_orth = L_inv @ rho @ L_inv.T.conj()
        return check_cauchy_schwarz(rho_orth, atol)

    # Full eigenvalue check - O(n³)
    n = overlap.shape[0]
    device = overlap.device

    # Ensure overlap is real for Cholesky
    S_real = overlap.real if overlap.is_complex() else overlap

    # First transform to orthogonal basis: S = L @ L^T
    L = torch.linalg.cholesky(S_real + 1e-6 * torch.eye(n, device=device))
    L_inv = torch.linalg.inv(L)

    # Convert L_inv to complex if rho is complex
    if rho.is_complex():
        L_inv = L_inv.to(rho.dtype)

    # Transform density: ρ' = L^{-1} @ ρ @ L^{-T}
    rho_transformed = L_inv @ rho @ L_inv.T.conj()

    # Get eigenvalues
    eigenvalues = torch.linalg.eigvalsh(rho_transformed).real

    min_eig = eigenvalues.min().item()
    return min_eig > -atol, min_eig


def check_idempotency(
    rho: Tensor,
    overlap: Tensor,
    atol: float = 1e-4,
) -> Tuple[bool, float]:
    """
    Check if density matrix is idempotent: ρSρ = ρ (for closed-shell).

    This is the condition for a pure-state density matrix in
    non-orthogonal basis.

    Args:
        rho: Density matrix, shape (..., n, n) complex
        overlap: Overlap matrix, shape (n, n)
        atol: Absolute tolerance

    Returns:
        Tuple of (is_idempotent, max_error)
    """
    rho_S_rho = rho @ overlap @ rho
    error = (rho_S_rho - rho).abs().max().item()
    return error < atol, error


def check_positive_occupations(
    rho: Tensor,
    overlap: Tensor,
    atol: float = 1e-4,
) -> Tuple[bool, float, float]:
    """
    Check if natural orbital occupations are non-negative.

    For AO basis density matrices, the constraint is that eigenvalues
    (natural orbital occupations) must be >= 0. They sum to N_electrons
    (enforced by trace constraint). Unlike MO basis, there is no upper
    bound of 2 per orbital in the AO representation.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        atol: Absolute tolerance for negativity

    Returns:
        Tuple of (all_non_negative, min_occupation, max_occupation)
    """
    # Diagonalize in orthogonal basis to get natural orbital occupations
    device = rho.device
    n = rho.shape[0]

    S_real = overlap.real if overlap.is_complex() else overlap
    L = torch.linalg.cholesky(S_real + 1e-6 * torch.eye(n, device=device))
    L_inv = torch.linalg.inv(L)

    # Ensure compatible dtypes
    if rho.is_complex():
        L_inv = L_inv.to(rho.dtype)

    rho_orth = L_inv @ rho @ L_inv.T.conj()
    occupations = torch.linalg.eigvalsh(rho_orth).real

    min_occ = occupations.min().item()
    max_occ = occupations.max().item()

    # Only check non-negativity - no upper bound for AO basis
    all_non_negative = min_occ > -atol
    return all_non_negative, min_occ, max_occ


# Keep old name as alias for backwards compatibility
def check_occupation_bounds(
    rho: Tensor,
    overlap: Tensor,
    max_occupation: float = 2.0,  # noqa: ARG001 - Ignored, kept for API compatibility
    atol: float = 1e-4,
) -> Tuple[bool, float, float]:
    """Deprecated: Use check_positive_occupations instead."""
    del max_occupation  # Unused, AO basis has no per-orbital upper bound
    return check_positive_occupations(rho, overlap, atol)


def check_all_constraints(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
) -> Dict[str, Tuple[bool, float]]:
    """
    Check all physics constraints on a density matrix.

    For AO basis density matrices, the constraints are:
    1. Hermiticity: ρ = ρ†
    2. Trace: Tr(ρS) = N_electrons
    3. Positive semi-definite: all eigenvalues >= 0
    4. Idempotency: ρSρ = ρ (soft constraint, only exact at equilibrium)

    Note: Unlike MO basis, AO basis does NOT have a per-orbital [0,2] bound.
    The only bounds are non-negativity and sum = N_electrons.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Number of electrons

    Returns:
        Dictionary with constraint names and (passed, error) tuples
    """
    results = {}

    # Hermiticity
    results['hermitian'] = check_hermiticity(rho)

    # Trace
    results['trace'] = check_trace(rho, overlap, n_electrons)

    # Positive semi-definite (eigenvalues >= 0)
    results['positive_semidefinite'] = check_positive_semidefinite(rho, overlap)

    # Idempotency (soft constraint)
    results['idempotent'] = check_idempotency(rho, overlap)

    # Non-negative occupations (no upper bound for AO basis)
    is_non_neg, min_occ, _ = check_positive_occupations(rho, overlap)
    # Error is how negative the most negative eigenvalue is
    results['positive_occupations'] = (is_non_neg, -min_occ if min_occ < 0 else 0.0)

    return results


def constraint_violation_loss(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
) -> Dict[str, Tensor]:
    """
    Compute differentiable constraint violations for loss function.

    Args:
        rho: Density matrix, shape (..., n, n) complex
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Number of electrons

    Returns:
        Dictionary of constraint violation tensors
    """
    losses = {}

    # Hermiticity violation
    rho_dagger = rho.conj().transpose(-2, -1)
    losses['hermitian'] = (rho - rho_dagger).abs().pow(2).mean()

    # Trace violation
    trace = torch.einsum("...ij,ji->...", rho, overlap).real
    losses['trace'] = (trace - n_electrons).pow(2).mean()

    # Idempotency violation
    rho_S_rho = rho @ overlap @ rho
    losses['idempotent'] = (rho_S_rho - rho).abs().pow(2).mean()

    return losses
