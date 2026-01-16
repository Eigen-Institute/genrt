"""
Complex tensor utilities for density matrix operations.

Density matrices in RT-TDDFT are complex Hermitian matrices. This module provides
utilities for converting between real/complex representations and enforcing
physical constraints.
"""

import torch
from torch import Tensor


def real_to_complex(real_part: Tensor, imag_part: Tensor) -> Tensor:
    """
    Combine real and imaginary parts into a complex tensor.

    Args:
        real_part: Real component tensor
        imag_part: Imaginary component tensor

    Returns:
        Complex tensor with dtype torch.complex64 or torch.complex128
    """
    if real_part.dtype == torch.float64:
        return torch.complex(real_part, imag_part)
    return torch.complex(real_part.float(), imag_part.float())


def complex_to_real(z: Tensor) -> tuple[Tensor, Tensor]:
    """
    Split a complex tensor into real and imaginary parts.

    Args:
        z: Complex tensor

    Returns:
        Tuple of (real_part, imag_part) tensors
    """
    return z.real, z.imag


def hermitianize(rho: Tensor) -> Tensor:
    """
    Project a matrix to be Hermitian: rho = (rho + rho^dagger) / 2

    For a valid density matrix, rho must equal its conjugate transpose.
    This function enforces that constraint by averaging.

    Args:
        rho: Complex matrix of shape (..., n, n)

    Returns:
        Hermitian matrix of the same shape
    """
    return 0.5 * (rho + rho.conj().transpose(-1, -2))


def trace_normalize(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int | float | Tensor,
) -> Tensor:
    """
    Scale density matrix to have correct trace: Tr(rho @ S) = n_electrons

    In a non-orthogonal basis, the trace is computed with the overlap matrix S.

    Args:
        rho: Density matrix of shape (..., n_basis, n_basis)
        overlap: Overlap matrix of shape (n_basis, n_basis)
        n_electrons: Target number of electrons

    Returns:
        Scaled density matrix with correct trace
    """
    # Compute current trace: Tr(rho @ S) = sum_ij rho_ij * S_ji
    current_trace = torch.einsum("...ij,ji->...", rho, overlap).real

    # Handle batch dimensions
    if current_trace.dim() == 0:
        scale = n_electrons / (current_trace + 1e-10)
    else:
        scale = n_electrons / (current_trace + 1e-10)
        # Add dimensions for broadcasting
        for _ in range(rho.dim() - current_trace.dim()):
            scale = scale.unsqueeze(-1)

    return rho * scale


def check_hermiticity(rho: Tensor, tol: float = 1e-6) -> Tensor:
    """
    Check how much a matrix deviates from being Hermitian.

    Args:
        rho: Complex matrix of shape (..., n, n)
        tol: Tolerance for considering matrix Hermitian

    Returns:
        Maximum absolute deviation from Hermiticity
    """
    diff = rho - rho.conj().transpose(-1, -2)
    return diff.abs().max()


def check_trace(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int | float | Tensor,
) -> Tensor:
    """
    Check trace error of density matrix.

    Args:
        rho: Density matrix of shape (..., n_basis, n_basis)
        overlap: Overlap matrix of shape (n_basis, n_basis)
        n_electrons: Expected number of electrons

    Returns:
        Absolute trace error
    """
    trace = torch.einsum("...ij,ji->...", rho, overlap).real
    return (trace - n_electrons).abs()


def check_idempotency(rho: Tensor, overlap: Tensor) -> Tensor:
    """
    Check idempotency error: ||rho @ S @ rho - rho||

    For a pure state density matrix, rho @ S @ rho = rho.

    Args:
        rho: Density matrix of shape (..., n_basis, n_basis)
        overlap: Overlap matrix of shape (n_basis, n_basis)

    Returns:
        Frobenius norm of idempotency violation
    """
    rho_S_rho = rho @ overlap @ rho
    diff = rho_S_rho - rho
    return diff.abs().mean()


def mcweeney_purification(
    rho: Tensor,
    overlap: Tensor,
    n_iterations: int = 3,
) -> Tensor:
    """
    McWeeney purification to enforce idempotency.

    Iteratively applies: rho = 3 * rho @ S @ rho - 2 * rho @ S @ rho @ S @ rho

    This converges to an idempotent matrix (rho @ S @ rho = rho) while
    preserving trace.

    Args:
        rho: Density matrix of shape (..., n_basis, n_basis)
        overlap: Overlap matrix of shape (n_basis, n_basis)
        n_iterations: Number of purification iterations

    Returns:
        Purified density matrix
    """
    for _ in range(n_iterations):
        rho_S = rho @ overlap
        rho_S_rho = rho_S @ rho
        rho = 3 * rho_S_rho - 2 * rho_S_rho @ overlap @ rho

    return rho


def density_eigenvalues(rho: Tensor, overlap: Tensor) -> Tensor:
    """
    Compute eigenvalues of the density matrix in the overlap metric.

    Solves the generalized eigenvalue problem: rho @ v = lambda @ S @ v

    For a valid density matrix, eigenvalues should be in [0, 1] for
    fractional occupation or {0, 1} for integer occupation.

    Args:
        rho: Density matrix of shape (..., n_basis, n_basis)
        overlap: Overlap matrix of shape (n_basis, n_basis)

    Returns:
        Eigenvalues sorted in descending order
    """
    # Convert to real symmetric problem using Cholesky decomposition
    # S = L @ L^T, then solve L^{-1} @ rho @ L^{-T} @ y = lambda @ y
    L = torch.linalg.cholesky(overlap)
    L_inv = torch.linalg.inv(L)

    # Transform density matrix
    rho_transformed = L_inv @ rho @ L_inv.conj().T

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(rho_transformed.real)

    # Sort descending
    return eigenvalues.flip(-1)
