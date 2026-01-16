"""
Physics projection operators for density matrices.

These functions project density matrices onto the constraint manifold,
enforcing physics requirements like Hermiticity, correct trace, and
idempotency.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


def hermitianize(rho: Tensor) -> Tensor:
    """
    Project density matrix to Hermitian: ρ → (ρ + ρ†) / 2.

    Args:
        rho: Density matrix, shape (..., n, n) complex

    Returns:
        Hermitian density matrix
    """
    return 0.5 * (rho + rho.conj().transpose(-2, -1))


def trace_normalize(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
) -> Tensor:
    """
    Scale density matrix to have correct trace: Tr(ρS) = n_electrons.

    Args:
        rho: Density matrix, shape (..., n, n) complex
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Target number of electrons

    Returns:
        Trace-normalized density matrix
    """
    # Current trace
    current_trace = torch.einsum("...ij,ji->...", rho, overlap).real

    # Handle batched case
    if current_trace.dim() == 0:
        scale = n_electrons / (current_trace + 1e-8)
    else:
        scale = n_electrons / (current_trace + 1e-8)
        # Reshape for broadcasting
        scale = scale.reshape(*scale.shape, *([1] * (rho.dim() - current_trace.dim())))

    return rho * scale


def mcweeney_purification(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
    n_iterations: int = 3,
    mix: float = 0.5,
) -> Tensor:
    """
    McWeeney purification for idempotent density matrix.

    Iteratively applies: ρ → 3ρSρ - 2ρSρSρ

    This drives the density matrix toward idempotency while
    approximately preserving the trace.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Target number of electrons
        n_iterations: Number of purification iterations
        mix: Mixing parameter (0 = no update, 1 = full update)

    Returns:
        Purified density matrix
    """
    for _ in range(n_iterations):
        # ρS
        rho_S = rho @ overlap

        # ρSρ
        rho_S_rho = rho_S @ rho

        # ρSρS
        rho_S_rho_S = rho_S_rho @ overlap

        # ρSρSρ
        rho_S_rho_S_rho = rho_S_rho_S @ rho

        # McWeeney update
        rho_new = 3 * rho_S_rho - 2 * rho_S_rho_S_rho

        # Mix with previous
        rho = mix * rho_new + (1 - mix) * rho

        # Enforce Hermiticity after each step
        rho = hermitianize(rho)

        # Re-normalize trace
        rho = trace_normalize(rho, overlap, n_electrons)

    return rho


def diis_purification(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
    n_iterations: int = 5,
) -> Tensor:
    """
    DIIS-accelerated purification for faster convergence.

    Uses Direct Inversion in the Iterative Subspace to accelerate
    convergence to the idempotent solution.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Target number of electrons
        n_iterations: Number of DIIS iterations

    Returns:
        Purified density matrix
    """
    # Store history for DIIS extrapolation
    rho_history = []
    error_history = []

    for iteration in range(n_iterations):
        # Compute error: ρSρ - ρ
        rho_S = rho @ overlap
        rho_S_rho = rho_S @ rho
        error = rho_S_rho - rho

        rho_history.append(rho.clone())
        error_history.append(error)

        # Apply McWeeney step
        rho_S_rho_S = rho_S_rho @ overlap
        rho_S_rho_S_rho = rho_S_rho_S @ rho
        rho_new = 3 * rho_S_rho - 2 * rho_S_rho_S_rho

        if len(rho_history) >= 2:
            # DIIS extrapolation
            n_hist = len(rho_history)
            B = torch.zeros(n_hist + 1, n_hist + 1, device=rho.device, dtype=rho.dtype)

            for i in range(n_hist):
                for j in range(n_hist):
                    # Error overlap
                    B[i, j] = torch.sum(error_history[i].conj() * error_history[j]).real

            B[-1, :-1] = -1
            B[:-1, -1] = -1
            B[-1, -1] = 0

            # Solve for coefficients
            rhs = torch.zeros(n_hist + 1, device=rho.device, dtype=rho.dtype)
            rhs[-1] = -1

            try:
                coeffs = torch.linalg.solve(B, rhs)
                coeffs = coeffs[:-1].real

                # Extrapolate
                rho_new = sum(c * r for c, r in zip(coeffs, rho_history))
            except:
                pass  # Fall back to standard update on failure

        rho = hermitianize(rho_new)
        rho = trace_normalize(rho, overlap, n_electrons)

        # Limit history size
        if len(rho_history) > 5:
            rho_history.pop(0)
            error_history.pop(0)

    return rho


def project_positive_semidefinite(
    rho: Tensor,
    overlap: Tensor,
    n_electrons: int,
    max_occupation: float = 2.0,
) -> Tensor:
    """
    Project density matrix to positive semi-definite with bounded occupations.

    Diagonalizes in orthogonal basis, clips eigenvalues to [0, max_occupation],
    and transforms back.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        n_electrons: Target number of electrons
        max_occupation: Maximum occupation number

    Returns:
        Projected density matrix
    """
    device = rho.device
    n = rho.shape[0]

    # Get Cholesky factor of overlap
    S_real = overlap.real if overlap.is_complex() else overlap
    L = torch.linalg.cholesky(S_real + 1e-6 * torch.eye(n, device=device))
    L_inv = torch.linalg.inv(L)

    # Transform to orthogonal basis
    rho_orth = L_inv @ rho @ L_inv.T.conj()

    # Diagonalize
    eigenvalues, eigenvectors = torch.linalg.eigh(rho_orth)
    eigenvalues = eigenvalues.real

    # Clip eigenvalues
    eigenvalues_clipped = eigenvalues.clamp(0, max_occupation)

    # Rescale to correct trace
    current_sum = eigenvalues_clipped.sum()
    if current_sum > 0:
        eigenvalues_clipped = eigenvalues_clipped * (n_electrons / current_sum)

    # Reconstruct in orthogonal basis
    rho_orth_new = eigenvectors @ torch.diag(eigenvalues_clipped.to(eigenvectors.dtype)) @ eigenvectors.T.conj()

    # Transform back to original basis
    rho_new = L @ rho_orth_new @ L.T.conj()

    return rho_new


class PhysicsProjection(nn.Module):
    """
    Differentiable physics projection layer.

    Applies a sequence of physics projections to ensure the output
    density matrix satisfies required constraints.

    The projection order is:
    1. Hermitianize
    2. Trace normalize
    3. McWeeney purification (optional)

    Args:
        apply_hermitian: Apply Hermitianization
        apply_trace: Apply trace normalization
        apply_mcweeney: Apply McWeeney purification
        mcweeney_iterations: Number of McWeeney iterations
        mcweeney_mix: McWeeney mixing parameter
    """

    def __init__(
        self,
        apply_hermitian: bool = True,
        apply_trace: bool = True,
        apply_mcweeney: bool = False,
        mcweeney_iterations: int = 3,
        mcweeney_mix: float = 0.5,
    ):
        super().__init__()
        self.apply_hermitian = apply_hermitian
        self.apply_trace = apply_trace
        self.apply_mcweeney = apply_mcweeney
        self.mcweeney_iterations = mcweeney_iterations
        self.mcweeney_mix = mcweeney_mix

    def forward(
        self,
        rho: Tensor,
        overlap: Tensor,
        n_electrons: int,
    ) -> Tensor:
        """
        Apply physics projections.

        Args:
            rho: Density matrix, shape (..., n, n) complex
            overlap: Overlap matrix, shape (n, n)
            n_electrons: Target number of electrons

        Returns:
            Projected density matrix
        """
        # 1. Hermitianize
        if self.apply_hermitian:
            rho = hermitianize(rho)

        # 2. Trace normalize
        if self.apply_trace:
            rho = trace_normalize(rho, overlap, n_electrons)

        # 3. McWeeney purification
        if self.apply_mcweeney:
            rho = mcweeney_purification(
                rho, overlap, n_electrons,
                n_iterations=self.mcweeney_iterations,
                mix=self.mcweeney_mix,
            )

        return rho


class SoftConstraintProjection(nn.Module):
    """
    Learnable soft constraint projection.

    Instead of hard projection, applies a learned transformation
    that encourages constraint satisfaction while remaining differentiable.

    Args:
        n_basis_max: Maximum basis set size
        hidden_dim: Hidden dimension for projection network
    """

    def __init__(
        self,
        n_basis_max: int = 50,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Network to predict correction
        self.correction_net = nn.Sequential(
            nn.Linear(2 * n_basis_max * n_basis_max + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * n_basis_max * n_basis_max),
        )

        self.n_basis_max = n_basis_max

    def forward(
        self,
        rho: Tensor,
        overlap: Tensor,
        n_electrons: int,
    ) -> Tensor:
        """
        Apply soft constraint projection.

        Args:
            rho: Density matrix, shape (n, n) complex
            overlap: Overlap matrix, shape (n, n)
            n_electrons: Target number of electrons

        Returns:
            Corrected density matrix
        """
        n = rho.shape[0]
        device = rho.device

        # Compute constraint violations
        trace = torch.einsum("ij,ji->", rho, overlap).real
        trace_error = trace - n_electrons

        rho_S_rho = rho @ overlap @ rho
        idem_error = (rho_S_rho - rho).abs().mean()

        # Flatten density for network input
        rho_flat = torch.zeros(2, self.n_basis_max, self.n_basis_max, device=device)
        rho_flat[0, :n, :n] = rho.real
        rho_flat[1, :n, :n] = rho.imag
        rho_flat = rho_flat.flatten()

        # Add constraint info
        features = torch.cat([
            rho_flat,
            trace_error.unsqueeze(0),
            idem_error.unsqueeze(0),
        ])

        # Predict correction
        correction_flat = self.correction_net(features)
        correction = correction_flat.reshape(2, self.n_basis_max, self.n_basis_max)

        # Apply correction
        correction_complex = torch.complex(correction[0, :n, :n], correction[1, :n, :n])
        rho_corrected = rho + 0.1 * correction_complex

        # Still apply hard Hermitianization
        rho_corrected = hermitianize(rho_corrected)

        return rho_corrected
