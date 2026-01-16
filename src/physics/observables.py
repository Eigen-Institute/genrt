"""
Physical observable calculations from density matrices.

This module provides functions to compute physical observables like
dipole moments, populations, and energies from density matrices.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple, Dict


def compute_dipole_moment(
    rho: Tensor,
    dipole_integrals: Tensor,
    overlap: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute electric dipole moment from density matrix.

    μ = Tr(ρ·D) where D is the dipole integral matrix.

    Args:
        rho: Density matrix, shape (..., n, n) complex
        dipole_integrals: Dipole integral matrices, shape (3, n, n)
        overlap: Optional overlap matrix for non-orthogonal basis

    Returns:
        Dipole moment vector, shape (..., 3)
    """
    # Handle spin channels
    if rho.dim() >= 3 and rho.shape[-3] > 1:
        # Sum over spin channels
        rho_total = rho.sum(dim=-3)
    else:
        rho_total = rho.squeeze(-3) if rho.dim() == 3 else rho

    # Ensure compatible dtypes
    if rho_total.is_complex() and not dipole_integrals.is_complex():
        dipole_integrals = dipole_integrals.to(rho_total.dtype)

    # Compute dipole components
    dipole = torch.einsum("...ij,cji->...c", rho_total, dipole_integrals)

    return dipole.real


def compute_mulliken_populations(
    rho: Tensor,
    overlap: Tensor,
    atom_basis_map: Tensor,
) -> Tensor:
    """
    Compute Mulliken atomic populations.

    P_A = Σ_{μ∈A} (ρS)_{μμ}

    Args:
        rho: Density matrix, shape (..., n, n) complex
        overlap: Overlap matrix, shape (n, n)
        atom_basis_map: Mapping from basis function to atom, shape (n,)

    Returns:
        Atomic populations, shape (..., n_atoms)
    """
    n_atoms = atom_basis_map.max().item() + 1
    device = rho.device

    # Compute ρS
    rho_S = rho @ overlap

    # Sum diagonal elements for each atom
    populations = torch.zeros(*rho.shape[:-2], n_atoms, device=device)

    for mu in range(rho.shape[-1]):
        atom_idx = atom_basis_map[mu].item()
        populations[..., atom_idx] += rho_S[..., mu, mu].real

    return populations


def compute_lowdin_populations(
    rho: Tensor,
    overlap: Tensor,
    atom_basis_map: Tensor,
) -> Tensor:
    """
    Compute Löwdin atomic populations.

    Uses S^{1/2} transformation for symmetric partitioning.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        atom_basis_map: Mapping from basis function to atom, shape (n,)

    Returns:
        Atomic populations, shape (n_atoms,)
    """
    n = overlap.shape[0]
    n_atoms = atom_basis_map.max().item() + 1
    device = rho.device

    # Compute S^{1/2}
    eigenvalues, eigenvectors = torch.linalg.eigh(overlap.real)
    S_half = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T

    # Transform density: ρ' = S^{1/2} ρ S^{1/2}
    rho_lowdin = S_half @ rho @ S_half

    # Sum diagonal elements for each atom
    populations = torch.zeros(n_atoms, device=device)

    for mu in range(n):
        atom_idx = atom_basis_map[mu].item()
        populations[atom_idx] += rho_lowdin[mu, mu].real

    return populations


def compute_natural_orbital_occupations(
    rho: Tensor,
    overlap: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Compute natural orbital occupations via diagonalization.

    The natural orbitals are eigenvectors of the density matrix
    in orthogonal basis, and eigenvalues are the occupations.

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)

    Returns:
        Tuple of:
        - Occupation numbers, shape (n,)
        - Natural orbital coefficients, shape (n, n)
    """
    n = overlap.shape[0]
    device = rho.device

    # Transform to orthogonal basis via Cholesky
    S_real = overlap.real if overlap.is_complex() else overlap
    L = torch.linalg.cholesky(S_real + 1e-6 * torch.eye(n, device=device))
    L_inv = torch.linalg.inv(L)

    # Ensure compatible dtypes
    if rho.is_complex():
        L_inv = L_inv.to(rho.dtype)

    # ρ' = L^{-1} ρ L^{-T}
    rho_orth = L_inv @ rho @ L_inv.T.conj()

    # Diagonalize
    occupations, coeffs_orth = torch.linalg.eigh(rho_orth)

    # Transform coefficients back to AO basis
    coeffs = L_inv.T.conj() @ coeffs_orth

    return occupations.real, coeffs


def compute_energy_components(
    rho: Tensor,
    hamiltonian: Tensor,
    overlap: Tensor,
    fock: Optional[Tensor] = None,
    two_electron: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """
    Compute energy components from density matrix.

    Args:
        rho: Density matrix, shape (n, n) complex
        hamiltonian: Core Hamiltonian (one-electron), shape (n, n)
        overlap: Overlap matrix, shape (n, n)
        fock: Optional Fock matrix, shape (n, n)
        two_electron: Optional two-electron integrals (not implemented)

    Returns:
        Dictionary of energy components
    """
    energies = {}

    # One-electron energy: Tr(ρH)
    energies['one_electron'] = torch.einsum("ij,ji->", rho, hamiltonian).real

    # If Fock matrix provided, compute total electronic energy
    if fock is not None:
        # E = 0.5 * Tr(ρ(H + F))
        energies['electronic'] = 0.5 * torch.einsum(
            "ij,ji->", rho, hamiltonian + fock
        ).real

    return energies


def compute_bond_order(
    rho: Tensor,
    overlap: Tensor,
    atom_A_basis: Tensor,
    atom_B_basis: Tensor,
) -> Tensor:
    """
    Compute Mayer bond order between two atoms.

    B_{AB} = Σ_{μ∈A} Σ_{ν∈B} (ρS)_{μν} (ρS)_{νμ}

    Args:
        rho: Density matrix, shape (n, n) complex
        overlap: Overlap matrix, shape (n, n)
        atom_A_basis: Basis indices on atom A, shape (n_A,)
        atom_B_basis: Basis indices on atom B, shape (n_B,)

    Returns:
        Bond order (scalar)
    """
    # Compute ρS
    rho_S = rho @ overlap

    # Sum over basis pairs
    bond_order = torch.tensor(0.0, device=rho.device)

    for mu in atom_A_basis:
        for nu in atom_B_basis:
            bond_order += (rho_S[mu, nu] * rho_S[nu, mu]).real

    return bond_order


def compute_electron_density_at_point(
    rho: Tensor,
    basis_values: Tensor,
) -> Tensor:
    """
    Compute electron density at a grid point.

    n(r) = Σ_{μν} ρ_{μν} χ_μ(r) χ_ν(r)

    Args:
        rho: Density matrix, shape (n, n) complex
        basis_values: Basis function values at point, shape (n,)

    Returns:
        Electron density (scalar)
    """
    # n(r) = χ^T ρ χ
    density = torch.einsum("i,ij,j->", basis_values, rho, basis_values)
    return density.real


def compute_current_density(
    rho: Tensor,
    rho_dot: Tensor,
    momentum_integrals: Tensor,
) -> Tensor:
    """
    Compute current density from time-dependent density matrix.

    j = -i/2 * Tr(ρ̇·p - ρ·ṗ) where p is momentum operator

    For RT-TDDFT, this gives the induced current under field.

    Args:
        rho: Density matrix at time t, shape (n, n) complex
        rho_dot: Time derivative of density, shape (n, n) complex
        momentum_integrals: Momentum integral matrices, shape (3, n, n)

    Returns:
        Current density vector, shape (3,)
    """
    # j_α = -i * Tr(ρ̇ · p_α)
    current = -1j * torch.einsum("ij,cji->c", rho_dot, momentum_integrals)

    return current.real


class ObservableCalculator(torch.nn.Module):
    """
    Module for computing observables from density matrices.

    Caches integral matrices and atom mappings for efficient
    repeated calculations.

    Args:
        dipole_integrals: Optional dipole integral matrices
        overlap: Overlap matrix
        atom_basis_map: Mapping from basis to atoms
    """

    def __init__(
        self,
        overlap: Tensor,
        atom_basis_map: Tensor,
        dipole_integrals: Optional[Tensor] = None,
    ):
        super().__init__()
        self.register_buffer('overlap', overlap)
        self.register_buffer('atom_basis_map', atom_basis_map)
        if dipole_integrals is not None:
            self.register_buffer('dipole_integrals', dipole_integrals)
        else:
            self.dipole_integrals = None

    def forward(self, rho: Tensor) -> Dict[str, Tensor]:
        """
        Compute all available observables.

        Args:
            rho: Density matrix, shape (..., n, n) complex

        Returns:
            Dictionary of observables
        """
        results = {}

        # Mulliken populations
        results['mulliken_populations'] = compute_mulliken_populations(
            rho, self.overlap, self.atom_basis_map
        )

        # Dipole moment if integrals available
        if self.dipole_integrals is not None:
            results['dipole_moment'] = compute_dipole_moment(
                rho, self.dipole_integrals
            )

        return results
