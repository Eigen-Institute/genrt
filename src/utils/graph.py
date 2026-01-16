"""
Molecular graph construction utilities.

This module provides functions for building molecular graphs suitable for
E3NN equivariant neural networks, including neighbor finding and edge
feature computation.
"""

import torch
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


@dataclass
class MolecularGraph:
    """
    Container for molecular graph data.

    Attributes:
        positions: Atomic positions, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers (Z), shape (n_atoms,)
        edge_index: Edge connectivity, shape (2, n_edges)
        edge_vec: Edge vectors (r_j - r_i), shape (n_edges, 3)
        edge_len: Edge lengths, shape (n_edges,)
        num_atoms: Number of atoms
        num_edges: Number of edges
    """

    positions: Tensor
    atomic_numbers: Tensor
    edge_index: Tensor
    edge_vec: Tensor
    edge_len: Tensor

    @property
    def num_atoms(self) -> int:
        return self.positions.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def build_molecular_graph(
    positions: Tensor,
    atomic_numbers: Tensor,
    cutoff: float = 5.0,
    self_loops: bool = False,
) -> MolecularGraph:
    """
    Build a molecular graph with edges between atoms within cutoff distance.

    Creates a fully connected graph within the cutoff radius, suitable for
    E3NN message passing. Edge features include displacement vectors which
    are used for SO(3) equivariance.

    Args:
        positions: Atomic positions in Angstroms, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        cutoff: Maximum distance for edges in Angstroms
        self_loops: Whether to include self-loops (i -> i edges)

    Returns:
        MolecularGraph containing the molecular graph structure
    """
    n_atoms = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    # Compute all pairwise distances
    # diff[i, j] = positions[j] - positions[i]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (n, n, 3)
    dist = torch.norm(diff, dim=-1)  # (n, n)

    # Find edges within cutoff
    if self_loops:
        mask = dist <= cutoff
    else:
        mask = (dist <= cutoff) & (dist > 0)

    # Get edge indices
    src, dst = torch.where(mask)
    edge_index = torch.stack([src, dst], dim=0)  # (2, n_edges)

    # Compute edge vectors and lengths
    edge_vec = diff[src, dst]  # (n_edges, 3)
    edge_len = dist[src, dst]  # (n_edges,)

    return MolecularGraph(
        positions=positions,
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_len=edge_len,
    )


def compute_edge_vectors(
    positions: Tensor,
    edge_index: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Compute edge displacement vectors and lengths.

    Args:
        positions: Atomic positions, shape (n_atoms, 3)
        edge_index: Edge connectivity, shape (2, n_edges)

    Returns:
        Tuple of (edge_vec, edge_len):
            - edge_vec: Displacement vectors (r_j - r_i), shape (n_edges, 3)
            - edge_len: Edge lengths, shape (n_edges,)
    """
    src, dst = edge_index
    edge_vec = positions[dst] - positions[src]
    edge_len = torch.norm(edge_vec, dim=-1)
    return edge_vec, edge_len


def compute_edge_spherical(edge_vec: Tensor) -> Tensor:
    """
    Compute unit vectors for edges (for E3NN spherical harmonics).

    Args:
        edge_vec: Edge displacement vectors, shape (n_edges, 3)

    Returns:
        Normalized edge vectors, shape (n_edges, 3)
    """
    edge_len = torch.norm(edge_vec, dim=-1, keepdim=True)
    # Avoid division by zero for self-loops
    edge_len = torch.clamp(edge_len, min=1e-8)
    return edge_vec / edge_len


def radial_basis_bessel(
    edge_len: Tensor,
    num_basis: int = 8,
    cutoff: float = 5.0,
) -> Tensor:
    """
    Compute Bessel radial basis functions.

    Uses the form: f_n(r) = sqrt(2/c) * sin(n * pi * r / c) / r
    where c is the cutoff and n = 1, 2, ..., num_basis

    Args:
        edge_len: Edge lengths, shape (n_edges,)
        num_basis: Number of basis functions
        cutoff: Cutoff distance

    Returns:
        Radial basis values, shape (n_edges, num_basis)
    """
    n = torch.arange(1, num_basis + 1, device=edge_len.device, dtype=edge_len.dtype)
    # Expand dimensions for broadcasting
    r = edge_len.unsqueeze(-1)  # (n_edges, 1)

    # Bessel basis
    prefactor = (2.0 / cutoff) ** 0.5
    arg = n * torch.pi * r / cutoff

    # Avoid division by zero at r=0
    r_safe = torch.clamp(r, min=1e-8)
    basis = prefactor * torch.sin(arg) / r_safe

    return basis


def smooth_cutoff(edge_len: Tensor, cutoff: float, width: float = 0.5) -> Tensor:
    """
    Compute smooth cutoff function that goes to zero at cutoff.

    Uses cosine cutoff: f(r) = 0.5 * (1 + cos(pi * r / cutoff)) for r < cutoff

    Args:
        edge_len: Edge lengths, shape (n_edges,)
        cutoff: Cutoff distance
        width: Width of the cutoff region (not used in cosine cutoff)

    Returns:
        Cutoff values in [0, 1], shape (n_edges,)
    """
    x = edge_len / cutoff
    cutoff_vals = torch.where(
        x < 1.0,
        0.5 * (1.0 + torch.cos(torch.pi * x)),
        torch.zeros_like(x),
    )
    return cutoff_vals


def get_edge_attributes(
    edge_vec: Tensor,
    edge_len: Tensor,
    num_basis: int = 8,
    cutoff: float = 5.0,
) -> dict[str, Tensor]:
    """
    Compute all edge attributes needed for E3NN convolution.

    Args:
        edge_vec: Edge displacement vectors, shape (n_edges, 3)
        edge_len: Edge lengths, shape (n_edges,)
        num_basis: Number of radial basis functions
        cutoff: Cutoff distance

    Returns:
        Dictionary containing:
            - 'edge_sh': Spherical harmonics input (unit vectors), shape (n_edges, 3)
            - 'edge_rbf': Radial basis functions, shape (n_edges, num_basis)
            - 'edge_cutoff': Smooth cutoff values, shape (n_edges,)
    """
    edge_sh = compute_edge_spherical(edge_vec)
    edge_rbf = radial_basis_bessel(edge_len, num_basis, cutoff)
    edge_cutoff = smooth_cutoff(edge_len, cutoff)

    return {
        "edge_sh": edge_sh,
        "edge_rbf": edge_rbf,
        "edge_cutoff": edge_cutoff,
    }


def batch_graphs(graphs: list[MolecularGraph]) -> MolecularGraph:
    """
    Batch multiple molecular graphs into a single graph.

    Adjusts edge indices to account for concatenated node lists.

    Args:
        graphs: List of MolecularGraph objects

    Returns:
        Batched MolecularGraph
    """
    if len(graphs) == 1:
        return graphs[0]

    positions_list = []
    atomic_numbers_list = []
    edge_index_list = []
    edge_vec_list = []
    edge_len_list = []

    node_offset = 0
    for g in graphs:
        positions_list.append(g.positions)
        atomic_numbers_list.append(g.atomic_numbers)

        # Offset edge indices
        edge_index_list.append(g.edge_index + node_offset)
        edge_vec_list.append(g.edge_vec)
        edge_len_list.append(g.edge_len)

        node_offset += g.num_atoms

    return MolecularGraph(
        positions=torch.cat(positions_list, dim=0),
        atomic_numbers=torch.cat(atomic_numbers_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_vec=torch.cat(edge_vec_list, dim=0),
        edge_len=torch.cat(edge_len_list, dim=0),
    )
