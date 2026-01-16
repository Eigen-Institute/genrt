"""
PyTorch Dataset classes for RT-TDDFT trajectories.

This module provides Dataset implementations for training the ML model,
supporting variable basis set sizes and efficient data loading from HDF5.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Union, Callable
from dataclasses import dataclass

from .trajectory import Trajectory
from ..utils.graph import build_molecular_graph, MolecularGraph


@dataclass
class TrajectoryBatch:
    """
    Batch of trajectory data for model input.

    Attributes:
        positions: Atomic positions, shape (batch, n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (batch, n_atoms)
        density_current: Current density matrix (complex), shape (batch, n_spin, n_basis, n_basis)
        density_next: Next density matrix (target), shape (batch, n_spin, n_basis, n_basis)
        field: External field, shape (batch, 3)
        overlap: Overlap matrix, shape (batch, n_basis, n_basis)
        n_electrons: Number of electrons, shape (batch,)
        graphs: List of MolecularGraph objects (one per sample)
        basis_metadata: Dictionary of basis function metadata
    """

    positions: Tensor
    atomic_numbers: Tensor
    density_current: Tensor
    density_next: Tensor
    field: Tensor
    overlap: Tensor
    n_electrons: Tensor
    graphs: list[MolecularGraph]
    basis_metadata: dict[str, Tensor]


class TrajectoryDataset(Dataset):
    """
    Dataset for a single RT-TDDFT trajectory.

    Each sample is a (current_state, next_state) pair for supervised learning
    of the dynamics.

    Args:
        trajectory: Trajectory object containing the simulation data
        transform: Optional transform to apply to samples
        sequence_length: Number of consecutive steps per sample (default 1)
        stride: Step stride for sampling (default 1)
        device: Device to load tensors on
    """

    def __init__(
        self,
        trajectory: Trajectory,
        transform: Optional[Callable] = None,
        sequence_length: int = 1,
        stride: int = 1,
        device: str = "cpu",
    ):
        self.trajectory = trajectory
        self.transform = transform
        self.sequence_length = sequence_length
        self.stride = stride
        self.device = device

        # Pre-compute number of valid samples
        # Need at least sequence_length + 1 steps (current + next)
        max_start = trajectory.n_steps - sequence_length
        self.n_samples = max(0, (max_start - 1) // stride + 1)

        # Pre-build molecular graph (static for trajectory)
        self.graph = build_molecular_graph(
            torch.tensor(trajectory.positions, dtype=torch.float32),
            torch.tensor(trajectory.atomic_numbers, dtype=torch.long),
        )

        # Pre-compute basis metadata tensors
        self.basis_metadata = {
            key: torch.tensor(val, dtype=torch.long)
            for key, val in trajectory.get_basis_metadata().items()
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - 'positions': (n_atoms, 3)
                - 'atomic_numbers': (n_atoms,)
                - 'density_current': (n_spin, n_basis, n_basis) complex
                - 'density_next': (n_spin, n_basis, n_basis) complex
                - 'field': (3,)
                - 'overlap': (n_basis, n_basis)
                - 'n_electrons': scalar
                - 'graph': MolecularGraph
                - 'basis_metadata': dict of tensors
        """
        # Map sample index to trajectory index
        t = idx * self.stride

        # Get density matrices
        density_current = self.trajectory.get_density_tensor(self.device)[t]
        density_next = self.trajectory.get_density_tensor(self.device)[t + 1]

        # Get field at current time
        field = self.trajectory.get_field_tensor(self.device)[t]

        sample = {
            "positions": self.trajectory.get_positions_tensor(self.device),
            "atomic_numbers": self.trajectory.get_atomic_numbers_tensor(self.device),
            "density_current": density_current,
            "density_next": density_next,
            "field": field,
            "overlap": self.trajectory.get_overlap_tensor(self.device),
            "n_electrons": torch.tensor(
                self.trajectory.n_electrons, dtype=torch.float32, device=self.device
            ),
            "graph": self.graph,
            "basis_metadata": self.basis_metadata,
            "time_idx": t,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class UnifiedTrajectoryDataset(Dataset):
    """
    Dataset combining multiple trajectories from different molecules.

    Supports variable basis set sizes by returning samples with their
    associated geometry and basis metadata.

    Args:
        trajectory_paths: List of paths to HDF5 trajectory files
        transform: Optional transform to apply to samples
        sequence_length: Number of consecutive steps per sample
        stride: Step stride for sampling within each trajectory
        device: Device to load tensors on
        cache_trajectories: Whether to cache loaded trajectories in memory
    """

    def __init__(
        self,
        trajectory_paths: list[Union[str, Path]],
        transform: Optional[Callable] = None,
        sequence_length: int = 1,
        stride: int = 1,
        device: str = "cpu",
        cache_trajectories: bool = True,
    ):
        self.trajectory_paths = [Path(p) for p in trajectory_paths]
        self.transform = transform
        self.sequence_length = sequence_length
        self.stride = stride
        self.device = device
        self.cache_trajectories = cache_trajectories

        # Trajectory cache
        self._trajectory_cache: dict[int, Trajectory] = {}
        self._graph_cache: dict[int, MolecularGraph] = {}
        self._basis_metadata_cache: dict[int, dict[str, Tensor]] = {}

        # Build index mapping: sample_idx -> (trajectory_idx, time_idx)
        self._build_index()

    def _build_index(self):
        """Build mapping from sample index to (trajectory, time) pair."""
        self.index_map = []  # List of (traj_idx, time_idx)
        self.traj_lengths = []

        for traj_idx, path in enumerate(self.trajectory_paths):
            # Load trajectory to get length (or read from metadata)
            traj = self._load_trajectory(traj_idx)
            n_steps = traj.n_steps

            # Number of valid samples from this trajectory
            max_start = n_steps - self.sequence_length
            n_samples = max(0, (max_start - 1) // self.stride + 1)
            self.traj_lengths.append(n_samples)

            for sample_idx in range(n_samples):
                t = sample_idx * self.stride
                self.index_map.append((traj_idx, t))

            # Clear cache if not caching
            if not self.cache_trajectories:
                self._trajectory_cache.pop(traj_idx, None)

    def _load_trajectory(self, traj_idx: int) -> Trajectory:
        """Load or retrieve cached trajectory."""
        if traj_idx in self._trajectory_cache:
            return self._trajectory_cache[traj_idx]

        traj = Trajectory.load(self.trajectory_paths[traj_idx])

        if self.cache_trajectories:
            self._trajectory_cache[traj_idx] = traj

        return traj

    def _get_graph(self, traj_idx: int, trajectory: Trajectory) -> MolecularGraph:
        """Get or build molecular graph for trajectory."""
        if traj_idx in self._graph_cache:
            return self._graph_cache[traj_idx]

        graph = build_molecular_graph(
            torch.tensor(trajectory.positions, dtype=torch.float32),
            torch.tensor(trajectory.atomic_numbers, dtype=torch.long),
        )

        if self.cache_trajectories:
            self._graph_cache[traj_idx] = graph

        return graph

    def _get_basis_metadata(
        self, traj_idx: int, trajectory: Trajectory
    ) -> dict[str, Tensor]:
        """Get basis metadata tensors for trajectory."""
        if traj_idx in self._basis_metadata_cache:
            return self._basis_metadata_cache[traj_idx]

        metadata = {
            key: torch.tensor(val, dtype=torch.long)
            for key, val in trajectory.get_basis_metadata().items()
        }

        if self.cache_trajectories:
            self._basis_metadata_cache[traj_idx] = metadata

        return metadata

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dictionary containing sample data with variable basis size.
        """
        traj_idx, t = self.index_map[idx]

        # Load trajectory
        trajectory = self._load_trajectory(traj_idx)

        # Get data
        density_current = trajectory.get_density_tensor(self.device)[t]
        density_next = trajectory.get_density_tensor(self.device)[t + 1]
        field = trajectory.get_field_tensor(self.device)[t]

        sample = {
            "positions": trajectory.get_positions_tensor(self.device),
            "atomic_numbers": trajectory.get_atomic_numbers_tensor(self.device),
            "density_current": density_current,
            "density_next": density_next,
            "field": field,
            "overlap": trajectory.get_overlap_tensor(self.device),
            "n_electrons": torch.tensor(
                trajectory.n_electrons, dtype=torch.float32, device=self.device
            ),
            "n_basis": trajectory.n_basis,
            "graph": self._get_graph(traj_idx, trajectory),
            "basis_metadata": self._get_basis_metadata(traj_idx, trajectory),
            "molecule": trajectory.molecule,
            "trajectory_idx": traj_idx,
            "time_idx": t,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_molecule_indices(self, molecule: str) -> list[int]:
        """Get indices of all samples from a specific molecule."""
        indices = []
        for idx, (traj_idx, _) in enumerate(self.index_map):
            traj = self._load_trajectory(traj_idx)
            if traj.molecule == molecule:
                indices.append(idx)
        return indices

    def get_trajectory_samples(self, traj_idx: int) -> list[int]:
        """Get indices of all samples from a specific trajectory."""
        return [
            idx
            for idx, (t_idx, _) in enumerate(self.index_map)
            if t_idx == traj_idx
        ]


def collate_variable_basis(samples: list[dict]) -> dict:
    """
    Custom collate function for variable basis set sizes.

    Since different molecules have different n_basis, we cannot simply
    stack density matrices. Instead, we return lists where appropriate.

    Args:
        samples: List of sample dictionaries

    Returns:
        Batched dictionary with lists for variable-size tensors
    """
    batch = {
        "positions": [s["positions"] for s in samples],
        "atomic_numbers": [s["atomic_numbers"] for s in samples],
        "density_current": [s["density_current"] for s in samples],
        "density_next": [s["density_next"] for s in samples],
        "field": torch.stack([s["field"] for s in samples]),
        "overlap": [s["overlap"] for s in samples],
        "n_electrons": torch.stack([s["n_electrons"] for s in samples]),
        "n_basis": [s["n_basis"] for s in samples],
        "graphs": [s["graph"] for s in samples],
        "basis_metadata": [s["basis_metadata"] for s in samples],
        "molecules": [s.get("molecule", "") for s in samples],
        "trajectory_indices": [s.get("trajectory_idx", -1) for s in samples],
        "time_indices": [s.get("time_idx", -1) for s in samples],
    }
    return batch


def collate_fixed_basis(samples: list[dict]) -> dict:
    """
    Collate function for fixed basis set size (single molecule type).

    All samples must have the same n_basis for this to work.

    Args:
        samples: List of sample dictionaries

    Returns:
        Batched dictionary with stacked tensors
    """
    # Check all samples have same basis size
    n_basis_set = {s["density_current"].shape[-1] for s in samples}
    if len(n_basis_set) != 1:
        raise ValueError(
            f"All samples must have same n_basis for fixed_basis collation, "
            f"got {n_basis_set}"
        )

    batch = {
        "positions": torch.stack([s["positions"] for s in samples]),
        "atomic_numbers": torch.stack([s["atomic_numbers"] for s in samples]),
        "density_current": torch.stack([s["density_current"] for s in samples]),
        "density_next": torch.stack([s["density_next"] for s in samples]),
        "field": torch.stack([s["field"] for s in samples]),
        "overlap": torch.stack([s["overlap"] for s in samples]),
        "n_electrons": torch.stack([s["n_electrons"] for s in samples]),
        "graphs": [s["graph"] for s in samples],
        "basis_metadata": samples[0]["basis_metadata"],  # Assume same for all
    }
    return batch
