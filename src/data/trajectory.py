"""
Trajectory container for RT-TDDFT simulation data.

This module provides classes for storing and loading RT-TDDFT trajectories
in HDF5 format, supporting variable basis set sizes and multiple spin channels.
"""

import h5py
import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class BasisFunctionInfo:
    """
    Metadata for a single basis function.

    Attributes:
        atom_index: Index of the atom this basis function is centered on
        atomic_number: Atomic number (Z) of the atom
        angular_momentum: Angular momentum quantum number (l)
        magnetic_quantum: Magnetic quantum number (m)
        exponents: Gaussian exponents for the basis function
    """

    atom_index: int
    atomic_number: int
    angular_momentum: int
    magnetic_quantum: int
    exponents: Optional[np.ndarray] = None

    @property
    def l(self) -> int:
        """Alias for angular_momentum."""
        return self.angular_momentum

    @property
    def m(self) -> int:
        """Alias for magnetic_quantum."""
        return self.magnetic_quantum


@dataclass
class Trajectory:
    """
    Container for a single RT-TDDFT trajectory.

    Stores all data needed for training: geometry, density matrices,
    external fields, and basis set metadata.

    Attributes:
        molecule: Molecule identifier (e.g., 'h2o', 'lih')
        positions: Atomic positions in Angstroms, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        density_real: Real part of density matrices, shape (n_steps, n_spin, n_basis, n_basis)
        density_imag: Imaginary part of density matrices, shape (n_steps, n_spin, n_basis, n_basis)
        field: External field vectors, shape (n_steps, 3)
        time: Time points in atomic units, shape (n_steps,)
        overlap: Overlap matrix, shape (n_basis, n_basis)
        basis_info: List of BasisFunctionInfo for each basis function
        n_electrons: Number of electrons
        dt: Time step in atomic units
        charge: Molecular charge
        multiplicity: Spin multiplicity
        energy: Optional total energy at each step, shape (n_steps,)
        dipole: Optional dipole moment at each step, shape (n_steps, 3)
    """

    molecule: str
    positions: np.ndarray
    atomic_numbers: np.ndarray
    density_real: np.ndarray
    density_imag: np.ndarray
    field: np.ndarray
    time: np.ndarray
    overlap: np.ndarray
    basis_info: list[BasisFunctionInfo]
    n_electrons: int
    dt: float
    charge: int = 0
    multiplicity: int = 1
    energy: Optional[np.ndarray] = None
    dipole: Optional[np.ndarray] = None
    core_hamiltonian: Optional[np.ndarray] = None
    # Extended metadata from simulation index
    geometry_name: Optional[str] = None
    field_type: Optional[str] = None
    field_polarization: Optional[str] = None
    basis_set: Optional[str] = None
    xc_functional: Optional[str] = None

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return self.positions.shape[0]

    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return self.overlap.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of time steps in the trajectory."""
        return self.density_real.shape[0]

    @property
    def n_spin(self) -> int:
        """Number of spin channels (1 for restricted, 2 for unrestricted)."""
        return self.density_real.shape[1]

    @property
    def density(self) -> np.ndarray:
        """Complex density matrices, shape (n_steps, n_spin, n_basis, n_basis)."""
        return self.density_real + 1j * self.density_imag

    def get_density_tensor(self, device: str = "cpu") -> Tensor:
        """Get density matrices as a complex PyTorch tensor."""
        return torch.complex(
            torch.tensor(self.density_real, dtype=torch.float32, device=device),
            torch.tensor(self.density_imag, dtype=torch.float32, device=device),
        )

    def get_positions_tensor(self, device: str = "cpu") -> Tensor:
        """Get atomic positions as a PyTorch tensor."""
        return torch.tensor(self.positions, dtype=torch.float32, device=device)

    def get_atomic_numbers_tensor(self, device: str = "cpu") -> Tensor:
        """Get atomic numbers as a PyTorch tensor."""
        return torch.tensor(self.atomic_numbers, dtype=torch.long, device=device)

    def get_overlap_tensor(self, device: str = "cpu") -> Tensor:
        """Get overlap matrix as a PyTorch tensor."""
        return torch.tensor(self.overlap, dtype=torch.float32, device=device)

    def get_field_tensor(self, device: str = "cpu") -> Tensor:
        """Get external field as a PyTorch tensor."""
        return torch.tensor(self.field, dtype=torch.float32, device=device)

    def get_basis_metadata(self) -> dict[str, np.ndarray]:
        """
        Get basis function metadata as numpy arrays.

        Returns:
            Dictionary with keys:
                - 'atom_idx': Atom index for each basis function
                - 'Z': Atomic number for each basis function
                - 'l': Angular momentum for each basis function
                - 'm': Magnetic quantum number for each basis function
        """
        return {
            "atom_idx": np.array([bf.atom_index for bf in self.basis_info]),
            "Z": np.array([bf.atomic_number for bf in self.basis_info]),
            "l": np.array([bf.angular_momentum for bf in self.basis_info]),
            "m": np.array([bf.magnetic_quantum for bf in self.basis_info]),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trajectory to HDF5 file.

        Args:
            filepath: Path to output HDF5 file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            # Metadata
            f.attrs["molecule"] = self.molecule
            f.attrs["n_atoms"] = self.n_atoms
            f.attrs["n_electrons"] = self.n_electrons
            f.attrs["n_basis"] = self.n_basis
            f.attrs["n_steps"] = self.n_steps
            f.attrs["dt"] = self.dt
            f.attrs["charge"] = self.charge
            f.attrs["multiplicity"] = self.multiplicity
            # Extended metadata (store as strings, empty string for None)
            f.attrs["geometry_name"] = self.geometry_name or ""
            f.attrs["field_type"] = self.field_type or ""
            f.attrs["field_polarization"] = self.field_polarization or ""
            f.attrs["basis_set"] = self.basis_set or ""
            f.attrs["xc_functional"] = self.xc_functional or ""

            # Geometry (static)
            geom = f.create_group("geometry")
            geom.create_dataset("positions", data=self.positions)
            geom.create_dataset("atomic_numbers", data=self.atomic_numbers)

            # Basis set metadata
            basis = f.create_group("basis")
            basis.create_dataset(
                "atom_index", data=[bf.atom_index for bf in self.basis_info]
            )
            basis.create_dataset(
                "angular_momentum", data=[bf.angular_momentum for bf in self.basis_info]
            )
            basis.create_dataset(
                "magnetic_quantum", data=[bf.magnetic_quantum for bf in self.basis_info]
            )

            # Static matrices
            matrices = f.create_group("matrices")
            matrices.create_dataset("overlap", data=self.overlap)
            if self.core_hamiltonian is not None:
                matrices.create_dataset("core_hamiltonian", data=self.core_hamiltonian)

            # Time series
            dynamics = f.create_group("dynamics")
            dynamics.create_dataset(
                "density_real",
                data=self.density_real.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )
            dynamics.create_dataset(
                "density_imag",
                data=self.density_imag.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )
            dynamics.create_dataset("field", data=self.field.astype(np.float32))
            dynamics.create_dataset("time", data=self.time.astype(np.float32))

            if self.energy is not None:
                dynamics.create_dataset("energy", data=self.energy.astype(np.float32))
            if self.dipole is not None:
                dynamics.create_dataset("dipole", data=self.dipole.astype(np.float32))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Trajectory":
        """
        Load trajectory from HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Trajectory object
        """
        with h5py.File(filepath, "r") as f:
            # Metadata
            molecule = f.attrs["molecule"]
            n_electrons = int(f.attrs["n_electrons"])
            dt = float(f.attrs["dt"])
            charge = int(f.attrs.get("charge", 0))
            multiplicity = int(f.attrs.get("multiplicity", 1))
            # Extended metadata (convert empty string to None)
            geometry_name = f.attrs.get("geometry_name", "") or None
            field_type = f.attrs.get("field_type", "") or None
            field_polarization = f.attrs.get("field_polarization", "") or None
            basis_set = f.attrs.get("basis_set", "") or None
            xc_functional = f.attrs.get("xc_functional", "") or None

            # Geometry
            positions = f["geometry/positions"][:]
            atomic_numbers = f["geometry/atomic_numbers"][:]

            # Basis metadata
            atom_indices = f["basis/atom_index"][:]
            angular_momenta = f["basis/angular_momentum"][:]
            magnetic_quanta = f["basis/magnetic_quantum"][:]

            basis_info = []
            for i in range(len(atom_indices)):
                bf = BasisFunctionInfo(
                    atom_index=int(atom_indices[i]),
                    atomic_number=int(atomic_numbers[atom_indices[i]]),
                    angular_momentum=int(angular_momenta[i]),
                    magnetic_quantum=int(magnetic_quanta[i]),
                )
                basis_info.append(bf)

            # Static matrices
            overlap = f["matrices/overlap"][:]
            core_hamiltonian = None
            if "core_hamiltonian" in f["matrices"]:
                core_hamiltonian = f["matrices/core_hamiltonian"][:]

            # Time series
            density_real = f["dynamics/density_real"][:]
            density_imag = f["dynamics/density_imag"][:]
            field_data = f["dynamics/field"][:]
            time = f["dynamics/time"][:]

            energy = None
            dipole = None
            if "energy" in f["dynamics"]:
                energy = f["dynamics/energy"][:]
            if "dipole" in f["dynamics"]:
                dipole = f["dynamics/dipole"][:]

            return cls(
                molecule=molecule,
                positions=positions,
                atomic_numbers=atomic_numbers,
                density_real=density_real,
                density_imag=density_imag,
                field=field_data,
                time=time,
                overlap=overlap,
                basis_info=basis_info,
                n_electrons=n_electrons,
                dt=dt,
                charge=charge,
                multiplicity=multiplicity,
                energy=energy,
                dipole=dipole,
                core_hamiltonian=core_hamiltonian,
                geometry_name=geometry_name,
                field_type=field_type,
                field_polarization=field_polarization,
                basis_set=basis_set,
                xc_functional=xc_functional,
            )

    def slice_time(self, start: int, end: int) -> "Trajectory":
        """
        Create a new trajectory with a subset of time steps.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New Trajectory with sliced time series
        """
        return Trajectory(
            molecule=self.molecule,
            positions=self.positions.copy(),
            atomic_numbers=self.atomic_numbers.copy(),
            density_real=self.density_real[start:end].copy(),
            density_imag=self.density_imag[start:end].copy(),
            field=self.field[start:end].copy(),
            time=self.time[start:end].copy(),
            overlap=self.overlap.copy(),
            basis_info=self.basis_info.copy(),
            n_electrons=self.n_electrons,
            dt=self.dt,
            charge=self.charge,
            multiplicity=self.multiplicity,
            energy=self.energy[start:end].copy() if self.energy is not None else None,
            dipole=self.dipole[start:end].copy() if self.dipole is not None else None,
            core_hamiltonian=self.core_hamiltonian.copy()
            if self.core_hamiltonian is not None
            else None,
            geometry_name=self.geometry_name,
            field_type=self.field_type,
            field_polarization=self.field_polarization,
            basis_set=self.basis_set,
            xc_functional=self.xc_functional,
        )
