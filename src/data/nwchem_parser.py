"""
NWChem output file parser for RT-TDDFT trajectories.

This module provides utilities for parsing NWChem restart files and
extracting density matrices, overlap matrices, and other quantities
needed for training the ML model.
"""

import numpy as np
import os
import re
import struct
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

from .trajectory import Trajectory, BasisFunctionInfo

if TYPE_CHECKING:
    from .simulation_index import SimulationRecord


@dataclass
class NWChemParseResult:
    """
    Intermediate result from parsing NWChem files.

    Attributes:
        positions: Atomic positions in Angstroms
        atomic_numbers: Atomic numbers
        overlap: Overlap matrix
        density_matrices: List of density matrices at each time step
        field_vectors: List of field vectors at each time step
        time_points: List of time points
        n_electrons: Number of electrons
        n_basis: Number of basis functions
        basis_info: List of basis function metadata
        energies: Optional list of energies
        dipoles: Optional list of dipole moments
    """

    positions: np.ndarray
    atomic_numbers: np.ndarray
    overlap: np.ndarray
    density_matrices: list[np.ndarray]
    field_vectors: list[np.ndarray]
    time_points: list[float]
    n_electrons: int
    n_basis: int
    basis_info: list[BasisFunctionInfo]
    energies: Optional[list[float]] = None
    dipoles: Optional[list[np.ndarray]] = None


class NWChemParser:
    """
    Parser for NWChem RT-TDDFT output files.

    This parser handles:
    - Geometry extraction from input/output files
    - Overlap matrix from movecs or output
    - Density matrices from restart files
    - External field from input parameters
    """

    # Atomic number mapping
    ELEMENT_TO_Z = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
    }

    # Angular momentum labels
    ANGULAR_MOMENTUM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize parser with base directory containing NWChem files.

        Args:
            base_dir: Directory containing NWChem output files.
                     If None, uses current directory (useful for parse_trajectory_from_record).
        """
        self.base_dir = Path(base_dir) if base_dir is not None else Path(".")

    def parse_geometry(self, filename: str = "geometry.xyz") -> tuple[np.ndarray, np.ndarray]:
        """
        Parse XYZ geometry file.

        Args:
            filename: Name of XYZ file

        Returns:
            Tuple of (positions, atomic_numbers)
        """
        filepath = self.base_dir / filename
        positions = []
        atomic_numbers = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        # Skip header lines (n_atoms and comment)
        n_atoms = int(lines[0].strip())
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

            atomic_numbers.append(self.ELEMENT_TO_Z.get(element, 0))
            positions.append([x, y, z])

        return np.array(positions), np.array(atomic_numbers, dtype=np.int32)

    def parse_overlap_matrix(self, filename: str = "overlap.dat") -> np.ndarray:
        """
        Parse overlap matrix from text file.

        Expects format: i j S_ij (one element per line) or full matrix.

        Args:
            filename: Name of overlap matrix file

        Returns:
            Overlap matrix as numpy array
        """
        filepath = self.base_dir / filename
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Try to detect format
        first_line = lines[0].split()
        if len(first_line) == 3:
            # Sparse format: i j value
            return self._parse_sparse_matrix(lines)
        else:
            # Dense format
            return self._parse_dense_matrix(lines)

    def _parse_sparse_matrix(self, lines: list[str]) -> np.ndarray:
        """Parse sparse matrix format (i j value)."""
        elements = []
        max_idx = 0

        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                i, j = int(parts[0]) - 1, int(parts[1]) - 1  # 1-indexed to 0-indexed
                val = float(parts[2])
                elements.append((i, j, val))
                max_idx = max(max_idx, i, j)

        n = max_idx + 1
        matrix = np.zeros((n, n), dtype=np.float64)
        for i, j, val in elements:
            matrix[i, j] = val
            matrix[j, i] = val  # Symmetric

        return matrix

    def _parse_dense_matrix(self, lines: list[str]) -> np.ndarray:
        """Parse dense matrix format."""
        data = []
        for line in lines:
            row = [float(x) for x in line.split()]
            data.append(row)
        return np.array(data, dtype=np.float64)

    def parse_density_restart(
        self,
        filename: str,
        n_basis: int,
        n_spin: int = 1,
    ) -> np.ndarray:
        """
        Parse density matrix from NWChem restart file.

        The restart file contains the real and imaginary parts of the
        density matrix in a specific binary or text format.

        Args:
            filename: Name of restart file
            n_basis: Number of basis functions
            n_spin: Number of spin channels

        Returns:
            Complex density matrix of shape (n_spin, n_basis, n_basis)
        """
        filepath = self.base_dir / filename

        # Try text format first
        try:
            return self._parse_density_text(filepath, n_basis, n_spin)
        except (ValueError, UnicodeDecodeError):
            # Try binary format
            return self._parse_density_binary(filepath, n_basis, n_spin)

    def _parse_density_text(
        self,
        filepath: Path,
        n_basis: int,
        n_spin: int,
    ) -> np.ndarray:
        """Parse density matrix from text format."""
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Assume format: real_ij imag_ij per line, row-major order
        values = []
        for line in lines:
            parts = line.split()
            for part in parts:
                values.append(float(part))

        # Check if we have real+imag or just real
        expected_real = n_spin * n_basis * n_basis
        expected_complex = 2 * expected_real

        if len(values) == expected_complex:
            # Interleaved real/imag
            real_vals = values[0::2]
            imag_vals = values[1::2]
        elif len(values) == expected_real * 2:
            # Sequential real then imag
            real_vals = values[:expected_real]
            imag_vals = values[expected_real:]
        elif len(values) == expected_real:
            # Real only
            real_vals = values
            imag_vals = [0.0] * expected_real
        else:
            raise ValueError(
                f"Unexpected number of values: {len(values)}, "
                f"expected {expected_real} or {expected_complex}"
            )

        real_arr = np.array(real_vals).reshape(n_spin, n_basis, n_basis)
        imag_arr = np.array(imag_vals).reshape(n_spin, n_basis, n_basis)

        return real_arr + 1j * imag_arr

    def _parse_density_binary(
        self,
        filepath: Path,
        n_basis: int,
        n_spin: int,
    ) -> np.ndarray:
        """Parse density matrix from binary format."""
        with open(filepath, "rb") as f:
            data = np.fromfile(f, dtype=np.float64)

        expected_real = n_spin * n_basis * n_basis
        expected_complex = 2 * expected_real

        if len(data) == expected_complex:
            # Interleaved complex
            complex_data = data.view(np.complex128)
            return complex_data.reshape(n_spin, n_basis, n_basis)
        elif len(data) == expected_real:
            # Real only
            return data.reshape(n_spin, n_basis, n_basis).astype(np.complex128)
        else:
            raise ValueError(
                f"Unexpected binary data size: {len(data)}, "
                f"expected {expected_real} or {expected_complex}"
            )

    def parse_rt_restart(
        self,
        filepath: Union[str, Path],
        n_basis: int,
        n_spin: int = 1,
    ) -> np.ndarray:
        """
        Parse density matrix from NWChem RT-TDDFT restart file.

        Supports two formats:
        1. Text format (newer NWChem): Header with metadata followed by
           space-separated real/imag pairs
        2. Binary format: Fortran unformatted with record markers

        Args:
            filepath: Path to rt_restart file
            n_basis: Number of basis functions
            n_spin: Number of spin channels

        Returns:
            Complex density matrix of shape (n_spin, n_basis, n_basis)
        """
        filepath = Path(filepath)
        filesize = filepath.stat().st_size

        # First, try to detect if it's a text file by reading the header
        with open(filepath, "rb") as f:
            header = f.read(20)

        # Check for text format (starts with "RT-TDDFT")
        if header.startswith(b"RT-TDDFT"):
            return self._parse_rt_restart_text(filepath, n_basis, n_spin)

        # Otherwise, try binary parsing strategies
        return self._parse_rt_restart_binary(filepath, n_basis, n_spin, filesize)

    def _parse_rt_restart_text(
        self,
        filepath: Path,
        n_basis: int,
        n_spin: int,
    ) -> np.ndarray:
        """
        Parse text-format RT-TDDFT restart file.

        Text format:
            RT-TDDFT restart file
            created   <timestamp>
            nmats     <n_spin>
            nbf_ao    <n_basis>
            it        <step>
            t         <time>
            checksum  <value>
            <real1> <imag1> <real2> <imag2> ...

        Args:
            filepath: Path to restart file
            n_basis: Expected number of basis functions
            n_spin: Expected number of spin channels

        Returns:
            Complex density matrix of shape (n_spin, n_basis, n_basis)
        """
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Parse header
        metadata = {}
        data_start_line = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for key-value pairs in header
            if line.startswith("RT-TDDFT"):
                continue
            elif line.startswith("created"):
                continue
            elif line.startswith("nmats"):
                metadata["nmats"] = int(line.split()[1])
            elif line.startswith("nbf_ao"):
                metadata["nbf_ao"] = int(line.split()[1])
            elif line.startswith("it"):
                metadata["it"] = int(line.split()[1])
            elif line.startswith("t "):
                metadata["t"] = float(line.split()[1])
            elif line.startswith("checksum"):
                metadata["checksum"] = float(line.split()[1])
                data_start_line = i + 1
                break

        # Validate metadata
        file_nmats = metadata.get("nmats", 1)
        file_nbf = metadata.get("nbf_ao", n_basis)

        if file_nbf != n_basis:
            raise ValueError(
                f"Basis size mismatch in {filepath}: "
                f"file has nbf_ao={file_nbf}, expected {n_basis}"
            )

        # Parse density matrix data
        # Data is space-separated real/imag pairs, may span multiple lines
        data_text = " ".join(lines[data_start_line:])
        values = [float(x) for x in data_text.split()]

        # Expected: n_spin matrices, each with n_basis^2 complex elements
        # Each complex element = 2 floats (real, imag)
        expected_values = file_nmats * n_basis * n_basis * 2

        if len(values) != expected_values:
            raise ValueError(
                f"Data size mismatch in {filepath}: "
                f"got {len(values)} values, expected {expected_values} "
                f"for nmats={file_nmats}, nbf={n_basis}"
            )

        # Convert to complex array
        # Values are interleaved: real1, imag1, real2, imag2, ...
        real_parts = np.array(values[0::2])
        imag_parts = np.array(values[1::2])
        complex_arr = real_parts + 1j * imag_parts

        # Reshape to (n_spin, n_basis, n_basis)
        # NWChem stores in column-major (Fortran) order
        matrices = complex_arr.reshape(file_nmats, n_basis, n_basis, order='F')

        # If file has fewer spin channels than requested, duplicate
        if file_nmats < n_spin:
            matrices = np.tile(matrices, (n_spin // file_nmats + 1, 1, 1))[:n_spin]

        return matrices[:n_spin]

    def _parse_rt_restart_binary(
        self,
        filepath: Path,
        n_basis: int,
        n_spin: int,
        filesize: int,
    ) -> np.ndarray:
        """
        Parse binary-format RT-TDDFT restart file (Fortran unformatted).

        Args:
            filepath: Path to restart file
            n_basis: Number of basis functions
            n_spin: Number of spin channels
            filesize: Size of file in bytes

        Returns:
            Complex density matrix of shape (n_spin, n_basis, n_basis)
        """
        # Expected size for complex matrix (2 doubles per element)
        matrix_size = n_spin * n_basis * n_basis
        expected_bytes = matrix_size * 16  # complex128 = 16 bytes

        with open(filepath, "rb") as f:
            data = f.read()

        # Strategy 1: Raw complex128 data (no Fortran markers)
        if filesize == expected_bytes:
            arr = np.frombuffer(data, dtype=np.complex128)
            return arr.reshape(n_spin, n_basis, n_basis, order='F')

        # Strategy 2: Fortran unformatted with record markers
        # Each record has: 4-byte marker, data, 4-byte marker
        if filesize == expected_bytes + 8:
            # Single record with markers
            arr = np.frombuffer(data[4:-4], dtype=np.complex128)
            return arr.reshape(n_spin, n_basis, n_basis, order='F')

        # Strategy 3: Multiple records (one per spin)
        if n_spin > 1:
            spin_size = n_basis * n_basis * 16
            record_size = spin_size + 8  # data + 2 markers
            if filesize == n_spin * record_size:
                matrices = []
                offset = 0
                for _ in range(n_spin):
                    arr = np.frombuffer(data[offset+4:offset+4+spin_size], dtype=np.complex128)
                    matrices.append(arr.reshape(n_basis, n_basis, order='F'))
                    offset += record_size
                return np.stack(matrices, axis=0)

        # Strategy 4: Try as flat float64 pairs (real, imag interleaved)
        n_floats = filesize // 8
        if n_floats == 2 * matrix_size:
            arr = np.frombuffer(data, dtype=np.float64)
            complex_arr = arr[::2] + 1j * arr[1::2]
            return complex_arr.reshape(n_spin, n_basis, n_basis, order='F')

        # Strategy 5: With Fortran markers, float64 pairs
        if n_floats == 2 * matrix_size + 2:  # +2 for markers as float64
            arr = np.frombuffer(data[4:-4], dtype=np.float64)
            complex_arr = arr[::2] + 1j * arr[1::2]
            return complex_arr.reshape(n_spin, n_basis, n_basis, order='F')

        raise ValueError(
            f"Cannot parse rt_restart file {filepath}: "
            f"size={filesize}, expected ~{expected_bytes} bytes for "
            f"n_basis={n_basis}, n_spin={n_spin}"
        )

    def find_restart_files(
        self,
        pattern: str,
        max_step: Optional[int] = None,
    ) -> list[tuple[int, Path]]:
        """
        Find all restart files matching a pattern.

        Args:
            pattern: Filename pattern with {:010d} or similar format spec
            max_step: Maximum step number to search for

        Returns:
            List of (step, filepath) tuples, sorted by step
        """
        # Extract prefix and suffix from pattern
        if "{" not in pattern:
            # No format specifier, try direct match
            path = self.base_dir / pattern
            if path.exists():
                return [(0, path)]
            return []

        # Parse pattern
        prefix = pattern.split("{")[0]
        suffix = pattern.split("}")[-1] if "}" in pattern else ""

        results = []
        for f in self.base_dir.iterdir():
            if f.name.startswith(prefix) and f.name.endswith(suffix):
                # Extract step number
                middle = f.name[len(prefix):-len(suffix)] if suffix else f.name[len(prefix):]
                try:
                    step = int(middle)
                    if max_step is None or step <= max_step:
                        results.append((step, f))
                except ValueError:
                    continue

        return sorted(results, key=lambda x: x[0])

    def parse_trajectory_from_record(
        self,
        record: "SimulationRecord",
        geometry_dir: Optional[Union[str, Path]] = None,
        overlap_file: Optional[str] = None,
        n_electrons: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Trajectory:
        """
        Parse a complete trajectory from a SimulationRecord.

        Args:
            record: SimulationRecord with trajectory metadata
            geometry_dir: Directory containing geometry files (default: density_dir)
            overlap_file: Name of overlap matrix file
            n_electrons: Number of electrons (if not auto-detected)
            max_steps: Maximum number of steps to load

        Returns:
            Trajectory object
        """
        from .simulation_index import SimulationRecord

        # Set up directories
        density_dir = record.get_density_dir()
        if geometry_dir is None:
            geometry_dir = density_dir
        else:
            geometry_dir = Path(geometry_dir)

        # Update parser base directory
        self.base_dir = density_dir

        # Try to find and parse geometry
        positions = None
        atomic_numbers = None

        # Try common geometry file locations
        geom_files = [
            geometry_dir / f"{record.molecule}.xyz",
            geometry_dir / "geometry.xyz",
            geometry_dir / f"{record.calc_name}.xyz",
            density_dir / f"{record.molecule}.xyz",
            density_dir / "geometry.xyz",
        ]

        for geom_file in geom_files:
            if geom_file.exists():
                try:
                    positions, atomic_numbers = self.parse_geometry(geom_file.name)
                    break
                except Exception:
                    continue

        if positions is None:
            # Create default geometry based on molecule
            positions, atomic_numbers = self._default_geometry(record.molecule, record.natoms)

        # Auto-detect n_electrons if not provided
        if n_electrons is None:
            n_electrons = self._estimate_n_electrons(record.molecule, atomic_numbers)

        # Parse overlap matrix
        overlap = None

        # First, try the ao_overlap field from the record (numpy file path)
        if hasattr(record, 'ao_overlap') and record.ao_overlap is not None:
            ao_overlap_path = Path(os.path.expanduser(record.ao_overlap))
            if ao_overlap_path.exists():
                try:
                    overlap = np.load(ao_overlap_path)
                except Exception as e:
                    print(f"Warning: Failed to load ao_overlap from {ao_overlap_path}: {e}")

        # Fall back to searching for overlap files in density_dir
        if overlap is None:
            overlap_files = [
                overlap_file,
                "overlap.dat",
                f"{record.calc_name}.overlap",
                f"{record.molecule}_overlap.dat",
            ]

            for of in overlap_files:
                if of is not None:
                    of_path = density_dir / of
                    if of_path.exists():
                        try:
                            overlap = self.parse_overlap_matrix(of)
                            break
                        except Exception:
                            continue

        if overlap is None:
            # Use identity matrix as fallback (orthonormal basis approximation)
            print(f"Warning: No overlap matrix found, using identity for {record.calc_name}")
            overlap = np.eye(record.nbf, dtype=np.float64)

        # Find and parse restart files
        if record.density_pattern is None:
            raise ValueError(f"No density pattern for record: {record}")

        restart_files = self.find_restart_files(record.density_pattern, max_steps)

        if not restart_files:
            raise ValueError(f"No restart files found in {density_dir} with pattern {record.density_pattern}")

        # Parse density matrices
        density_matrices = []
        for step, filepath in restart_files:
            try:
                density = self.parse_rt_restart(filepath, record.nbf, record.n_spin)
                density_matrices.append(density)
            except Exception as e:
                print(f"Warning: Failed to parse {filepath}: {e}")
                break

        if not density_matrices:
            raise ValueError(f"Failed to parse any density matrices from {density_dir}")

        n_steps = len(density_matrices)
        density_stack = np.stack(density_matrices, axis=0)  # (n_steps, n_spin, n_basis, n_basis)

        # Generate field based on field_type
        time_points = np.arange(n_steps) * record.dt

        if record.field_type == "delta":
            field_vectors = generate_delta_kick_field(
                n_steps=n_steps,
                dt=record.dt,
                amplitude=0.0001,  # Small kick
                direction=record.field_polarization,
                kick_step=0,
            )
        elif record.field_type == "gaussian":
            field_vectors = generate_gaussian_pulse_field(
                n_steps=n_steps,
                dt=record.dt,
                amplitude=0.001,
                omega=record.field_freq if record.field_freq > 0 else 0.1,
                direction=record.field_polarization,
            )
        else:
            # Zero field
            field_vectors = np.zeros((n_steps, 3))

        # Create basis info (default)
        basis_info = []
        for i in range(record.nbf):
            atom_idx = i % len(atomic_numbers)
            basis_info.append(
                BasisFunctionInfo(
                    atom_index=atom_idx,
                    atomic_number=int(atomic_numbers[atom_idx]),
                    angular_momentum=0,
                    magnetic_quantum=0,
                )
            )

        return Trajectory(
            molecule=record.molecule,
            positions=positions,
            atomic_numbers=atomic_numbers,
            density_real=density_stack.real,
            density_imag=density_stack.imag,
            field=field_vectors,
            time=time_points,
            overlap=overlap,
            basis_info=basis_info,
            n_electrons=n_electrons,
            dt=record.dt,
            # Extended metadata
            geometry_name=record.geometry,
            field_type=record.field_type,
            field_polarization=record.field_polarization,
            basis_set=record.basis_set,
            xc_functional=record.xc_functional,
        )

    def _default_geometry(self, molecule: str, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate default geometry for common molecules."""
        # Default geometries in Angstroms
        GEOMETRIES = {
            "h2": (np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]), np.array([1, 1])),
            "h2p": (np.array([[0.0, 0.0, 0.0], [1.06, 0.0, 0.0]]), np.array([1, 1])),
            "lih": (np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]), np.array([3, 1])),
            "h2o": (
                np.array([[0.0, 0.0, 0.117], [-0.756, 0.0, -0.468], [0.756, 0.0, -0.468]]),
                np.array([8, 1, 1]),
            ),
            "ch4": (
                np.array([
                    [0.0, 0.0, 0.0],
                    [0.629, 0.629, 0.629],
                    [-0.629, -0.629, 0.629],
                    [-0.629, 0.629, -0.629],
                    [0.629, -0.629, -0.629],
                ]),
                np.array([6, 1, 1, 1, 1]),
            ),
            "nh3": (
                np.array([
                    [0.0, 0.0, 0.116],
                    [0.0, 0.939, -0.269],
                    [0.813, -0.469, -0.269],
                    [-0.813, -0.469, -0.269],
                ]),
                np.array([7, 1, 1, 1]),
            ),
            "c2h4": (
                np.array([
                    [0.0, 0.0, 0.667],
                    [0.0, 0.0, -0.667],
                    [0.0, 0.923, 1.237],
                    [0.0, -0.923, 1.237],
                    [0.0, 0.923, -1.237],
                    [0.0, -0.923, -1.237],
                ]),
                np.array([6, 6, 1, 1, 1, 1]),
            ),
        }

        if molecule.lower() in GEOMETRIES:
            return GEOMETRIES[molecule.lower()]

        # Fallback: create dummy geometry
        positions = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            positions[i, 0] = i * 1.5  # 1.5 Angstrom spacing
        atomic_numbers = np.ones(n_atoms, dtype=np.int32)  # All hydrogen

        return positions, atomic_numbers

    def _estimate_n_electrons(self, molecule: str, atomic_numbers: np.ndarray) -> int:
        """Estimate number of electrons from molecule name or atomic numbers."""
        # Known molecules
        ELECTRONS = {
            "h2": 2,
            "h2p": 1,
            "lih": 4,
            "h2o": 10,
            "ch4": 10,
            "nh3": 10,
            "c2h4": 16,
            "c6h6": 42,
        }

        if molecule.lower() in ELECTRONS:
            return ELECTRONS[molecule.lower()]

        # Sum of atomic numbers (neutral molecule)
        return int(atomic_numbers.sum())

    def parse_field_protocol(
        self,
        filename: str = "field.dat",
        n_steps: Optional[int] = None,
        dt: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse external field protocol.

        Args:
            filename: Name of field protocol file
            n_steps: Number of time steps (if not in file)
            dt: Time step in atomic units

        Returns:
            Tuple of (field_vectors, time_points)
        """
        filepath = self.base_dir / filename

        if filepath.exists():
            with open(filepath, "r") as f:
                lines = f.readlines()

            field_vectors = []
            time_points = []

            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    t = float(parts[0])
                    ex, ey, ez = float(parts[1]), float(parts[2]), float(parts[3])
                    time_points.append(t)
                    field_vectors.append([ex, ey, ez])
                elif len(parts) == 3:
                    # No time column, just field
                    ex, ey, ez = float(parts[0]), float(parts[1]), float(parts[2])
                    field_vectors.append([ex, ey, ez])

            if not time_points and n_steps is not None:
                time_points = [i * dt for i in range(len(field_vectors))]

            return np.array(field_vectors), np.array(time_points)
        else:
            # Generate zero field if file doesn't exist
            if n_steps is None:
                raise ValueError("n_steps required when field file doesn't exist")
            field_vectors = np.zeros((n_steps, 3))
            time_points = np.arange(n_steps) * dt
            return field_vectors, time_points

    def parse_basis_info(
        self,
        filename: str = "basis.dat",
        atomic_numbers: Optional[np.ndarray] = None,
    ) -> list[BasisFunctionInfo]:
        """
        Parse basis function metadata.

        Args:
            filename: Name of basis info file
            atomic_numbers: Atomic numbers array (for Z values)

        Returns:
            List of BasisFunctionInfo objects
        """
        filepath = self.base_dir / filename
        basis_info = []

        if filepath.exists():
            with open(filepath, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    atom_idx = int(parts[0])
                    l = int(parts[1]) if parts[1].isdigit() else self.ANGULAR_MOMENTUM.get(
                        parts[1].lower(), 0
                    )
                    m = int(parts[2]) if len(parts) > 2 else 0

                    z = 0
                    if atomic_numbers is not None and atom_idx < len(atomic_numbers):
                        z = int(atomic_numbers[atom_idx])

                    basis_info.append(
                        BasisFunctionInfo(
                            atom_index=atom_idx,
                            atomic_number=z,
                            angular_momentum=l,
                            magnetic_quantum=m,
                        )
                    )
        else:
            # Generate default basis info if file doesn't exist
            # This is a fallback; proper basis info should come from NWChem
            pass

        return basis_info

    def parse_trajectory(
        self,
        molecule: str,
        n_basis: int,
        n_electrons: int,
        n_spin: int = 1,
        dt: float = 0.2,
        geometry_file: str = "geometry.xyz",
        overlap_file: str = "overlap.dat",
        restart_pattern: str = "density_{:05d}.dat",
        field_file: str = "field.dat",
        basis_file: str = "basis.dat",
    ) -> Trajectory:
        """
        Parse a complete RT-TDDFT trajectory from NWChem output files.

        Args:
            molecule: Molecule identifier
            n_basis: Number of basis functions
            n_electrons: Number of electrons
            n_spin: Number of spin channels
            dt: Time step in atomic units
            geometry_file: Name of geometry XYZ file
            overlap_file: Name of overlap matrix file
            restart_pattern: Pattern for restart files (with step number)
            field_file: Name of field protocol file
            basis_file: Name of basis info file

        Returns:
            Trajectory object
        """
        # Parse geometry
        positions, atomic_numbers = self.parse_geometry(geometry_file)

        # Parse overlap matrix
        overlap = self.parse_overlap_matrix(overlap_file)

        # Parse basis info
        basis_info = self.parse_basis_info(basis_file, atomic_numbers)

        # If basis_info is empty, create default
        if not basis_info:
            for i in range(n_basis):
                # Assign to atoms round-robin (approximate)
                atom_idx = i % len(atomic_numbers)
                basis_info.append(
                    BasisFunctionInfo(
                        atom_index=atom_idx,
                        atomic_number=int(atomic_numbers[atom_idx]),
                        angular_momentum=0,
                        magnetic_quantum=0,
                    )
                )

        # Find all restart files
        density_matrices = []
        step = 0
        while True:
            restart_file = restart_pattern.format(step)
            if not (self.base_dir / restart_file).exists():
                break
            density = self.parse_density_restart(restart_file, n_basis, n_spin)
            density_matrices.append(density)
            step += 1

        if not density_matrices:
            raise ValueError(f"No density restart files found in {self.base_dir}")

        n_steps = len(density_matrices)

        # Stack density matrices
        density_stack = np.stack(density_matrices, axis=0)  # (n_steps, n_spin, n_basis, n_basis)

        # Parse field
        field_vectors, time_points = self.parse_field_protocol(field_file, n_steps, dt)

        # Ensure time points match
        if len(time_points) != n_steps:
            time_points = np.arange(n_steps) * dt
        if len(field_vectors) != n_steps:
            # Pad or truncate field
            if len(field_vectors) < n_steps:
                padding = np.zeros((n_steps - len(field_vectors), 3))
                field_vectors = np.vstack([field_vectors, padding])
            else:
                field_vectors = field_vectors[:n_steps]

        return Trajectory(
            molecule=molecule,
            positions=positions,
            atomic_numbers=atomic_numbers,
            density_real=density_stack.real,
            density_imag=density_stack.imag,
            field=field_vectors,
            time=time_points,
            overlap=overlap,
            basis_info=basis_info,
            n_electrons=n_electrons,
            dt=dt,
        )


def generate_delta_kick_field(
    n_steps: int,
    dt: float,
    amplitude: float = 0.001,
    direction: str = "z",
    kick_step: int = 0,
) -> np.ndarray:
    """
    Generate a delta-kick field protocol.

    Args:
        n_steps: Number of time steps
        dt: Time step in atomic units
        amplitude: Kick amplitude in atomic units
        direction: Kick direction ('x', 'y', or 'z')
        kick_step: Step at which to apply the kick

    Returns:
        Field vectors of shape (n_steps, 3)
    """
    field = np.zeros((n_steps, 3))
    dir_idx = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    field[kick_step, dir_idx] = amplitude / dt  # Delta function approximation
    return field


def generate_gaussian_pulse_field(
    n_steps: int,
    dt: float,
    amplitude: float = 0.001,
    omega: float = 0.1,
    sigma: float = 10.0,
    t0: float = 50.0,
    direction: str = "z",
) -> np.ndarray:
    """
    Generate a Gaussian-enveloped sinusoidal field.

    E(t) = A * exp(-(t-t0)^2 / (2*sigma^2)) * cos(omega*t)

    Args:
        n_steps: Number of time steps
        dt: Time step in atomic units
        amplitude: Peak amplitude in atomic units
        omega: Carrier frequency in atomic units
        sigma: Gaussian width in atomic units
        t0: Center time in atomic units
        direction: Field direction ('x', 'y', or 'z')

    Returns:
        Field vectors of shape (n_steps, 3)
    """
    field = np.zeros((n_steps, 3))
    dir_idx = {"x": 0, "y": 1, "z": 2}[direction.lower()]

    for i in range(n_steps):
        t = i * dt
        envelope = np.exp(-((t - t0) ** 2) / (2 * sigma**2))
        carrier = np.cos(omega * t)
        field[i, dir_idx] = amplitude * envelope * carrier

    return field
