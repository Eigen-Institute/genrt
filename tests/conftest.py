"""
Pytest fixtures for genrt tests.

Provides common test data including sample molecules, geometries,
and density matrices for testing.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from src.data.trajectory import Trajectory, BasisFunctionInfo
from src.data.simulation_index import SimulationIndex, SimulationRecord
from src.utils.graph import build_molecular_graph
import pandas as pd


# Sample molecule geometries (in Angstroms)
MOLECULES = {
    "h2": {
        "positions": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        "atomic_numbers": np.array([1, 1]),
        "n_electrons": 2,
        "n_basis": 4,  # STO-3G: 1s on each H, doubled for spin
    },
    "h2_plus": {
        "positions": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.06]]),
        "atomic_numbers": np.array([1, 1]),
        "n_electrons": 1,
        "n_basis": 2,  # Minimal basis
    },
    "lih": {
        "positions": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]]),
        "atomic_numbers": np.array([3, 1]),
        "n_electrons": 4,
        "n_basis": 11,  # STO-3G: Li(1s,2s,2px,2py,2pz) + H(1s,...)
    },
    "h2o": {
        "positions": np.array([
            [0.0, 0.0, 0.117],      # O
            [0.0, 0.757, -0.469],   # H
            [0.0, -0.757, -0.469],  # H
        ]),
        "atomic_numbers": np.array([8, 1, 1]),
        "n_electrons": 10,
        "n_basis": 13,
    },
}


@pytest.fixture
def h2_geometry():
    """H2 molecule geometry."""
    return {
        "positions": torch.tensor(MOLECULES["h2"]["positions"], dtype=torch.float32),
        "atomic_numbers": torch.tensor(MOLECULES["h2"]["atomic_numbers"], dtype=torch.long),
    }


@pytest.fixture
def h2_plus_geometry():
    """H2+ molecule geometry."""
    return {
        "positions": torch.tensor(MOLECULES["h2_plus"]["positions"], dtype=torch.float32),
        "atomic_numbers": torch.tensor(MOLECULES["h2_plus"]["atomic_numbers"], dtype=torch.long),
    }


@pytest.fixture
def lih_geometry():
    """LiH molecule geometry."""
    return {
        "positions": torch.tensor(MOLECULES["lih"]["positions"], dtype=torch.float32),
        "atomic_numbers": torch.tensor(MOLECULES["lih"]["atomic_numbers"], dtype=torch.long),
    }


@pytest.fixture
def h2o_geometry():
    """H2O molecule geometry."""
    return {
        "positions": torch.tensor(MOLECULES["h2o"]["positions"], dtype=torch.float32),
        "atomic_numbers": torch.tensor(MOLECULES["h2o"]["atomic_numbers"], dtype=torch.long),
    }


@pytest.fixture
def h2_graph(h2_geometry):
    """Molecular graph for H2."""
    return build_molecular_graph(
        h2_geometry["positions"],
        h2_geometry["atomic_numbers"],
        cutoff=5.0,
    )


@pytest.fixture
def h2o_graph(h2o_geometry):
    """Molecular graph for H2O."""
    return build_molecular_graph(
        h2o_geometry["positions"],
        h2o_geometry["atomic_numbers"],
        cutoff=5.0,
    )


@pytest.fixture
def random_density_matrix():
    """Generate a random valid density matrix."""
    def _make_density(n_basis: int, n_electrons: int, n_spin: int = 1):
        # Create random Hermitian matrix
        A = np.random.randn(n_spin, n_basis, n_basis) + 1j * np.random.randn(n_spin, n_basis, n_basis)
        rho = 0.5 * (A + A.conj().transpose(0, 2, 1))

        # Make it positive semi-definite via eigenvalue clipping
        for s in range(n_spin):
            eigvals, eigvecs = np.linalg.eigh(rho[s])
            eigvals = np.clip(eigvals, 0, None)
            rho[s] = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

        # Normalize trace (assuming identity overlap for simplicity)
        for s in range(n_spin):
            trace = np.trace(rho[s]).real
            if trace > 0:
                rho[s] = rho[s] * (n_electrons / n_spin / trace)

        return rho

    return _make_density


@pytest.fixture
def sample_overlap_matrix():
    """Generate a sample overlap matrix."""
    def _make_overlap(n_basis: int):
        # Create positive definite matrix
        A = np.random.randn(n_basis, n_basis)
        S = A @ A.T + np.eye(n_basis) * n_basis
        # Normalize to have 1s on diagonal
        d = np.sqrt(np.diag(S))
        S = S / np.outer(d, d)
        return S

    return _make_overlap


@pytest.fixture
def sample_trajectory(random_density_matrix, sample_overlap_matrix):
    """Generate a sample trajectory for testing."""
    def _make_trajectory(
        molecule: str = "h2",
        n_steps: int = 100,
        dt: float = 0.2,
    ):
        mol_data = MOLECULES[molecule]
        n_basis = mol_data["n_basis"]
        n_electrons = mol_data["n_electrons"]
        n_atoms = len(mol_data["atomic_numbers"])

        # Generate density matrices for each step
        density_list = []
        for _ in range(n_steps):
            rho = random_density_matrix(n_basis, n_electrons, n_spin=1)
            density_list.append(rho)

        density_stack = np.stack(density_list, axis=0)

        # Generate field (delta kick at t=0)
        field = np.zeros((n_steps, 3))
        field[0, 2] = 0.001  # Z-kick

        # Time points
        time = np.arange(n_steps) * dt

        # Overlap matrix
        overlap = sample_overlap_matrix(n_basis)

        # Basis info (simplified)
        basis_info = []
        for i in range(n_basis):
            bf = BasisFunctionInfo(
                atom_index=i % n_atoms,
                atomic_number=int(mol_data["atomic_numbers"][i % n_atoms]),
                angular_momentum=0,
                magnetic_quantum=0,
            )
            basis_info.append(bf)

        return Trajectory(
            molecule=molecule,
            positions=mol_data["positions"].copy(),
            atomic_numbers=mol_data["atomic_numbers"].copy(),
            density_real=density_stack.real,
            density_imag=density_stack.imag,
            field=field,
            time=time,
            overlap=overlap,
            basis_info=basis_info,
            n_electrons=n_electrons,
            dt=dt,
        )

    return _make_trajectory


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = Path(f.name)
    yield filepath
    # Cleanup
    if filepath.exists():
        filepath.unlink()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def random_rotation_matrix():
    """Generate a random SO(3) rotation matrix."""
    def _make_rotation():
        # Random axis
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # Random angle
        angle = np.random.uniform(0, 2 * np.pi)

        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return torch.tensor(R, dtype=torch.float32)

    return _make_rotation


@pytest.fixture(scope="session")
def device():
    """Get device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_simulation_dataframe():
    """Create a sample simulation metadata DataFrame."""
    data = {
        "molecule": ["h2", "h2", "h2p", "lih", "h2o"],
        "calc_name": ["h2_eq_delta_z", "h2_pdq_delta_z", "h2p_eq_delta_z", "lih_eq_delta_z", "h2o_eq_delta_x"],
        "geometry": ["eq", "pdq", "eq", "eq", "eq"],
        "n_spin": [1, 1, 1, 1, 1],
        "field_name": ["delta_z", "delta_z", "delta_z", "delta_z", "delta_x"],
        "field_polarization": ["z", "z", "z", "z", "x"],
        "field_type": ["delta", "delta", "delta", "delta", "delta"],
        "field_freq": [0.0, 0.0, 0.0, 0.0, 0.0],
        "tmax": [100.0, 100.0, 100.0, 100.0, 100.0],
        "tsteps": [500, 500, 500, 500, 500],
        "dt": [0.2, 0.2, 0.2, 0.2, 0.2],
        "natoms": [2, 2, 2, 2, 3],
        "nbf": [4, 4, 2, 11, 13],
        "basis_set": ["cc-pvtz", "cc-pvtz", "cc-pvtz", "cc-pvtz", "cc-pvtz"],
        "xc_functional": ["pbe0", "pbe0", "pbe0", "pbe0", "pbe0"],
        "mode": [0, 1, 0, 0, 0],
        "walltime": [100.0, 120.0, 80.0, 150.0, 200.0],
        "density_dir": ["/nonexistent/h2_eq", "/nonexistent/h2_pdq", "/nonexistent/h2p_eq",
                       "/nonexistent/lih_eq", "/nonexistent/h2o_eq"],
        "density_pattern": ["h2_rt.rt_restart.{:010d}", "h2_rt.rt_restart.{:010d}",
                          "h2p_rt.rt_restart.{:010d}", "lih_rt.rt_restart.{:010d}",
                          "h2o_rt.rt_restart.{:010d}"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_simulation_index(sample_simulation_dataframe):
    """Create a sample SimulationIndex."""
    return SimulationIndex(sample_simulation_dataframe)


@pytest.fixture
def sample_simulation_index_file(sample_simulation_dataframe, temp_directory):
    """Create a sample simulation index pickle file."""
    filepath = temp_directory / "test_simulations.pkl"
    sample_simulation_dataframe.to_pickle(filepath)
    return filepath


@pytest.fixture
def sample_rt_restart_file(temp_directory):
    """Create a sample NWChem rt_restart binary file."""
    def _create_restart(n_basis: int, n_spin: int = 1, filename: str = "test.rt_restart"):
        """
        Create a mock rt_restart file in Fortran unformatted format.

        Format: For each spin channel:
            - 4-byte record marker (size of data)
            - complex128 density matrix in Fortran column-major byte order
            - 4-byte record marker (size of data)

        Fortran stores column-major, so we transpose before tobytes to mimic
        reading columns as rows in the byte stream.
        """
        filepath = temp_directory / filename

        # Create random density matrix
        density = np.random.randn(n_spin, n_basis, n_basis) + \
                  1j * np.random.randn(n_spin, n_basis, n_basis)
        # Make Hermitian
        density = 0.5 * (density + density.conj().transpose(0, 2, 1))

        with open(filepath, 'wb') as f:
            for s in range(n_spin):
                # Fortran writes column-major: transpose to get correct byte order
                # When read and reshaped with order='F', this gives back the original
                data = density[s].T.astype(np.complex128).tobytes()
                record_size = len(data)
                # Write Fortran record: marker, data, marker
                f.write(np.array([record_size], dtype=np.int32).tobytes())
                f.write(data)
                f.write(np.array([record_size], dtype=np.int32).tobytes())

        return filepath, density

    return _create_restart


@pytest.fixture
def sample_rt_restart_trajectory(temp_directory):
    """Create a sequence of rt_restart files for trajectory testing."""
    def _create_trajectory(n_basis: int, n_steps: int, n_spin: int = 1, prefix: str = "mol_rt.rt_restart"):
        """Create multiple restart files simulating a trajectory."""
        densities = []
        filepaths = []

        for step in range(n_steps):
            filename = f"{prefix}.{step:010d}"
            filepath = temp_directory / filename

            # Create density with small time evolution
            density = np.random.randn(n_spin, n_basis, n_basis) + \
                      1j * np.random.randn(n_spin, n_basis, n_basis)
            density = 0.5 * (density + density.conj().transpose(0, 2, 1))

            with open(filepath, 'wb') as f:
                for s in range(n_spin):
                    # Fortran writes column-major: transpose to get correct byte order
                    data = density[s].T.astype(np.complex128).tobytes()
                    record_size = len(data)
                    f.write(np.array([record_size], dtype=np.int32).tobytes())
                    f.write(data)
                    f.write(np.array([record_size], dtype=np.int32).tobytes())

            densities.append(density)
            filepaths.append(filepath)

        return filepaths, np.stack(densities, axis=0)

    return _create_trajectory
