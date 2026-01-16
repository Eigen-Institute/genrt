"""
Tests for data loading and processing.

Tests cover HDF5 trajectory I/O, dataset classes, data transforms,
simulation index, and NWChem parsing.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.data.trajectory import Trajectory, BasisFunctionInfo
from src.data.dataset import TrajectoryDataset, UnifiedTrajectoryDataset, collate_variable_basis
from src.data.transforms import (
    GeometryNoiseTransform,
    FieldNoiseTransform,
    RandomRotationTransform,
    ComposeTransforms,
    get_training_transforms,
)
from src.data.simulation_index import SimulationIndex, SimulationRecord
from src.data.nwchem_parser import NWChemParser
from src.utils.complex_tensor import hermitianize, trace_normalize, check_hermiticity


class TestTrajectory:
    """Tests for Trajectory class."""

    def test_trajectory_creation(self, sample_trajectory):
        """Test creating a trajectory."""
        traj = sample_trajectory("h2", n_steps=50)

        assert traj.n_atoms == 2
        assert traj.n_steps == 50
        assert traj.molecule == "h2"
        assert traj.n_basis == 4

    def test_trajectory_properties(self, sample_trajectory):
        """Test trajectory property accessors."""
        traj = sample_trajectory("h2o", n_steps=100)

        assert traj.n_atoms == 3
        assert traj.n_electrons == 10
        assert traj.n_spin == 1

        # Complex density should combine real and imag
        density = traj.density
        assert density.dtype == np.complex128
        assert density.shape == (100, 1, 13, 13)

    def test_trajectory_save_load(self, sample_trajectory, temp_hdf5_file):
        """Test saving and loading trajectory to HDF5."""
        traj_original = sample_trajectory("h2", n_steps=30)

        # Save
        traj_original.save(temp_hdf5_file)
        assert temp_hdf5_file.exists()

        # Load
        traj_loaded = Trajectory.load(temp_hdf5_file)

        # Compare
        assert traj_loaded.molecule == traj_original.molecule
        assert traj_loaded.n_atoms == traj_original.n_atoms
        assert traj_loaded.n_steps == traj_original.n_steps
        assert traj_loaded.n_basis == traj_original.n_basis
        assert traj_loaded.n_electrons == traj_original.n_electrons

        np.testing.assert_allclose(traj_loaded.positions, traj_original.positions)
        np.testing.assert_array_equal(traj_loaded.atomic_numbers, traj_original.atomic_numbers)
        np.testing.assert_allclose(traj_loaded.overlap, traj_original.overlap)
        np.testing.assert_allclose(traj_loaded.density_real, traj_original.density_real)
        np.testing.assert_allclose(traj_loaded.density_imag, traj_original.density_imag)

    def test_trajectory_tensor_conversion(self, sample_trajectory):
        """Test converting trajectory data to PyTorch tensors."""
        traj = sample_trajectory("lih", n_steps=20)

        positions = traj.get_positions_tensor()
        assert isinstance(positions, torch.Tensor)
        assert positions.dtype == torch.float32
        assert positions.shape == (2, 3)

        density = traj.get_density_tensor()
        assert torch.is_complex(density)
        assert density.shape == (20, 1, 11, 11)

    def test_trajectory_slice(self, sample_trajectory):
        """Test slicing trajectory in time."""
        traj = sample_trajectory("h2", n_steps=100)

        sliced = traj.slice_time(10, 50)

        assert sliced.n_steps == 40
        assert sliced.molecule == traj.molecule
        assert sliced.n_atoms == traj.n_atoms
        np.testing.assert_allclose(sliced.positions, traj.positions)

    def test_basis_metadata(self, sample_trajectory):
        """Test basis function metadata extraction."""
        traj = sample_trajectory("h2o", n_steps=10)

        metadata = traj.get_basis_metadata()

        assert "atom_idx" in metadata
        assert "Z" in metadata
        assert "l" in metadata
        assert "m" in metadata

        assert len(metadata["atom_idx"]) == traj.n_basis


class TestTrajectoryDataset:
    """Tests for TrajectoryDataset class."""

    def test_dataset_length(self, sample_trajectory):
        """Test dataset length calculation."""
        traj = sample_trajectory("h2", n_steps=100)
        dataset = TrajectoryDataset(traj, stride=1)

        # n_samples = n_steps - 1 (need current and next)
        assert len(dataset) == 99

    def test_dataset_length_with_stride(self, sample_trajectory):
        """Test dataset length with stride."""
        traj = sample_trajectory("h2", n_steps=100)
        dataset = TrajectoryDataset(traj, stride=5)

        # (100 - 1) // 5 + 1 = 20
        assert len(dataset) == 20

    def test_dataset_getitem(self, sample_trajectory):
        """Test getting a sample from dataset."""
        traj = sample_trajectory("h2", n_steps=50)
        dataset = TrajectoryDataset(traj)

        sample = dataset[0]

        assert "positions" in sample
        assert "atomic_numbers" in sample
        assert "density_current" in sample
        assert "density_next" in sample
        assert "field" in sample
        assert "overlap" in sample
        assert "n_electrons" in sample
        assert "graph" in sample

        assert torch.is_complex(sample["density_current"])
        assert torch.is_complex(sample["density_next"])

    def test_dataset_with_transform(self, sample_trajectory):
        """Test dataset with transform applied."""
        traj = sample_trajectory("h2", n_steps=50)
        transform = GeometryNoiseTransform(noise_std=0.01)
        dataset = TrajectoryDataset(traj, transform=transform)

        sample = dataset[0]

        # Positions should be slightly different from trajectory
        # (though not guaranteed with noise)
        assert "positions" in sample


class TestUnifiedTrajectoryDataset:
    """Tests for UnifiedTrajectoryDataset with multiple molecules."""

    def test_unified_dataset_creation(self, sample_trajectory, temp_directory):
        """Test creating unified dataset from multiple trajectories."""
        # Create and save trajectories
        paths = []
        for mol in ["h2", "lih"]:
            traj = sample_trajectory(mol, n_steps=50)
            path = temp_directory / f"{mol}.h5"
            traj.save(path)
            paths.append(path)

        dataset = UnifiedTrajectoryDataset(paths)

        # Total samples = sum of (n_steps - 1) for each trajectory
        # H2: 49, LiH: 49
        assert len(dataset) == 98

    def test_unified_dataset_variable_basis(self, sample_trajectory, temp_directory):
        """Test that unified dataset handles variable basis sizes."""
        # H2: n_basis=4, LiH: n_basis=11
        paths = []
        for mol in ["h2", "lih"]:
            traj = sample_trajectory(mol, n_steps=30)
            path = temp_directory / f"{mol}.h5"
            traj.save(path)
            paths.append(path)

        dataset = UnifiedTrajectoryDataset(paths)

        # Get samples from different molecules
        sample_h2 = dataset[0]  # First trajectory
        sample_lih = dataset[29 + 10]  # Second trajectory

        assert sample_h2["n_basis"] == 4
        assert sample_lih["n_basis"] == 11


class TestCollateFunction:
    """Tests for batch collation functions."""

    def test_collate_variable_basis(self, sample_trajectory, temp_directory):
        """Test collating samples with variable basis sizes."""
        paths = []
        for mol in ["h2", "lih"]:
            traj = sample_trajectory(mol, n_steps=20)
            path = temp_directory / f"{mol}.h5"
            traj.save(path)
            paths.append(path)

        dataset = UnifiedTrajectoryDataset(paths)

        # Get samples from different molecules
        samples = [dataset[0], dataset[25]]

        batch = collate_variable_basis(samples)

        assert len(batch["positions"]) == 2
        assert len(batch["density_current"]) == 2
        assert batch["field"].shape == (2, 3)

        # Different basis sizes
        assert batch["density_current"][0].shape[-1] == 4  # H2
        assert batch["density_current"][1].shape[-1] == 11  # LiH


class TestTransforms:
    """Tests for data augmentation transforms."""

    def test_geometry_noise_transform(self, sample_trajectory):
        """Test geometry noise transform."""
        traj = sample_trajectory("h2", n_steps=10)
        dataset = TrajectoryDataset(traj)
        sample = dataset[0]

        transform = GeometryNoiseTransform(noise_std=0.1, seed=42)
        transformed = transform(sample.copy())

        # Positions should be different
        diff = (transformed["positions"] - sample["positions"]).abs().sum()
        assert diff > 0.01

    def test_field_noise_transform(self, sample_trajectory):
        """Test field noise transform."""
        traj = sample_trajectory("h2", n_steps=10)
        dataset = TrajectoryDataset(traj)
        sample = dataset[0]

        # Make field non-zero
        sample["field"] = torch.tensor([0.001, 0.0, 0.0])

        transform = FieldNoiseTransform(noise_std=0.0001, seed=42)
        transformed = transform(sample.copy())

        # Field should be slightly different
        diff = (transformed["field"] - sample["field"]).abs().sum()
        assert diff > 1e-6

    def test_random_rotation_transform(self, sample_trajectory):
        """Test random rotation transform."""
        traj = sample_trajectory("h2", n_steps=10)
        dataset = TrajectoryDataset(traj)
        sample = dataset[0]

        transform = RandomRotationTransform(seed=42)
        transformed = transform(sample.copy())

        # Check rotation matrix was stored
        assert "_rotation_matrix" in transformed

        # Positions should be different
        diff = (transformed["positions"] - sample["positions"]).abs().sum()
        assert diff > 0.01

        # But distances should be preserved
        orig_dist = torch.norm(sample["positions"][0] - sample["positions"][1])
        trans_dist = torch.norm(transformed["positions"][0] - transformed["positions"][1])
        assert torch.allclose(orig_dist, trans_dist, atol=1e-5)

    def test_compose_transforms(self, sample_trajectory):
        """Test composing multiple transforms."""
        traj = sample_trajectory("h2", n_steps=10)
        dataset = TrajectoryDataset(traj)
        sample = dataset[0]

        transform = ComposeTransforms([
            GeometryNoiseTransform(noise_std=0.05),
            FieldNoiseTransform(noise_std=0.001),
        ])

        transformed = transform(sample.copy())

        assert "positions" in transformed
        assert "field" in transformed

    def test_training_transforms_factory(self):
        """Test training transforms factory function."""
        transform = get_training_transforms(
            geometry_noise=0.05,
            field_noise=0.001,
            use_rotation=True,
        )

        assert len(transform.transforms) == 3


class TestComplexTensorUtilities:
    """Tests for complex tensor utility functions."""

    def test_hermitianize(self):
        """Test Hermitianization."""
        n = 5
        A = torch.randn(n, n) + 1j * torch.randn(n, n)
        H = hermitianize(A)

        # Check Hermitian: H == H^dagger
        diff = (H - H.conj().T).abs().max()
        assert diff < 1e-6

    def test_trace_normalize(self):
        """Test trace normalization."""
        n = 5
        n_elec = 4

        # Create Hermitian matrix
        A = torch.randn(n, n) + 1j * torch.randn(n, n)
        rho = hermitianize(A)

        # Identity overlap for simplicity (complex for einsum compatibility)
        S = torch.eye(n, dtype=torch.complex64)

        rho_normalized = trace_normalize(rho, S, n_elec)

        # Check trace
        trace = torch.einsum("ij,ji->", rho_normalized, S).real
        assert torch.allclose(trace, torch.tensor(float(n_elec)), atol=1e-5)

    def test_check_hermiticity(self):
        """Test Hermiticity check."""
        n = 5

        # Hermitian matrix
        A = torch.randn(n, n) + 1j * torch.randn(n, n)
        H = hermitianize(A)
        error_H = check_hermiticity(H)
        assert error_H < 1e-6

        # Non-Hermitian matrix
        B = torch.randn(n, n) + 1j * torch.randn(n, n)
        error_B = check_hermiticity(B)
        assert error_B > 1e-3  # Should have non-zero error


class TestSimulationRecord:
    """Tests for SimulationRecord dataclass."""

    def test_record_from_series(self, sample_simulation_dataframe):
        """Test creating SimulationRecord from pandas Series."""
        row = sample_simulation_dataframe.iloc[0]
        record = SimulationRecord.from_series(row)

        assert record.molecule == "h2"
        assert record.calc_name == "h2_eq_delta_z"
        assert record.geometry == "eq"
        assert record.n_spin == 1
        assert record.nbf == 4
        assert record.tsteps == 500
        assert record.dt == 0.2
        assert record.basis_set == "cc-pvtz"
        assert record.xc_functional == "pbe0"

    def test_record_get_density_dir(self, sample_simulation_dataframe):
        """Test getting density directory path."""
        row = sample_simulation_dataframe.iloc[0]
        record = SimulationRecord.from_series(row)

        density_dir = record.get_density_dir()
        assert isinstance(density_dir, Path)
        assert str(density_dir) == "/nonexistent/h2_eq"

    def test_record_get_restart_path(self, sample_simulation_dataframe):
        """Test getting restart file path."""
        row = sample_simulation_dataframe.iloc[0]
        record = SimulationRecord.from_series(row)

        restart_path = record.get_restart_path(1000)
        assert "h2_rt.rt_restart.0000001000" in str(restart_path)

    def test_record_has_data(self, sample_simulation_dataframe):
        """Test checking data availability."""
        row = sample_simulation_dataframe.iloc[0]
        record = SimulationRecord.from_series(row)

        # Directory doesn't exist, should return False
        assert record.has_data() is False

    def test_record_to_dict(self, sample_simulation_dataframe):
        """Test converting record to dictionary."""
        row = sample_simulation_dataframe.iloc[0]
        record = SimulationRecord.from_series(row)

        d = record.to_dict()
        assert d["molecule"] == "h2"
        assert d["nbf"] == 4
        assert d["field_type"] == "delta"


class TestSimulationIndex:
    """Tests for SimulationIndex class."""

    def test_index_creation(self, sample_simulation_index):
        """Test creating SimulationIndex."""
        assert len(sample_simulation_index) == 5

    def test_index_load_from_pickle(self, sample_simulation_index_file):
        """Test loading SimulationIndex from pickle file."""
        index = SimulationIndex.load(sample_simulation_index_file)
        assert len(index) == 5

    def test_index_molecules_property(self, sample_simulation_index):
        """Test molecules property."""
        molecules = sample_simulation_index.molecules
        assert set(molecules) == {"h2", "h2p", "lih", "h2o"}

    def test_index_geometries_property(self, sample_simulation_index):
        """Test geometries property."""
        geometries = sample_simulation_index.geometries
        assert set(geometries) == {"eq", "pdq"}

    def test_index_field_types_property(self, sample_simulation_index):
        """Test field_types property."""
        field_types = sample_simulation_index.field_types
        assert field_types == ["delta"]

    def test_index_iteration(self, sample_simulation_index):
        """Test iterating over index."""
        records = list(sample_simulation_index)
        assert len(records) == 5
        assert all(isinstance(r, SimulationRecord) for r in records)

    def test_index_getitem(self, sample_simulation_index):
        """Test getting record by index."""
        record = sample_simulation_index[0]
        assert isinstance(record, SimulationRecord)
        assert record.molecule == "h2"

    def test_index_records_property(self, sample_simulation_index):
        """Test records property (cached)."""
        records1 = sample_simulation_index.records
        records2 = sample_simulation_index.records
        assert records1 is records2  # Same object (cached)
        assert len(records1) == 5

    def test_index_filter_molecule(self, sample_simulation_index):
        """Test filtering by molecule."""
        h2_index = sample_simulation_index.filter(molecule="h2")
        assert len(h2_index) == 2  # Two h2 entries

        h2p_index = sample_simulation_index.filter(molecule="h2p")
        assert len(h2p_index) == 1

    def test_index_filter_multiple_molecules(self, sample_simulation_index):
        """Test filtering by multiple molecules."""
        filtered = sample_simulation_index.filter(molecule=["h2", "lih"])
        assert len(filtered) == 3  # 2 h2 + 1 lih

    def test_index_filter_geometry(self, sample_simulation_index):
        """Test filtering by geometry."""
        eq_index = sample_simulation_index.filter(geometry="eq")
        assert len(eq_index) == 4  # All except h2_pdq

        pdq_index = sample_simulation_index.filter(geometry="pdq")
        assert len(pdq_index) == 1

    def test_index_filter_field_polarization(self, sample_simulation_index):
        """Test filtering by field polarization."""
        z_index = sample_simulation_index.filter(field_polarization="z")
        assert len(z_index) == 4

        x_index = sample_simulation_index.filter(field_polarization="x")
        assert len(x_index) == 1

    def test_index_filter_nbf_range(self, sample_simulation_index):
        """Test filtering by basis function range."""
        # min_nbf only
        filtered = sample_simulation_index.filter(min_nbf=10)
        assert len(filtered) == 2  # lih (11) and h2o (13)

        # max_nbf only
        filtered = sample_simulation_index.filter(max_nbf=4)
        assert len(filtered) == 3  # h2 (4) x2 and h2p (2)

        # Both
        filtered = sample_simulation_index.filter(min_nbf=4, max_nbf=11)
        assert len(filtered) == 3  # h2 (4) x2 and lih (11)

    def test_index_filter_combined(self, sample_simulation_index):
        """Test combining multiple filters."""
        filtered = sample_simulation_index.filter(
            molecule="h2",
            geometry="eq",
            field_polarization="z"
        )
        assert len(filtered) == 1
        assert filtered[0].calc_name == "h2_eq_delta_z"

    def test_index_filter_has_data(self, sample_simulation_index):
        """Test filtering by data availability."""
        # All directories are nonexistent
        with_data = sample_simulation_index.filter(has_data=True)
        assert len(with_data) == 0

        without_data = sample_simulation_index.filter(has_data=False)
        assert len(without_data) == 5

    def test_index_group_by_molecule(self, sample_simulation_index):
        """Test grouping by molecule."""
        groups = sample_simulation_index.group_by_molecule()

        assert "h2" in groups
        assert "h2p" in groups
        assert "lih" in groups
        assert "h2o" in groups

        assert len(groups["h2"]) == 2
        assert len(groups["h2p"]) == 1

    def test_index_summary(self, sample_simulation_index):
        """Test summary statistics."""
        summary = sample_simulation_index.summary()

        # Should be a DataFrame with molecules as rows, field_types as columns
        assert "delta" in summary.columns
        assert summary.loc["h2", "delta"] == 2

    def test_index_get_unique_configs(self, sample_simulation_index):
        """Test getting unique configurations."""
        configs = sample_simulation_index.get_unique_configs()

        assert "molecule" in configs.columns
        assert "nbf" in configs.columns
        assert len(configs) <= 5  # At most 5 unique configs

    def test_index_to_dataframe(self, sample_simulation_index):
        """Test converting to DataFrame."""
        df = sample_simulation_index.to_dataframe()

        assert len(df) == 5
        assert "molecule" in df.columns
        assert "nbf" in df.columns

    def test_index_save_load(self, sample_simulation_index, temp_directory):
        """Test saving and loading index."""
        filepath = temp_directory / "saved_index.pkl"

        sample_simulation_index.save(filepath)
        assert filepath.exists()

        loaded = SimulationIndex.load(filepath)
        assert len(loaded) == len(sample_simulation_index)

    def test_index_repr(self, sample_simulation_index):
        """Test string representation."""
        repr_str = repr(sample_simulation_index)
        assert "SimulationIndex" in repr_str
        assert "5 simulations" in repr_str


class TestNWChemParserRtRestart:
    """Tests for NWChem rt_restart file parsing."""

    def test_parser_init_no_base_dir(self):
        """Test parser can be initialized without base_dir."""
        parser = NWChemParser()
        assert parser.base_dir == Path(".")

    def test_parser_init_with_base_dir(self, temp_directory):
        """Test parser initialization with base_dir."""
        parser = NWChemParser(temp_directory)
        assert parser.base_dir == temp_directory

    def test_parse_rt_restart_single_spin(self, sample_rt_restart_file):
        """Test parsing single-spin rt_restart file."""
        filepath, expected_density = sample_rt_restart_file(n_basis=4, n_spin=1)

        parser = NWChemParser()
        parsed = parser.parse_rt_restart(filepath, n_basis=4, n_spin=1)

        assert parsed.shape == (1, 4, 4)
        assert parsed.dtype == np.complex128

        # Check values match
        np.testing.assert_allclose(parsed, expected_density, rtol=1e-10)

    def test_parse_rt_restart_two_spin(self, sample_rt_restart_file):
        """Test parsing two-spin (unrestricted) rt_restart file."""
        filepath, expected_density = sample_rt_restart_file(n_basis=6, n_spin=2)

        parser = NWChemParser()
        parsed = parser.parse_rt_restart(filepath, n_basis=6, n_spin=2)

        assert parsed.shape == (2, 6, 6)
        np.testing.assert_allclose(parsed, expected_density, rtol=1e-10)

    def test_parse_rt_restart_large_basis(self, sample_rt_restart_file):
        """Test parsing rt_restart with larger basis set."""
        filepath, expected_density = sample_rt_restart_file(n_basis=17, n_spin=1)

        parser = NWChemParser()
        parsed = parser.parse_rt_restart(filepath, n_basis=17, n_spin=1)

        assert parsed.shape == (1, 17, 17)
        np.testing.assert_allclose(parsed, expected_density, rtol=1e-10)

    def test_parse_rt_restart_hermitian(self, sample_rt_restart_file):
        """Test that parsed density is Hermitian."""
        filepath, _ = sample_rt_restart_file(n_basis=8, n_spin=1)

        parser = NWChemParser()
        parsed = parser.parse_rt_restart(filepath, n_basis=8, n_spin=1)

        # Check Hermiticity: ρ = ρ†
        diff = np.abs(parsed - parsed.conj().transpose(0, 2, 1)).max()
        assert diff < 1e-10

    def test_find_restart_files(self, sample_rt_restart_trajectory):
        """Test finding restart files matching pattern."""
        filepaths, _ = sample_rt_restart_trajectory(n_basis=4, n_steps=5)

        parser = NWChemParser(filepaths[0].parent)
        pattern = "mol_rt.rt_restart.{:010d}"

        found = parser.find_restart_files(pattern)

        assert len(found) == 5
        # Returns list of (step, path) tuples, should be sorted by step
        steps = [step for step, path in found]
        assert steps == sorted(steps)
        assert steps == [0, 1, 2, 3, 4]

    def test_find_restart_files_empty(self, temp_directory):
        """Test finding restart files when none exist."""
        parser = NWChemParser(temp_directory)
        found = parser.find_restart_files("nonexistent.{:010d}")
        assert len(found) == 0

    def test_parse_multiple_restart_files(self, sample_rt_restart_trajectory):
        """Test parsing multiple restart files as trajectory."""
        filepaths, expected_densities = sample_rt_restart_trajectory(
            n_basis=4, n_steps=10
        )

        parser = NWChemParser()

        parsed_densities = []
        for fp in filepaths:
            density = parser.parse_rt_restart(fp, n_basis=4, n_spin=1)
            parsed_densities.append(density)

        parsed_stack = np.stack(parsed_densities, axis=0)

        assert parsed_stack.shape == (10, 1, 4, 4)
        np.testing.assert_allclose(parsed_stack, expected_densities, rtol=1e-10)

    def test_default_geometry_h2(self):
        """Test default geometry for H2."""
        parser = NWChemParser()
        positions, atomic_numbers = parser._default_geometry("h2", n_atoms=2)

        assert positions.shape == (2, 3)
        assert len(atomic_numbers) == 2
        assert all(z == 1 for z in atomic_numbers)  # Both hydrogen

    def test_default_geometry_h2o(self):
        """Test default geometry for H2O."""
        parser = NWChemParser()
        positions, atomic_numbers = parser._default_geometry("h2o", n_atoms=3)

        assert positions.shape == (3, 3)
        assert len(atomic_numbers) == 3
        assert atomic_numbers[0] == 8  # Oxygen
        assert atomic_numbers[1] == 1  # Hydrogen
        assert atomic_numbers[2] == 1  # Hydrogen

    def test_default_geometry_ch4(self):
        """Test default geometry for CH4."""
        parser = NWChemParser()
        positions, atomic_numbers = parser._default_geometry("ch4", n_atoms=5)

        assert positions.shape == (5, 3)
        assert len(atomic_numbers) == 5
        assert atomic_numbers[0] == 6  # Carbon
        assert sum(z == 1 for z in atomic_numbers) == 4  # Four hydrogens

    def test_default_geometry_unknown(self):
        """Test default geometry for unknown molecule falls back to linear."""
        parser = NWChemParser()
        positions, atomic_numbers = parser._default_geometry("unknown_mol", n_atoms=2)

        # Unknown molecules get a default linear geometry (1.5 A spacing)
        assert positions.shape == (2, 3)
        assert len(atomic_numbers) == 2
        # Default uses hydrogen (Z=1) for all atoms
        assert all(z == 1 for z in atomic_numbers)
        # Check positions are along x-axis with 1.5 A spacing
        assert positions[0, 0] == 0.0
        assert positions[1, 0] == 1.5

    def test_estimate_n_electrons(self):
        """Test electron count estimation."""
        parser = NWChemParser()

        # Test with known molecules (atomic_numbers not used)
        assert parser._estimate_n_electrons("h2", np.array([1, 1])) == 2
        assert parser._estimate_n_electrons("h2p", np.array([1, 1])) == 1
        assert parser._estimate_n_electrons("lih", np.array([3, 1])) == 4
        assert parser._estimate_n_electrons("h2o", np.array([8, 1, 1])) == 10
        assert parser._estimate_n_electrons("ch4", np.array([6, 1, 1, 1, 1])) == 10

        # Test fallback to atomic number sum for unknown molecule
        assert parser._estimate_n_electrons("unknown", np.array([6, 1, 1])) == 8


class TestTrajectoryMetadataFields:
    """Tests for extended trajectory metadata fields."""

    def test_trajectory_with_metadata(self, sample_trajectory):
        """Test trajectory creation with extended metadata."""
        traj = sample_trajectory("h2", n_steps=10)

        # Add metadata
        traj_with_meta = Trajectory(
            molecule=traj.molecule,
            positions=traj.positions,
            atomic_numbers=traj.atomic_numbers,
            density_real=traj.density_real,
            density_imag=traj.density_imag,
            field=traj.field,
            time=traj.time,
            overlap=traj.overlap,
            basis_info=traj.basis_info,
            n_electrons=traj.n_electrons,
            dt=traj.dt,
            geometry_name="eq",
            field_type="delta",
            field_polarization="z",
            basis_set="cc-pvtz",
            xc_functional="pbe0",
        )

        assert traj_with_meta.geometry_name == "eq"
        assert traj_with_meta.field_type == "delta"
        assert traj_with_meta.field_polarization == "z"
        assert traj_with_meta.basis_set == "cc-pvtz"
        assert traj_with_meta.xc_functional == "pbe0"

    def test_trajectory_metadata_save_load(self, sample_trajectory, temp_hdf5_file):
        """Test saving and loading trajectory with metadata."""
        traj = sample_trajectory("h2", n_steps=10)

        # Create trajectory with metadata
        traj_with_meta = Trajectory(
            molecule=traj.molecule,
            positions=traj.positions,
            atomic_numbers=traj.atomic_numbers,
            density_real=traj.density_real,
            density_imag=traj.density_imag,
            field=traj.field,
            time=traj.time,
            overlap=traj.overlap,
            basis_info=traj.basis_info,
            n_electrons=traj.n_electrons,
            dt=traj.dt,
            geometry_name="pdq",
            field_type="gaussian",
            field_polarization="x",
            basis_set="6-31g",
            xc_functional="b3lyp",
        )

        # Save and load
        traj_with_meta.save(temp_hdf5_file)
        loaded = Trajectory.load(temp_hdf5_file)

        assert loaded.geometry_name == "pdq"
        assert loaded.field_type == "gaussian"
        assert loaded.field_polarization == "x"
        assert loaded.basis_set == "6-31g"
        assert loaded.xc_functional == "b3lyp"

    def test_trajectory_metadata_defaults_none(self, sample_trajectory, temp_hdf5_file):
        """Test that metadata defaults to None."""
        traj = sample_trajectory("h2", n_steps=10)

        assert traj.geometry_name is None
        assert traj.field_type is None
        assert traj.field_polarization is None
        assert traj.basis_set is None
        assert traj.xc_functional is None

        # Save and load preserves None
        traj.save(temp_hdf5_file)
        loaded = Trajectory.load(temp_hdf5_file)

        assert loaded.geometry_name is None
        assert loaded.field_type is None

    def test_trajectory_slice_preserves_metadata(self, sample_trajectory):
        """Test that slicing preserves metadata."""
        traj = sample_trajectory("h2", n_steps=50)

        # Create trajectory with metadata
        traj_with_meta = Trajectory(
            molecule=traj.molecule,
            positions=traj.positions,
            atomic_numbers=traj.atomic_numbers,
            density_real=traj.density_real,
            density_imag=traj.density_imag,
            field=traj.field,
            time=traj.time,
            overlap=traj.overlap,
            basis_info=traj.basis_info,
            n_electrons=traj.n_electrons,
            dt=traj.dt,
            geometry_name="eq",
            field_type="delta",
            field_polarization="z",
            basis_set="cc-pvtz",
            xc_functional="pbe0",
        )

        sliced = traj_with_meta.slice_time(10, 30)

        assert sliced.n_steps == 20
        assert sliced.geometry_name == "eq"
        assert sliced.field_type == "delta"
        assert sliced.field_polarization == "z"
        assert sliced.basis_set == "cc-pvtz"
        assert sliced.xc_functional == "pbe0"
