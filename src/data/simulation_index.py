"""
Simulation index for managing RT-TDDFT trajectory metadata.

This module provides classes for loading and querying trajectory metadata
stored in pandas DataFrames, typically exported from simulation management scripts.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Iterator
from dataclasses import dataclass


@dataclass
class SimulationRecord:
    """
    Metadata for a single RT-TDDFT simulation.

    Attributes:
        molecule: Molecule identifier (e.g., 'h2', 'ch4')
        calc_name: NWChem calculation name
        geometry: Geometry identifier (e.g., 'eq', 'pdq')
        n_spin: Number of spin channels (1 or 2)
        field_name: Field identifier (e.g., 'delta_z')
        field_polarization: Field direction ('x', 'y', 'z')
        field_type: Field type ('delta', 'gaussian')
        field_freq: Field frequency (for gaussian pulses)
        tmax: Maximum simulation time (a.u.)
        tsteps: Number of time steps
        dt: Time step (a.u.)
        natoms: Number of atoms
        nbf: Number of basis functions
        basis_set: Basis set name
        xc_functional: Exchange-correlation functional
        mode: Vibrational mode index (0 for equilibrium)
        walltime: Simulation walltime (seconds)
        density_dir: Directory containing density restart files
        density_pattern: Pattern for restart file names
    """

    molecule: str
    calc_name: str
    geometry: str
    n_spin: int
    field_name: str
    field_polarization: str
    field_type: str
    field_freq: float
    tmax: float
    tsteps: int
    dt: float
    natoms: int
    nbf: int
    basis_set: str
    xc_functional: str
    mode: int
    walltime: float
    density_dir: str
    density_pattern: Optional[str]

    @classmethod
    def from_series(cls, row: pd.Series) -> "SimulationRecord":
        """Create SimulationRecord from pandas Series."""
        return cls(
            molecule=str(row["molecule"]),
            calc_name=str(row["calc_name"]),
            geometry=str(row["geometry"]),
            n_spin=int(row["n_spin"]) if pd.notna(row["n_spin"]) else 1,
            field_name=str(row["field_name"]),
            field_polarization=str(row["field_polarization"]),
            field_type=str(row["field_type"]),
            field_freq=float(row["field_freq"]) if pd.notna(row["field_freq"]) else 0.0,
            tmax=float(row["tmax"]) if pd.notna(row["tmax"]) else 0.0,
            tsteps=int(row["tsteps"]) if pd.notna(row["tsteps"]) else 0,
            dt=float(row["dt"]) if pd.notna(row["dt"]) else 0.2,
            natoms=int(row["natoms"]) if pd.notna(row["natoms"]) else 0,
            nbf=int(row["nbf"]) if pd.notna(row["nbf"]) else 0,
            basis_set=str(row["basis_set"]),
            xc_functional=str(row["xc_functional"]),
            mode=int(row["mode"]) if pd.notna(row["mode"]) else 0,
            walltime=float(row["walltime"]) if pd.notna(row["walltime"]) else 0.0,
            density_dir=str(row["density_dir"]),
            density_pattern=str(row["density_pattern"]) if pd.notna(row["density_pattern"]) else None,
        )

    def get_density_dir(self) -> Path:
        """Get expanded density directory path."""
        return Path(os.path.expanduser(self.density_dir))

    def get_restart_path(self, step: int) -> Path:
        """Get path to restart file for a given step."""
        if self.density_pattern is None:
            raise ValueError("No density pattern available")
        filename = self.density_pattern.format(step)
        return self.get_density_dir() / filename

    def has_data(self) -> bool:
        """Check if density data directory exists."""
        return self.get_density_dir().exists()

    def count_restart_files(self) -> int:
        """Count available restart files."""
        if not self.has_data() or self.density_pattern is None:
            return 0

        density_dir = self.get_density_dir()
        # Extract prefix from pattern (before the format specifier)
        prefix = self.density_pattern.split("{")[0]

        count = 0
        for f in density_dir.iterdir():
            if f.name.startswith(prefix):
                count += 1
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "molecule": self.molecule,
            "calc_name": self.calc_name,
            "geometry": self.geometry,
            "n_spin": self.n_spin,
            "field_name": self.field_name,
            "field_polarization": self.field_polarization,
            "field_type": self.field_type,
            "field_freq": self.field_freq,
            "tmax": self.tmax,
            "tsteps": self.tsteps,
            "dt": self.dt,
            "natoms": self.natoms,
            "nbf": self.nbf,
            "basis_set": self.basis_set,
            "xc_functional": self.xc_functional,
            "mode": self.mode,
            "walltime": self.walltime,
            "density_dir": self.density_dir,
            "density_pattern": self.density_pattern,
        }


class SimulationIndex:
    """
    Index of RT-TDDFT simulations loaded from a pandas DataFrame.

    Provides filtering, querying, and iteration over simulation metadata.

    Example:
        >>> index = SimulationIndex.load("rt_simulations.pkl")
        >>> h2_sims = index.filter(molecule="h2")
        >>> for record in h2_sims:
        ...     print(record.molecule, record.nbf)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize from pandas DataFrame.

        Args:
            df: DataFrame with simulation metadata
        """
        self.df = df.copy()
        self._records: Optional[List[SimulationRecord]] = None

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SimulationIndex":
        """
        Load simulation index from pickle file.

        Args:
            filepath: Path to pickled DataFrame

        Returns:
            SimulationIndex instance
        """
        df = pd.read_pickle(filepath)
        return cls(df)

    @classmethod
    def from_csv(cls, filepath: Union[str, Path]) -> "SimulationIndex":
        """
        Load simulation index from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            SimulationIndex instance
        """
        df = pd.read_csv(filepath)
        return cls(df)

    def __len__(self) -> int:
        """Number of simulations in index."""
        return len(self.df)

    def __iter__(self) -> Iterator[SimulationRecord]:
        """Iterate over simulation records."""
        for idx, row in self.df.iterrows():
            yield SimulationRecord.from_series(row)

    def __getitem__(self, idx: int) -> SimulationRecord:
        """Get simulation record by index."""
        return SimulationRecord.from_series(self.df.iloc[idx])

    @property
    def records(self) -> List[SimulationRecord]:
        """Get all records as a list (cached)."""
        if self._records is None:
            self._records = [SimulationRecord.from_series(row) for _, row in self.df.iterrows()]
        return self._records

    @property
    def molecules(self) -> List[str]:
        """Get unique molecule names."""
        return self.df["molecule"].unique().tolist()

    @property
    def geometries(self) -> List[str]:
        """Get unique geometry identifiers."""
        return self.df["geometry"].unique().tolist()

    @property
    def field_types(self) -> List[str]:
        """Get unique field types."""
        return self.df["field_type"].unique().tolist()

    @property
    def basis_sets(self) -> List[str]:
        """Get unique basis sets."""
        return self.df["basis_set"].unique().tolist()

    def filter(
        self,
        molecule: Optional[Union[str, List[str]]] = None,
        geometry: Optional[Union[str, List[str]]] = None,
        field_type: Optional[Union[str, List[str]]] = None,
        field_polarization: Optional[Union[str, List[str]]] = None,
        basis_set: Optional[Union[str, List[str]]] = None,
        min_nbf: Optional[int] = None,
        max_nbf: Optional[int] = None,
        min_tsteps: Optional[int] = None,
        has_data: Optional[bool] = None,
    ) -> "SimulationIndex":
        """
        Filter simulations by criteria.

        Args:
            molecule: Filter by molecule(s)
            geometry: Filter by geometry type(s)
            field_type: Filter by field type(s)
            field_polarization: Filter by field polarization(s)
            basis_set: Filter by basis set(s)
            min_nbf: Minimum number of basis functions
            max_nbf: Maximum number of basis functions
            min_tsteps: Minimum number of time steps
            has_data: Filter by data availability

        Returns:
            New SimulationIndex with filtered data
        """
        mask = pd.Series([True] * len(self.df), index=self.df.index)

        if molecule is not None:
            if isinstance(molecule, str):
                molecule = [molecule]
            mask &= self.df["molecule"].isin(molecule)

        if geometry is not None:
            if isinstance(geometry, str):
                geometry = [geometry]
            mask &= self.df["geometry"].isin(geometry)

        if field_type is not None:
            if isinstance(field_type, str):
                field_type = [field_type]
            mask &= self.df["field_type"].isin(field_type)

        if field_polarization is not None:
            if isinstance(field_polarization, str):
                field_polarization = [field_polarization]
            mask &= self.df["field_polarization"].isin(field_polarization)

        if basis_set is not None:
            if isinstance(basis_set, str):
                basis_set = [basis_set]
            mask &= self.df["basis_set"].isin(basis_set)

        if min_nbf is not None:
            mask &= self.df["nbf"] >= min_nbf

        if max_nbf is not None:
            mask &= self.df["nbf"] <= max_nbf

        if min_tsteps is not None:
            mask &= self.df["tsteps"] >= min_tsteps

        if has_data is not None:
            data_exists = [
                SimulationRecord.from_series(row).has_data()
                for _, row in self.df.iterrows()
            ]
            if has_data:
                mask &= pd.Series(data_exists, index=self.df.index)
            else:
                mask &= ~pd.Series(data_exists, index=self.df.index)

        return SimulationIndex(self.df[mask])

    def group_by_molecule(self) -> Dict[str, "SimulationIndex"]:
        """Group simulations by molecule."""
        return {
            mol: SimulationIndex(group)
            for mol, group in self.df.groupby("molecule")
        }

    def summary(self) -> pd.DataFrame:
        """
        Get summary statistics.

        Returns:
            DataFrame with counts per molecule/field type
        """
        return self.df.groupby(["molecule", "field_type"]).size().unstack(fill_value=0)

    def get_unique_configs(self) -> pd.DataFrame:
        """
        Get unique molecule/basis/geometry configurations.

        Returns:
            DataFrame with unique configurations
        """
        return self.df[["molecule", "basis_set", "geometry", "nbf", "natoms"]].drop_duplicates()

    def to_dataframe(self) -> pd.DataFrame:
        """Get underlying DataFrame."""
        return self.df.copy()

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save index to pickle file.

        Args:
            filepath: Output path
        """
        self.df.to_pickle(filepath)

    def __repr__(self) -> str:
        return f"SimulationIndex({len(self)} simulations, {len(self.molecules)} molecules)"
