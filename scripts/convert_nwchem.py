#!/usr/bin/env python
"""
Convert NWChem RT-TDDFT output files to HDF5 trajectories.

Usage:
    # Single directory conversion:
    python convert_nwchem.py --input data/raw/h2p_001/ --output data/processed/h2p_001.h5

    # From simulation index (pandas DataFrame):
    python convert_nwchem.py --index rt_simulations.pkl --output data/processed/

This script parses NWChem restart files and geometry data, converting them
into the HDF5 trajectory format used for training.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nwchem_parser import NWChemParser
from src.data.trajectory import Trajectory
from src.data.simulation_index import SimulationIndex, SimulationRecord


def convert_directory(
    input_dir: Path,
    output_file: Path,
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
    verbose: bool = True,
) -> Trajectory:
    """
    Convert a single NWChem output directory to HDF5 trajectory.

    Args:
        input_dir: Directory containing NWChem output files
        output_file: Path for output HDF5 file
        molecule: Molecule identifier (e.g., 'h2', 'lih')
        n_basis: Number of basis functions
        n_electrons: Number of electrons
        n_spin: Number of spin channels (1 or 2)
        dt: Time step in atomic units
        geometry_file: Name of geometry file
        overlap_file: Name of overlap matrix file
        restart_pattern: Pattern for density restart files
        field_file: Name of field protocol file
        basis_file: Name of basis info file
        verbose: Whether to print progress

    Returns:
        Converted Trajectory object
    """
    if verbose:
        print(f"Converting {input_dir} -> {output_file}")

    parser = NWChemParser(input_dir)

    # Parse trajectory
    trajectory = parser.parse_trajectory(
        molecule=molecule,
        n_basis=n_basis,
        n_electrons=n_electrons,
        n_spin=n_spin,
        dt=dt,
        geometry_file=geometry_file,
        overlap_file=overlap_file,
        restart_pattern=restart_pattern,
        field_file=field_file,
        basis_file=basis_file,
    )

    if verbose:
        print(f"  Molecule: {trajectory.molecule}")
        print(f"  Atoms: {trajectory.n_atoms}")
        print(f"  Basis functions: {trajectory.n_basis}")
        print(f"  Electrons: {trajectory.n_electrons}")
        print(f"  Time steps: {trajectory.n_steps}")
        print(f"  Time step: {trajectory.dt} a.u.")

    # Save to HDF5
    output_file.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(output_file)

    if verbose:
        print(f"  Saved to {output_file}")

    return trajectory


def convert_batch(
    input_base: Path,
    output_base: Path,
    molecule_config: dict,
    verbose: bool = True,
) -> list[Path]:
    """
    Convert multiple NWChem directories to HDF5 trajectories.

    Args:
        input_base: Base directory containing subdirectories for each trajectory
        output_base: Base directory for output HDF5 files
        molecule_config: Dictionary with molecule parameters:
            {
                'molecule': str,
                'n_basis': int,
                'n_electrons': int,
                'n_spin': int,
                'dt': float,
            }
        verbose: Whether to print progress

    Returns:
        List of output file paths
    """
    output_files = []

    # Find all subdirectories with density files
    for subdir in sorted(input_base.iterdir()):
        if not subdir.is_dir():
            continue

        # Check if directory has density files
        density_files = list(subdir.glob("*.rt_restart.*.dat"))
        if not density_files:
            if verbose:
                print(f"Skipping {subdir}: no density files found")
            continue

        # Output file
        output_file = output_base / f"{subdir.name}.h5"

        try:
            convert_directory(
                input_dir=subdir,
                output_file=output_file,
                **molecule_config,
                verbose=verbose,
            )
            output_files.append(output_file)
        except Exception as e:
            print(f"Error converting {subdir}: {e}")
            continue

    return output_files


def convert_from_index(
    index_file: Path,
    output_dir: Path,
    molecule_filter: Optional[str] = None,
    geometry_filter: Optional[str] = None,
    field_type_filter: Optional[str] = None,
    max_trajectories: Optional[int] = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Convert trajectories from a simulation index (pandas DataFrame).

    Args:
        index_file: Path to pickled simulation index DataFrame
        output_dir: Directory for output HDF5 files
        molecule_filter: Only convert trajectories for this molecule
        geometry_filter: Only convert trajectories with this geometry
        field_type_filter: Only convert trajectories with this field type
        max_trajectories: Maximum number of trajectories to convert
        verbose: Whether to print progress

    Returns:
        List of output file paths
    """
    if verbose:
        print(f"Loading simulation index from {index_file}")

    index = SimulationIndex.load(index_file)

    if verbose:
        print(f"  Found {len(index)} simulations")

    # Apply filters
    if molecule_filter:
        index = index.filter(molecule=molecule_filter)
        if verbose:
            print(f"  After molecule filter: {len(index)}")

    if geometry_filter:
        index = index.filter(geometry=geometry_filter)
        if verbose:
            print(f"  After geometry filter: {len(index)}")

    if field_type_filter:
        index = index.filter(field_type=field_type_filter)
        if verbose:
            print(f"  After field_type filter: {len(index)}")

    # Filter to only those with data available
    index = index.filter(has_data=True)
    if verbose:
        print(f"  With available data: {len(index)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    parser = NWChemParser()

    records = list(index)
    if max_trajectories:
        records = records[:max_trajectories]

    for i, record in enumerate(records):
        # Generate output filename from record metadata
        output_name = f"{record.molecule}_{record.geometry}_{record.field_name}.h5"
        output_file = output_dir / record.molecule / output_name

        if verbose:
            print(f"[{i+1}/{len(records)}] Converting {record.calc_name}")
            print(f"  Molecule: {record.molecule}, nbf: {record.nbf}, tsteps: {record.tsteps}")

        try:
            trajectory = parser.parse_trajectory_from_record(record)

            if trajectory is None:
                if verbose:
                    print(f"  Skipped: no valid data")
                continue

            if verbose:
                print(f"  Loaded {trajectory.n_steps} steps, {trajectory.n_basis} basis functions")

            output_file.parent.mkdir(parents=True, exist_ok=True)
            trajectory.save(output_file)
            output_files.append(output_file)

            if verbose:
                print(f"  Saved to {output_file}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert NWChem RT-TDDFT output to HDF5 trajectories"
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Input directory containing NWChem output files",
    )
    input_group.add_argument(
        "--index",
        type=Path,
        help="Simulation index pickle file (pandas DataFrame)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 file or directory",
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default="unknown",
        help="Molecule identifier (e.g., 'h2', 'lih', 'h2o')",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        help="Number of basis functions (required for --input mode)",
    )
    parser.add_argument(
        "--n-electrons",
        type=int,
        help="Number of electrons (required for --input mode)",
    )
    parser.add_argument(
        "--n-spin",
        type=int,
        default=1,
        help="Number of spin channels (1 for restricted, 2 for unrestricted)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.2,
        help="Time step in atomic units",
    )
    parser.add_argument(
        "--geometry-file",
        type=str,
        default="geometry.xyz",
        help="Name of geometry XYZ file",
    )
    parser.add_argument(
        "--overlap-file",
        type=str,
        default="overlap.dat",
        help="Name of overlap matrix file",
    )
    parser.add_argument(
        "--restart-pattern",
        type=str,
        default="density_{:05d}.dat",
        help="Pattern for density restart files",
    )
    parser.add_argument(
        "--field-file",
        type=str,
        default="field.dat",
        help="Name of field protocol file",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all subdirectories in input",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # Index-mode specific options
    parser.add_argument(
        "--filter-molecule",
        type=str,
        help="Only convert trajectories for this molecule (index mode)",
    )
    parser.add_argument(
        "--filter-geometry",
        type=str,
        help="Only convert trajectories with this geometry (index mode)",
    )
    parser.add_argument(
        "--filter-field-type",
        type=str,
        help="Only convert trajectories with this field type (index mode)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        help="Maximum number of trajectories to convert (index mode)",
    )

    args = parser.parse_args()

    # Validate required args for non-index modes
    if not args.index:
        if args.n_basis is None:
            parser.error("--n-basis is required when using --input mode")
        if args.n_electrons is None:
            parser.error("--n-electrons is required when using --input mode")

    if args.index:
        # Index-based conversion mode
        output_files = convert_from_index(
            index_file=args.index,
            output_dir=args.output,
            molecule_filter=args.filter_molecule,
            geometry_filter=args.filter_geometry,
            field_type_filter=args.filter_field_type,
            max_trajectories=args.max_trajectories,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print(f"\nConverted {len(output_files)} trajectories")

    elif args.batch:
        # Batch conversion mode
        molecule_config = {
            "molecule": args.molecule,
            "n_basis": args.n_basis,
            "n_electrons": args.n_electrons,
            "n_spin": args.n_spin,
            "dt": args.dt,
        }

        output_files = convert_batch(
            input_base=args.input,
            output_base=args.output,
            molecule_config=molecule_config,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print(f"\nConverted {len(output_files)} trajectories")

    else:
        # Single file conversion mode
        convert_directory(
            input_dir=args.input,
            output_file=args.output,
            molecule=args.molecule,
            n_basis=args.n_basis,
            n_electrons=args.n_electrons,
            n_spin=args.n_spin,
            dt=args.dt,
            geometry_file=args.geometry_file,
            overlap_file=args.overlap_file,
            restart_pattern=args.restart_pattern,
            field_file=args.field_file,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
