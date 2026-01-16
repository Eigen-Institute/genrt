"""
Data loading and processing for RT-TDDFT trajectories.

This module provides:
- HDF5-based trajectory storage and loading
- NWChem output file parsing
- PyTorch Dataset classes for training
- Data augmentation transforms
"""

from .trajectory import Trajectory, BasisFunctionInfo
from .dataset import TrajectoryDataset, UnifiedTrajectoryDataset
from .nwchem_parser import NWChemParser
from .transforms import GeometryNoiseTransform, FieldNoiseTransform, ComposeTransforms
from .simulation_index import SimulationIndex, SimulationRecord

__all__ = [
    "Trajectory",
    "BasisFunctionInfo",
    "TrajectoryDataset",
    "UnifiedTrajectoryDataset",
    "NWChemParser",
    "GeometryNoiseTransform",
    "FieldNoiseTransform",
    "ComposeTransforms",
    "SimulationIndex",
    "SimulationRecord",
]
