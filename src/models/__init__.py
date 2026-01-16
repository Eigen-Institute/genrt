"""
Neural network modules for RT-TDDFT acceleration.

This module provides:
- E3NN-based geometry encoder with SO(3) equivariance
- Variable-basis density matrix encoder
- Equivariant field encoder
- Mamba-based dynamics module with FiLM conditioning
- Physics-aware decoder
"""

from .geometry_encoder import GeometryEncoder
from .density_encoder import (
    DensityEdgeProjection,
    OrbitalEmbedding,
    DensityElementEncoder,
    UniversalBlockEncoder,
)
from .field_encoder import (
    EquivariantFieldEncoder,
    FieldGeometryCoupling,
    FieldTimeEncoder,
    TimeEmbedding,
)
from .dynamics import (
    GeometryConditionedMamba,
    FiLMLayer,
    MambaBlock,
    DynamicsConfig,
    create_dynamics_model,
)
from .decoder import (
    DensityDecoder,
    EquivariantDensityDecoder,
    ResidualDensityDecoder,
    QueryTokenDecoder,
)
from .full_model import (
    RTTDDFTModel,
    RTTDDFTConfig,
    RTTDDFTModelLite,
    create_model,
)

__all__ = [
    # Geometry encoding
    "GeometryEncoder",
    # Density encoding
    "DensityEdgeProjection",
    "OrbitalEmbedding",
    "DensityElementEncoder",
    "UniversalBlockEncoder",
    # Field encoding
    "EquivariantFieldEncoder",
    "FieldGeometryCoupling",
    "FieldTimeEncoder",
    "TimeEmbedding",
    # Dynamics
    "GeometryConditionedMamba",
    "FiLMLayer",
    "MambaBlock",
    "DynamicsConfig",
    "create_dynamics_model",
    # Decoder
    "DensityDecoder",
    "EquivariantDensityDecoder",
    "ResidualDensityDecoder",
    "QueryTokenDecoder",
    # Full model
    "RTTDDFTModel",
    "RTTDDFTConfig",
    "RTTDDFTModelLite",
    "create_model",
]
