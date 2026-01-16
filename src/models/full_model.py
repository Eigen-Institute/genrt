"""
Complete RT-TDDFT acceleration model.

This module provides the RTTDDFTModel class which combines all components:
- GeometryEncoder: E3NN equivariant geometry encoding
- DensityEdgeProjection: Variable-basis density encoding
- EquivariantFieldEncoder: External field encoding
- GeometryConditionedMamba: Mamba dynamics with FiLM
- DensityDecoder: Latent to density reconstruction

The model predicts density matrix evolution under external fields.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .geometry_encoder import GeometryEncoder
from .density_encoder import DensityEdgeProjection
from .field_encoder import EquivariantFieldEncoder, FieldTimeEncoder
from .dynamics import GeometryConditionedMamba, create_dynamics_model, DynamicsConfig
from .decoder import DensityDecoder, ResidualDensityDecoder


@dataclass
class RTTDDFTConfig:
    """Configuration for RTTDDFTModel."""
    # Geometry encoder
    geometry_irreps: str = "32x0e + 16x1o + 8x2e"
    geometry_layers: int = 4
    max_radius: float = 5.0
    num_radial_basis: int = 8

    # Density encoder
    n_query_tokens: int = 32
    latent_dim: int = 256

    # Field encoder
    field_scalar_dim: int = 32
    field_vector_multiplicity: int = 16

    # Dynamics
    mamba_d_model: int = 256
    mamba_d_state: int = 16
    mamba_layers: int = 6
    mamba_dropout: float = 0.1

    # Decoder
    max_l: int = 2

    # Physics
    apply_hermitian: bool = True
    apply_trace: bool = True
    apply_mcweeney: bool = False
    mcweeney_iterations: int = 3


class RTTDDFTModel(nn.Module):
    """
    Complete RT-TDDFT acceleration model.

    Predicts density matrix evolution ρ(t) → ρ(t+dt) given:
    - Molecular geometry (positions, atomic numbers)
    - Current density matrix ρ(t)
    - External field E(t)

    Architecture:
    1. Encode geometry → equivariant node features
    2. Encode density matrix → fixed-size latent
    3. Encode external field → equivariant features
    4. Combine and evolve with Mamba dynamics
    5. Decode to density matrix
    6. Apply physics constraints

    Args:
        config: Model configuration
    """

    def __init__(self, config: Optional[RTTDDFTConfig] = None):
        super().__init__()
        if config is None:
            config = RTTDDFTConfig()
        self.config = config

        # Geometry encoder
        self.geometry_encoder = GeometryEncoder(
            irreps_out=config.geometry_irreps,
            num_layers=config.geometry_layers,
            max_radius=config.max_radius,
            num_basis=config.num_radial_basis,
        )

        # Get geometry output dimension
        from e3nn.o3 import Irreps
        geometry_dim = Irreps(config.geometry_irreps).dim

        # Density encoder
        self.density_encoder = DensityEdgeProjection(
            latent_dim=config.latent_dim,
            n_query_tokens=config.n_query_tokens,
            max_l=config.max_l,
        )

        # Field encoder
        self.field_encoder = FieldTimeEncoder(
            scalar_dim=config.field_scalar_dim,
            vector_multiplicity=config.field_vector_multiplicity,
        )

        # Geometry pooling for dynamics conditioning
        self.geometry_pool = nn.Sequential(
            nn.Linear(geometry_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

        # Dynamics model
        dynamics_config = DynamicsConfig(
            d_model=config.mamba_d_model,
            d_state=config.mamba_d_state,
            n_layers=config.mamba_layers,
            dropout=config.mamba_dropout,
        )
        self.dynamics = create_dynamics_model(
            config=dynamics_config,
            latent_dim=config.latent_dim,
            geometry_dim=config.latent_dim,  # After projection
            use_simplified=True,  # Use simplified for compatibility
        )

        # Decoder
        self.decoder = DensityDecoder(
            latent_dim=config.latent_dim,
            n_query_tokens=config.n_query_tokens,
            max_l=config.max_l,
        )

        # Optional: Residual decoder for predicting changes
        self.use_residual = True
        if self.use_residual:
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def encode_geometry(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode molecular geometry.

        Args:
            positions: Atomic positions, shape (n_atoms, 3)
            atomic_numbers: Atomic numbers, shape (n_atoms,)

        Returns:
            Tuple of:
            - Per-atom features, shape (n_atoms, geometry_dim)
            - Pooled features for conditioning, shape (latent_dim,)
        """
        # Encode geometry
        node_features = self.geometry_encoder(positions, atomic_numbers)

        # Pool for global conditioning
        pooled = node_features.mean(dim=0)  # Simple mean pooling
        pooled = self.geometry_pool(pooled)

        return node_features, pooled

    def encode_density(
        self,
        rho: Tensor,
        basis_metadata: Dict[str, Tensor],
    ) -> Tensor:
        """
        Encode density matrix to latent representation.

        Args:
            rho: Density matrix, shape (n_spin, n_basis, n_basis) complex
            basis_metadata: Basis function metadata

        Returns:
            Latent representation, shape (n_query_tokens, latent_dim)
        """
        return self.density_encoder(rho, basis_metadata)

    def encode_field(
        self,
        field: Tensor,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode external field and time.

        Args:
            field: Electric field, shape (3,)
            time: Optional time, shape ()

        Returns:
            Field features, shape (field_dim,)
        """
        return self.field_encoder(field, time)

    def decode_density(
        self,
        latent: Tensor,
        basis_metadata: Dict[str, Tensor],
        n_spin: int = 1,
    ) -> Tensor:
        """
        Decode latent to density matrix.

        Args:
            latent: Latent representation, shape (n_query_tokens, latent_dim)
            basis_metadata: Basis function metadata
            n_spin: Number of spin channels

        Returns:
            Density matrix, shape (n_spin, n_basis, n_basis) complex
        """
        return self.decoder(latent, basis_metadata, n_spin)

    def apply_physics_constraints(
        self,
        rho: Tensor,
        overlap: Tensor,
        n_electrons: int,
    ) -> Tensor:
        """
        Apply physics constraints to density matrix.

        Args:
            rho: Density matrix, shape (n_spin, n_basis, n_basis)
            overlap: Overlap matrix, shape (n_basis, n_basis)
            n_electrons: Number of electrons

        Returns:
            Constrained density matrix
        """
        from ..utils.complex_tensor import hermitianize, trace_normalize, mcweeney_purification

        # Process each spin channel
        rho_constrained = []
        for s in range(rho.shape[0]):
            rho_s = rho[s]

            # 1. Hermitianize
            if self.config.apply_hermitian:
                rho_s = hermitianize(rho_s)

            # 2. Trace normalize
            if self.config.apply_trace:
                n_elec_per_spin = n_electrons // rho.shape[0]
                rho_s = trace_normalize(rho_s, overlap, n_elec_per_spin)

            # 3. McWeeney purification (optional)
            if self.config.apply_mcweeney:
                rho_s = mcweeney_purification(
                    rho_s, overlap, n_elec_per_spin,
                    n_iterations=self.config.mcweeney_iterations
                )

            rho_constrained.append(rho_s)

        return torch.stack(rho_constrained, dim=0)

    def forward(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
        rho_current: Tensor,
        field: Tensor,
        basis_metadata: Dict[str, Tensor],
        overlap: Optional[Tensor] = None,
        n_electrons: Optional[int] = None,
        time: Optional[Tensor] = None,
        apply_constraints: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Predict next density matrix.

        Args:
            positions: Atomic positions, shape (n_atoms, 3)
            atomic_numbers: Atomic numbers, shape (n_atoms,)
            rho_current: Current density, shape (n_spin, n_basis, n_basis)
            field: External field, shape (3,)
            basis_metadata: Basis function metadata
            overlap: Optional overlap matrix for constraints
            n_electrons: Optional electron count for constraints
            time: Optional time for field encoding
            apply_constraints: Whether to apply physics constraints

        Returns:
            Dictionary with:
            - 'rho_pred': Predicted density, shape (n_spin, n_basis, n_basis)
            - 'latent_current': Current latent, shape (n_query, latent_dim)
            - 'latent_next': Next latent, shape (n_query, latent_dim)
            - 'geometry_features': Geometry features, shape (n_atoms, geom_dim)
        """
        device = positions.device
        n_spin = rho_current.shape[0]

        # 1. Encode geometry
        geometry_features, geometry_pooled = self.encode_geometry(
            positions, atomic_numbers
        )

        # 2. Encode current density
        latent_current = self.encode_density(rho_current, basis_metadata)

        # 3. Encode field
        field_features = self.encode_field(field, time)

        # 4. Flatten latent for dynamics (n_query_tokens, latent_dim) -> (1, n_query_tokens, latent_dim)
        latent_flat = latent_current.unsqueeze(0)

        # 5. Evolve with dynamics
        latent_next = self.dynamics(latent_flat, geometry_pooled.unsqueeze(0))
        latent_next = latent_next.squeeze(0)

        # 6. Decode to density
        if self.use_residual:
            # Predict change and add to current
            delta_rho = self.decode_density(latent_next, basis_metadata, n_spin)
            rho_pred = rho_current + self.residual_scale * delta_rho
        else:
            rho_pred = self.decode_density(latent_next, basis_metadata, n_spin)

        # 7. Apply physics constraints
        if apply_constraints and overlap is not None and n_electrons is not None:
            rho_pred = self.apply_physics_constraints(rho_pred, overlap, n_electrons)

        return {
            'rho_pred': rho_pred,
            'latent_current': latent_current,
            'latent_next': latent_next,
            'geometry_features': geometry_features,
        }

    def predict_trajectory(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
        rho_initial: Tensor,
        field_sequence: Tensor,
        basis_metadata: Dict[str, Tensor],
        overlap: Optional[Tensor] = None,
        n_electrons: Optional[int] = None,
        apply_constraints: bool = True,
    ) -> Tensor:
        """
        Predict a trajectory of density matrices.

        Args:
            positions: Atomic positions, shape (n_atoms, 3)
            atomic_numbers: Atomic numbers, shape (n_atoms,)
            rho_initial: Initial density, shape (n_spin, n_basis, n_basis)
            field_sequence: Field at each step, shape (n_steps, 3)
            basis_metadata: Basis function metadata
            overlap: Optional overlap matrix
            n_electrons: Optional electron count
            apply_constraints: Whether to apply constraints

        Returns:
            Density trajectory, shape (n_steps, n_spin, n_basis, n_basis)
        """
        n_steps = field_sequence.shape[0]
        n_spin = rho_initial.shape[0]
        n_basis = rho_initial.shape[1]
        device = positions.device

        # Encode geometry once
        geometry_features, geometry_pooled = self.encode_geometry(
            positions, atomic_numbers
        )

        # Initialize
        rho_current = rho_initial
        trajectory = []

        for t in range(n_steps):
            field_t = field_sequence[t]

            # Encode current density
            latent_current = self.encode_density(rho_current, basis_metadata)

            # Evolve
            latent_flat = latent_current.unsqueeze(0)
            latent_next = self.dynamics(latent_flat, geometry_pooled.unsqueeze(0))
            latent_next = latent_next.squeeze(0)

            # Decode
            if self.use_residual:
                delta_rho = self.decode_density(latent_next, basis_metadata, n_spin)
                rho_next = rho_current + self.residual_scale * delta_rho
            else:
                rho_next = self.decode_density(latent_next, basis_metadata, n_spin)

            # Constraints
            if apply_constraints and overlap is not None and n_electrons is not None:
                rho_next = self.apply_physics_constraints(rho_next, overlap, n_electrons)

            trajectory.append(rho_next)
            rho_current = rho_next

        return torch.stack(trajectory, dim=0)


class RTTDDFTModelLite(nn.Module):
    """
    Lightweight version of RTTDDFTModel for faster inference.

    Uses smaller dimensions and fewer layers, suitable for
    initial prototyping and small molecules.
    """

    def __init__(self):
        super().__init__()
        config = RTTDDFTConfig(
            geometry_irreps="16x0e + 8x1o + 4x2e",
            geometry_layers=2,
            n_query_tokens=16,
            latent_dim=128,
            mamba_d_model=128,
            mamba_layers=3,
        )
        # Build as standard model with lite config
        self.model = RTTDDFTModel(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def create_model(
    model_size: str = "base",
    **kwargs,
) -> RTTDDFTModel:
    """
    Factory function to create RTTDDFTModel with predefined configurations.

    Args:
        model_size: One of "lite", "base", "large"
        **kwargs: Override config parameters

    Returns:
        RTTDDFTModel instance
    """
    configs = {
        "lite": RTTDDFTConfig(
            geometry_irreps="16x0e + 8x1o + 4x2e",
            geometry_layers=2,
            n_query_tokens=16,
            latent_dim=128,
            mamba_d_model=128,
            mamba_layers=3,
        ),
        "base": RTTDDFTConfig(),
        "large": RTTDDFTConfig(
            geometry_irreps="64x0e + 32x1o + 16x2e",
            geometry_layers=6,
            n_query_tokens=64,
            latent_dim=512,
            mamba_d_model=512,
            mamba_layers=8,
        ),
    }

    config = configs.get(model_size, configs["base"])

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return RTTDDFTModel(config)
