"""
Equivariant field encoder for external electric fields.

This module provides the EquivariantFieldEncoder class which encodes
time-dependent external electric fields E(t) into equivariant representations
that couple correctly with molecular geometry features.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from e3nn.o3 import Irreps, spherical_harmonics


class FieldMagnitudeEncoder(nn.Module):
    """
    Encode the magnitude of the electric field.

    The field magnitude is a scalar (L=0) quantity that can be processed
    with standard nonlinearities.

    Args:
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension for scalar features
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, field_magnitude: Tensor) -> Tensor:
        """
        Encode field magnitude.

        Args:
            field_magnitude: Field magnitude, shape (...,) or (..., 1)

        Returns:
            Scalar features, shape (..., output_dim)
        """
        if field_magnitude.dim() == 0 or field_magnitude.shape[-1] != 1:
            field_magnitude = field_magnitude.unsqueeze(-1)
        return self.encoder(field_magnitude)


class EquivariantFieldEncoder(nn.Module):
    """
    Encode external electric field E(t) into equivariant features.

    The electric field is a 3D vector that transforms as L=1 under rotations.
    This encoder creates equivariant features that can be combined with
    geometry features while preserving SO(3) equivariance.

    The encoder outputs:
    - L=0 (scalar) features from field magnitude
    - L=1 (vector) features from field direction

    Args:
        scalar_dim: Dimension of scalar (L=0) output features
        vector_multiplicity: Number of L=1 output channels
        hidden_dim: Hidden dimension for MLPs
    """

    def __init__(
        self,
        scalar_dim: int = 32,
        vector_multiplicity: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_multiplicity = vector_multiplicity

        # Output irreps: scalars + vectors
        self.irreps_out = Irreps(f"{scalar_dim}x0e + {vector_multiplicity}x1o")

        # Encode field magnitude (scalar)
        self.magnitude_encoder = FieldMagnitudeEncoder(
            hidden_dim=hidden_dim,
            output_dim=scalar_dim,
        )

        # Generate vector channel weights from magnitude
        # Each L=1 channel gets 3 components, weighted by learned scalar
        self.vector_weight_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vector_multiplicity),
        )

    def forward(self, field: Tensor) -> Tensor:
        """
        Encode electric field to equivariant features.

        Args:
            field: Electric field vector E(t), shape (..., 3)

        Returns:
            Equivariant features, shape (..., irreps_out.dim)
            Contains scalar_dim scalars followed by vector_multiplicity * 3 vector components
        """
        # Compute field magnitude
        magnitude = torch.norm(field, dim=-1, keepdim=True)  # (..., 1)

        # Encode magnitude to scalar features
        scalar_features = self.magnitude_encoder(magnitude)  # (..., scalar_dim)

        # Compute field direction (handle zero field)
        direction = field / (magnitude + 1e-8)  # (..., 3)

        # Generate weights for vector channels
        vector_weights = self.vector_weight_net(magnitude)  # (..., vector_multiplicity)

        # Create vector features by scaling direction
        # direction: (..., 3), weights: (..., vector_multiplicity)
        # output: (..., vector_multiplicity, 3) -> (..., vector_multiplicity * 3)
        vector_features = direction.unsqueeze(-2) * vector_weights.unsqueeze(-1)
        vector_features = vector_features.reshape(*field.shape[:-1], -1)

        # Concatenate scalar and vector features
        output = torch.cat([scalar_features, vector_features], dim=-1)

        return output

    def get_scalar_features(self, features: Tensor) -> Tensor:
        """Extract scalar (L=0) components from encoded features."""
        return features[..., :self.scalar_dim]

    def get_vector_features(self, features: Tensor) -> Tensor:
        """
        Extract vector (L=1) components from encoded features.

        Returns:
            Vector features, shape (..., vector_multiplicity, 3)
        """
        vec_flat = features[..., self.scalar_dim:]
        return vec_flat.reshape(*features.shape[:-1], self.vector_multiplicity, 3)


class FieldGeometryCoupling(nn.Module):
    """
    Couple field features with geometry features via tensor product.

    This module combines the equivariant field encoding with per-atom
    geometry features to create field-conditioned atomic representations.

    Args:
        geometry_irreps: Irreps of geometry encoder output
        field_irreps: Irreps of field encoder output
        output_irreps: Desired output irreps
    """

    def __init__(
        self,
        geometry_irreps: str = "32x0e + 16x1o + 8x2e",
        field_irreps: str = "32x0e + 16x1o",
        output_irreps: str = "32x0e + 16x1o + 8x2e",
    ):
        super().__init__()
        from e3nn import o3

        self.geometry_irreps = Irreps(geometry_irreps)
        self.field_irreps = Irreps(field_irreps)
        self.output_irreps = Irreps(output_irreps)

        # Tensor product for coupling
        self.tp = o3.FullyConnectedTensorProduct(
            self.geometry_irreps,
            self.field_irreps,
            self.output_irreps,
        )

        # Residual connection for geometry
        self.residual = o3.Linear(self.geometry_irreps, self.output_irreps)

    def forward(
        self,
        geometry_features: Tensor,
        field_features: Tensor,
    ) -> Tensor:
        """
        Couple geometry and field features.

        Args:
            geometry_features: Per-atom geometry features, shape (n_atoms, geometry_irreps.dim)
            field_features: Global field features, shape (field_irreps.dim,) or (1, field_irreps.dim)

        Returns:
            Coupled features, shape (n_atoms, output_irreps.dim)
        """
        n_atoms = geometry_features.shape[0]

        # Broadcast field features to all atoms
        if field_features.dim() == 1:
            field_features = field_features.unsqueeze(0)
        field_broadcast = field_features.expand(n_atoms, -1)

        # Tensor product coupling
        coupled = self.tp(geometry_features, field_broadcast)

        # Add residual from geometry
        output = coupled + self.residual(geometry_features)

        return output


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for temporal context.

    Provides positional encoding for time, useful for conditioning
    on the temporal position within a trajectory.

    Args:
        embed_dim: Output embedding dimension
        max_period: Maximum period for sinusoidal encoding
    """

    def __init__(
        self,
        embed_dim: int = 64,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

        # Projection to output dimension
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Compute time embedding.

        Args:
            t: Time values, shape (...,) or (..., 1)

        Returns:
            Time embeddings, shape (..., embed_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[-1] != 1:
            t = t.unsqueeze(-1)

        # Compute frequencies
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )

        # Compute sinusoidal embedding
        args = t * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Handle odd embed_dim
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)

        return self.proj(embedding)


class FieldTimeEncoder(nn.Module):
    """
    Combined encoder for field and time information.

    Encodes both the external electric field E(t) and the time t
    into a combined representation for conditioning the dynamics model.

    Args:
        scalar_dim: Dimension of scalar output features
        vector_multiplicity: Number of L=1 output channels
        time_embed_dim: Dimension of time embedding
        hidden_dim: Hidden dimension for MLPs
    """

    def __init__(
        self,
        scalar_dim: int = 32,
        vector_multiplicity: int = 16,
        time_embed_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_multiplicity = vector_multiplicity
        self.time_embed_dim = time_embed_dim

        # Field encoder
        self.field_encoder = EquivariantFieldEncoder(
            scalar_dim=scalar_dim,
            vector_multiplicity=vector_multiplicity,
            hidden_dim=hidden_dim,
        )

        # Time encoder
        self.time_encoder = TimeEmbedding(
            embed_dim=time_embed_dim,
        )

        # Combine scalar features from field and time
        self.scalar_combiner = nn.Sequential(
            nn.Linear(scalar_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, scalar_dim),
        )

        # Output irreps matches field encoder but with combined scalars
        self.irreps_out = self.field_encoder.irreps_out

    def forward(
        self,
        field: Tensor,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode field and time.

        Args:
            field: Electric field vector E(t), shape (..., 3)
            time: Optional time value, shape (...,) or (..., 1)

        Returns:
            Combined equivariant features, shape (..., irreps_out.dim)
        """
        # Encode field
        field_features = self.field_encoder(field)

        # Extract components
        scalar_features = self.field_encoder.get_scalar_features(field_features)
        vector_features = field_features[..., self.scalar_dim:]

        # If time provided, combine with scalar features
        if time is not None:
            time_features = self.time_encoder(time)
            combined_scalars = torch.cat([scalar_features, time_features], dim=-1)
            scalar_features = self.scalar_combiner(combined_scalars)

        # Reconstruct output
        output = torch.cat([scalar_features, vector_features], dim=-1)

        return output
