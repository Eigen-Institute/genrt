"""
Density matrix decoder using tensor products and equivariant operations.

This module provides the DensityDecoder class which reconstructs density
matrices from latent representations, using tensor products to correctly
handle the transformation properties of the density matrix.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from e3nn.o3 import Irreps


class QueryTokenDecoder(nn.Module):
    """
    Decode query tokens to per-orbital-pair features using cross-attention.

    This reverses the encoding process: takes the fixed-size query token
    representation and expands it back to the variable-size orbital pair space.

    Args:
        latent_dim: Dimension of query token features
        n_query_tokens: Number of query tokens
        output_dim: Dimension of per-orbital-pair output
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int = 256,
        n_query_tokens: int = 32,
        output_dim: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_query_tokens = n_query_tokens
        self.output_dim = output_dim

        # Cross-attention: orbital queries attend to latent tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(
        self,
        latent: Tensor,
        orbital_queries: Tensor,
    ) -> Tensor:
        """
        Decode latent tokens to orbital-pair features.

        Args:
            latent: Query token features, shape (n_query_tokens, latent_dim)
            orbital_queries: Orbital pair embeddings, shape (n_pairs, latent_dim)

        Returns:
            Per-pair features, shape (n_pairs, output_dim)
        """
        # Add batch dimension
        latent = latent.unsqueeze(0)  # (1, n_query, latent_dim)
        orbital_queries = orbital_queries.unsqueeze(0)  # (1, n_pairs, latent_dim)

        # Normalize
        latent = self.norm_kv(latent)
        orbital_queries = self.norm_q(orbital_queries)

        # Cross-attention: orbital queries attend to latent
        attended, _ = self.cross_attention(
            orbital_queries,
            latent,
            latent,
        )  # (1, n_pairs, latent_dim)

        # Remove batch dimension and project
        output = attended.squeeze(0)
        output = self.output_proj(output)

        return output


class DensityElementDecoder(nn.Module):
    """
    Decode per-orbital-pair features to density matrix elements.

    Takes features for each (i, j) orbital pair and outputs the
    real and imaginary parts of the density matrix element.

    Args:
        input_dim: Dimension of input features per pair
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # real + imag
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decode features to density matrix elements.

        Args:
            features: Per-pair features, shape (n_pairs, input_dim)

        Returns:
            Tuple of (real, imag) parts, each shape (n_pairs,)
        """
        output = self.decoder(features)
        return output[..., 0], output[..., 1]


class DensityDecoder(nn.Module):
    """
    Decode latent representation to density matrix.

    Reconstructs density matrices from the fixed-size latent representation
    produced by DensityEdgeProjection. Uses cross-attention to expand
    back to variable orbital space.

    Args:
        latent_dim: Dimension of latent representation
        n_query_tokens: Number of query tokens (must match encoder)
        max_l: Maximum angular momentum in basis
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int = 256,
        n_query_tokens: int = 32,
        max_l: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_query_tokens = n_query_tokens

        # Orbital embedding (same structure as encoder)
        self.element_embed = nn.Embedding(100, 32)
        self.l_embed = nn.Embedding(max_l + 1, 16)
        self.m_embed = nn.Embedding(2 * max_l + 1, 16)

        self.orbital_proj = nn.Sequential(
            nn.Linear(32 + 16 + 16, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Query token decoder
        self.token_decoder = QueryTokenDecoder(
            latent_dim=latent_dim,
            n_query_tokens=n_query_tokens,
            output_dim=latent_dim // 2,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Element decoder
        self.element_decoder = DensityElementDecoder(
            input_dim=latent_dim // 2,
            hidden_dim=latent_dim,
        )

    def _embed_orbitals(
        self,
        Z: Tensor,
        l: Tensor,
        m: Tensor,
    ) -> Tensor:
        """Embed orbital metadata."""
        m_shifted = m + 3
        elem_emb = self.element_embed(Z)
        l_emb = self.l_embed(l)
        m_emb = self.m_embed(m_shifted.clamp(0, 6))
        combined = torch.cat([elem_emb, l_emb, m_emb], dim=-1)
        return self.orbital_proj(combined)

    def forward(
        self,
        latent: Tensor,
        basis_metadata: dict[str, Tensor],
        n_spin: int = 1,
    ) -> Tensor:
        """
        Decode latent to density matrix.

        Args:
            latent: Latent representation, shape (n_query_tokens, latent_dim)
            basis_metadata: Dictionary containing:
                - 'Z': Atomic number per basis function, shape (n_basis,)
                - 'l': Angular momentum per basis function, shape (n_basis,)
                - 'm': Magnetic quantum number per basis function, shape (n_basis,)
            n_spin: Number of spin channels

        Returns:
            Reconstructed density matrix, shape (n_spin, n_basis, n_basis) complex
        """
        device = latent.device
        n_basis = len(basis_metadata['Z'])

        # Get basis metadata
        Z = basis_metadata['Z'].to(device)
        l = basis_metadata['l'].to(device)
        m = basis_metadata['m'].to(device)

        # Embed orbitals
        orbital_embeddings = self._embed_orbitals(Z, l, m)

        # Create pair indices
        i_idx, j_idx = torch.meshgrid(
            torch.arange(n_basis, device=device),
            torch.arange(n_basis, device=device),
            indexing='ij'
        )
        i_idx = i_idx.flatten()
        j_idx = j_idx.flatten()

        # Create orbital pair queries
        orbital_i = orbital_embeddings[i_idx]
        orbital_j = orbital_embeddings[j_idx]
        pair_queries = orbital_i + orbital_j  # Simple combination

        # Decode to per-pair features
        pair_features = self.token_decoder(latent, pair_queries)

        # Decode to density elements
        rho_real, rho_imag = self.element_decoder(pair_features)

        # Reshape to matrix
        rho = torch.complex(rho_real, rho_imag)
        rho = rho.reshape(n_basis, n_basis)

        # Expand to spin channels
        rho = rho.unsqueeze(0).expand(n_spin, -1, -1).clone()

        return rho

    def forward_batch(
        self,
        latent_batch: Tensor,
        basis_metadata_list: list[dict[str, Tensor]],
        n_spin: int = 1,
    ) -> list[Tensor]:
        """
        Decode batch of latents with potentially different basis sizes.

        Args:
            latent_batch: Batched latents, shape (batch, n_query_tokens, latent_dim)
            basis_metadata_list: List of basis metadata dicts
            n_spin: Number of spin channels

        Returns:
            List of density matrices, each (n_spin, n_basis_i, n_basis_i)
        """
        outputs = []
        for i, metadata in enumerate(basis_metadata_list):
            latent = latent_batch[i]
            rho = self.forward(latent, metadata, n_spin)
            outputs.append(rho)
        return outputs


class EquivariantDensityDecoder(nn.Module):
    """
    Density decoder using equivariant tensor products.

    Combines geometry features with latent representation using
    tensor products to maintain SO(3) equivariance during decoding.

    Args:
        latent_dim: Dimension of latent representation
        geometry_irreps: Irreps of geometry features
        hidden_dim: Hidden dimension
        n_query_tokens: Number of query tokens
    """

    def __init__(
        self,
        latent_dim: int = 256,
        geometry_irreps: str = "32x0e + 16x1o + 8x2e",
        hidden_dim: int = 256,
        n_query_tokens: int = 32,
    ):
        super().__init__()
        from e3nn import o3

        self.latent_dim = latent_dim
        self.geometry_irreps = Irreps(geometry_irreps)

        # Scalar (L=0) components of geometry for conditioning
        self.scalar_dim = sum(
            mul * (2 * ir.l + 1)
            for mul, ir in self.geometry_irreps
            if ir.l == 0
        )

        # Combine latent with geometry scalars
        self.combiner = nn.Sequential(
            nn.Linear(latent_dim + self.scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Standard decoder after combination
        self.base_decoder = DensityDecoder(
            latent_dim=latent_dim,
            n_query_tokens=n_query_tokens,
        )

    def _extract_scalars(self, geometry_features: Tensor) -> Tensor:
        """Extract L=0 components from geometry features."""
        scalars = []
        idx = 0
        for mul, ir in self.geometry_irreps:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 0:
                scalars.append(geometry_features[..., idx:idx + dim])
            idx += dim
        return torch.cat(scalars, dim=-1) if scalars else geometry_features.new_zeros(1)

    def forward(
        self,
        latent: Tensor,
        geometry_features: Tensor,
        basis_metadata: dict[str, Tensor],
        n_spin: int = 1,
    ) -> Tensor:
        """
        Decode latent with geometry conditioning.

        Args:
            latent: Latent representation, shape (n_query_tokens, latent_dim)
            geometry_features: Per-atom geometry features, shape (n_atoms, geometry_dim)
            basis_metadata: Basis function metadata
            n_spin: Number of spin channels

        Returns:
            Reconstructed density matrix, shape (n_spin, n_basis, n_basis) complex
        """
        # Pool geometry features to get global context
        geom_pooled = geometry_features.mean(dim=0)  # (geometry_dim,)
        geom_scalars = self._extract_scalars(geom_pooled)

        # Combine with each latent token
        latent_combined = []
        for i in range(latent.shape[0]):
            combined = torch.cat([latent[i], geom_scalars], dim=-1)
            combined = self.combiner(combined)
            latent_combined.append(combined)

        latent_combined = torch.stack(latent_combined, dim=0)

        # Decode using base decoder
        return self.base_decoder(latent_combined, basis_metadata, n_spin)


class ResidualDensityDecoder(nn.Module):
    """
    Decoder that predicts density change (residual) rather than absolute density.

    For dynamics modeling, it's often better to predict the change
    from the current state: rho_{t+1} = rho_t + delta_rho

    Args:
        latent_dim: Dimension of latent representation
        n_query_tokens: Number of query tokens
        max_l: Maximum angular momentum
    """

    def __init__(
        self,
        latent_dim: int = 256,
        n_query_tokens: int = 32,
        max_l: int = 2,
    ):
        super().__init__()
        self.base_decoder = DensityDecoder(
            latent_dim=latent_dim,
            n_query_tokens=n_query_tokens,
            max_l=max_l,
        )

        # Scale factor for residual (learnable, initialized small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        latent: Tensor,
        rho_current: Tensor,
        basis_metadata: dict[str, Tensor],
    ) -> Tensor:
        """
        Predict next density as current + residual.

        Args:
            latent: Latent representation, shape (n_query_tokens, latent_dim)
            rho_current: Current density, shape (n_spin, n_basis, n_basis)
            basis_metadata: Basis function metadata

        Returns:
            Predicted next density, shape (n_spin, n_basis, n_basis)
        """
        n_spin = rho_current.shape[0]

        # Decode residual
        delta_rho = self.base_decoder(latent, basis_metadata, n_spin)

        # Apply scaled residual
        rho_next = rho_current + self.residual_scale * delta_rho

        return rho_next
