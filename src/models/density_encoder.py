"""
Variable-basis density matrix encoder using attention-based projection.

This module provides the DensityEdgeProjection class which encodes density
matrices of variable size into fixed-size latent representations. This is
essential for handling different molecules with different numbers of basis
functions.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from e3nn.o3 import Irreps


class OrbitalEmbedding(nn.Module):
    """
    Embed basis function metadata into learnable features.

    Each basis function is characterized by:
    - The element it's centered on (Z)
    - Its angular momentum (l)
    - Its magnetic quantum number (m)

    Args:
        max_atomic_number: Maximum atomic number to embed
        max_l: Maximum angular momentum quantum number
        embed_dim: Output embedding dimension
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        max_l: int = 3,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_l = max_l

        # Embeddings for each property
        self.element_embed = nn.Embedding(max_atomic_number, 32)
        self.l_embed = nn.Embedding(max_l + 1, 16)
        self.m_embed = nn.Embedding(2 * max_l + 1, 16)  # m ranges from -l to +l

        # Projection to final embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(32 + 16 + 16, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        atomic_numbers: Tensor,
        angular_momentum: Tensor,
        magnetic_quantum: Tensor,
    ) -> Tensor:
        """
        Embed basis function metadata.

        Args:
            atomic_numbers: Atomic numbers (Z), shape (n_basis,)
            angular_momentum: Angular momentum (l), shape (n_basis,)
            magnetic_quantum: Magnetic quantum number (m), shape (n_basis,)

        Returns:
            Orbital embeddings, shape (n_basis, embed_dim)
        """
        # Shift m to be non-negative for embedding lookup
        # m ranges from -l to +l, so shift by max_l to get indices 0 to 2*max_l
        m_shifted = magnetic_quantum + self.max_l

        elem_emb = self.element_embed(atomic_numbers)
        l_emb = self.l_embed(angular_momentum)
        m_emb = self.m_embed(m_shifted.clamp(0, 2 * self.max_l))

        combined = torch.cat([elem_emb, l_emb, m_emb], dim=-1)
        return self.proj(combined)


class DensityElementEncoder(nn.Module):
    """
    Encode individual density matrix elements with orbital context.

    Takes the real and imaginary parts of a density matrix element along
    with the orbital embeddings of both basis functions involved.

    Args:
        orbital_dim: Dimension of orbital embeddings
        hidden_dim: Hidden dimension
        output_dim: Output dimension per element
    """

    def __init__(
        self,
        orbital_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        # Input: 2 orbital embeddings + real + imag
        input_dim = 2 * orbital_dim + 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        rho_real: Tensor,
        rho_imag: Tensor,
        orbital_i: Tensor,
        orbital_j: Tensor,
    ) -> Tensor:
        """
        Encode density matrix elements.

        Args:
            rho_real: Real part of density elements, shape (n_elements,)
            rho_imag: Imaginary part, shape (n_elements,)
            orbital_i: Embeddings for row orbitals, shape (n_elements, orbital_dim)
            orbital_j: Embeddings for col orbitals, shape (n_elements, orbital_dim)

        Returns:
            Encoded elements, shape (n_elements, output_dim)
        """
        # Combine all features
        features = torch.cat([
            orbital_i,
            orbital_j,
            rho_real.unsqueeze(-1),
            rho_imag.unsqueeze(-1),
        ], dim=-1)

        return self.encoder(features)


class DensityEdgeProjection(nn.Module):
    """
    Encode density matrices of variable size to fixed-size latent vectors.

    Uses cross-attention to aggregate variable-sized density matrix
    representations into a fixed number of learnable query tokens.
    This allows handling molecules with different numbers of basis functions.

    Architecture:
    1. Embed each basis function using OrbitalEmbedding
    2. Encode each density matrix element with orbital context
    3. Use cross-attention with learnable queries to get fixed-size output

    Args:
        latent_dim: Dimension of latent representation
        n_query_tokens: Number of query tokens (output size)
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

        # Orbital embedding
        self.orbital_embedding = OrbitalEmbedding(
            max_atomic_number=100,
            max_l=max_l,
            embed_dim=latent_dim // 2,
        )

        # Density element encoder
        self.element_encoder = DensityElementEncoder(
            orbital_dim=latent_dim // 2,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
        )

        # Learnable query tokens for attention pooling
        self.query_tokens = nn.Parameter(torch.randn(n_query_tokens, latent_dim))

        # Cross-attention for pooling
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        rho: Tensor,
        basis_metadata: dict[str, Tensor],
        geometry_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode a density matrix to fixed-size latent representation.

        Args:
            rho: Complex density matrix, shape (n_spin, n_basis, n_basis)
            basis_metadata: Dictionary containing:
                - 'atom_idx': Atom index per basis function, shape (n_basis,)
                - 'Z': Atomic number per basis function, shape (n_basis,)
                - 'l': Angular momentum per basis function, shape (n_basis,)
                - 'm': Magnetic quantum number per basis function, shape (n_basis,)
            geometry_features: Optional geometry context from encoder,
                shape (n_atoms, geom_dim)

        Returns:
            Latent representation, shape (n_query_tokens, latent_dim)
        """
        n_spin, n_basis, _ = rho.shape
        device = rho.device

        # Get basis metadata
        Z = basis_metadata['Z'].to(device)
        l = basis_metadata['l'].to(device)
        m = basis_metadata['m'].to(device)

        # Embed orbitals
        orbital_embeddings = self.orbital_embedding(Z, l, m)  # (n_basis, orbital_dim)

        # Encode all density matrix elements
        # Create indices for all pairs
        i_idx, j_idx = torch.meshgrid(
            torch.arange(n_basis, device=device),
            torch.arange(n_basis, device=device),
            indexing='ij'
        )
        i_idx = i_idx.flatten()
        j_idx = j_idx.flatten()

        # Get orbital embeddings for pairs
        orbital_i = orbital_embeddings[i_idx]  # (n_basis^2, orbital_dim)
        orbital_j = orbital_embeddings[j_idx]

        # Flatten density matrix and encode elements
        # Handle multiple spin channels
        encoded_elements_list = []
        for s in range(n_spin):
            rho_s = rho[s]  # (n_basis, n_basis)
            rho_flat = rho_s.flatten()

            encoded = self.element_encoder(
                rho_flat.real,
                rho_flat.imag,
                orbital_i,
                orbital_j,
            )  # (n_basis^2, latent_dim)
            encoded_elements_list.append(encoded)

        # Concatenate spin channels
        encoded_elements = torch.cat(encoded_elements_list, dim=0)  # (n_spin * n_basis^2, latent_dim)

        # Apply layer norm
        encoded_elements = self.norm_kv(encoded_elements)

        # Add batch dimension for attention
        encoded_elements = encoded_elements.unsqueeze(0)  # (1, n_elements, latent_dim)
        queries = self.query_tokens.unsqueeze(0)  # (1, n_query, latent_dim)
        queries = self.norm_q(queries)

        # Cross-attention pooling
        attended, _ = self.cross_attention(
            queries,
            encoded_elements,
            encoded_elements,
        )  # (1, n_query, latent_dim)

        # Remove batch dimension and project
        output = attended.squeeze(0)  # (n_query, latent_dim)
        output = self.output_proj(output)

        return output

    def forward_batch(
        self,
        rho_list: list[Tensor],
        basis_metadata_list: list[dict[str, Tensor]],
        geometry_features_list: Optional[list[Tensor]] = None,
    ) -> Tensor:
        """
        Encode a batch of density matrices with potentially different sizes.

        Args:
            rho_list: List of density matrices, each (n_spin, n_basis_i, n_basis_i)
            basis_metadata_list: List of basis metadata dicts
            geometry_features_list: Optional list of geometry features

        Returns:
            Batched latent representations, shape (batch, n_query_tokens, latent_dim)
        """
        batch_outputs = []
        for i, (rho, metadata) in enumerate(zip(rho_list, basis_metadata_list)):
            geom_feat = geometry_features_list[i] if geometry_features_list else None
            output = self.forward(rho, metadata, geom_feat)
            batch_outputs.append(output)

        return torch.stack(batch_outputs, dim=0)


class UniversalBlockEncoder(nn.Module):
    """
    Encode atom-pair blocks of the density matrix.

    Instead of encoding individual elements, this encoder works at the
    block level, encoding the interaction between pairs of atoms.

    Args:
        latent_dim: Dimension of block latent representation
        max_l: Maximum angular momentum
        n_heads: Number of attention heads
    """

    def __init__(
        self,
        latent_dim: int = 64,
        max_l: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Element embedding
        self.element_embed = nn.Embedding(100, 32)
        self.l_embed = nn.Embedding(max_l + 1, 16)

        # Block encoder (processes flattened block)
        self.block_encoder = nn.Sequential(
            nn.Linear(2, 64),  # real + imag
            nn.SiLU(),
            nn.Linear(64, latent_dim),
        )

        # Attention for variable-size blocks
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Query token for pooling
        self.query = nn.Parameter(torch.randn(1, latent_dim))

    def encode_block(
        self,
        rho_block: Tensor,
        Z_A: int,
        Z_B: int,
        l_A: Tensor,
        l_B: Tensor,
    ) -> Tensor:
        """
        Encode a single atom-pair block.

        Args:
            rho_block: Density block, shape (n_basis_A, n_basis_B) complex
            Z_A, Z_B: Atomic numbers of atoms A and B
            l_A, l_B: Angular momenta for basis functions on A and B

        Returns:
            Block encoding, shape (latent_dim,)
        """
        device = rho_block.device
        n_A, n_B = rho_block.shape

        # Encode each element
        elements = []
        for i in range(n_A):
            for j in range(n_B):
                val = rho_block[i, j]
                elem_features = self.block_encoder(
                    torch.tensor([val.real, val.imag], device=device)
                )
                elements.append(elem_features)

        if not elements:
            return torch.zeros(self.latent_dim, device=device)

        # Stack and apply attention pooling
        elements = torch.stack(elements, dim=0).unsqueeze(0)  # (1, n_A*n_B, latent_dim)
        query = self.query.unsqueeze(0).to(device)  # (1, 1, latent_dim)

        pooled, _ = self.attention(query, elements, elements)
        return pooled.squeeze(0).squeeze(0)  # (latent_dim,)
