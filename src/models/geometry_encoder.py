"""
E3NN-based SO(3)-equivariant geometry encoder.

This module provides the GeometryEncoder class which encodes molecular
geometries into equivariant feature representations using E3NN convolutions.
The output features transform correctly under rotations, enabling the
model to generalize across molecular orientations.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, spherical_harmonics

from ..utils.graph import MolecularGraph, build_molecular_graph


class RadialBasisEncoding(nn.Module):
    """
    Encode edge distances using learnable radial basis functions.

    Uses Bessel basis with smooth cutoff, followed by an MLP.

    Args:
        num_basis: Number of radial basis functions
        cutoff: Cutoff distance in Angstroms
        hidden_dim: Hidden dimension of the MLP
        output_dim: Output dimension
    """

    def __init__(
        self,
        num_basis: int = 8,
        cutoff: float = 5.0,
        hidden_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        # Learnable frequencies for Bessel basis
        self.register_buffer(
            "frequencies",
            torch.arange(1, num_basis + 1, dtype=torch.float32) * torch.pi / cutoff,
        )

        # MLP to process radial features
        self.mlp = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, edge_len: Tensor) -> Tensor:
        """
        Encode edge lengths to radial features.

        Args:
            edge_len: Edge lengths, shape (n_edges,)

        Returns:
            Radial features, shape (n_edges, output_dim)
        """
        # Bessel basis: sin(n * pi * r / cutoff) / r
        r = edge_len.unsqueeze(-1)  # (n_edges, 1)
        r_safe = torch.clamp(r, min=1e-8)

        # Compute basis
        basis = torch.sin(self.frequencies * r) / r_safe  # (n_edges, num_basis)

        # Apply smooth cutoff
        cutoff_fn = self._smooth_cutoff(edge_len)
        basis = basis * cutoff_fn.unsqueeze(-1)

        return self.mlp(basis)

    def _smooth_cutoff(self, r: Tensor) -> Tensor:
        """Smooth cosine cutoff function."""
        x = r / self.cutoff
        return torch.where(
            x < 1.0,
            0.5 * (1.0 + torch.cos(torch.pi * x)),
            torch.zeros_like(x),
        )


class EquivariantConvolution(nn.Module):
    """
    Single equivariant convolution layer using E3NN tensor products.

    Performs message passing on a molecular graph with SO(3)-equivariant
    features. Uses spherical harmonics for angular encoding of edges.

    Args:
        irreps_in: Input irreps for node features
        irreps_out: Output irreps for node features
        irreps_edge: Irreps for edge spherical harmonics (typically "1x0e + 1x1o + 1x2e")
        radial_dim: Dimension of radial embedding
        num_neighbors: Expected number of neighbors (for normalization)
    """

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        irreps_edge: Union[str, Irreps] = "1x0e + 1x1o + 1x2e",
        radial_dim: int = 64,
        num_neighbors: float = 10.0,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_edge = Irreps(irreps_edge)
        self.num_neighbors = num_neighbors

        # Tensor product for message computation
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_edge,
            self.irreps_out,
            shared_weights=False,
        )

        # MLP to generate tensor product weights from radial features
        self.weight_nn = FullyConnectedNet(
            [radial_dim, 64, self.tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Self-connection (linear transformation of input)
        self.self_connection = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_sh: Tensor,
        edge_radial: Tensor,
    ) -> Tensor:
        """
        Forward pass of equivariant convolution.

        Args:
            node_features: Node features, shape (n_nodes, irreps_in.dim)
            edge_index: Edge indices, shape (2, n_edges)
            edge_sh: Edge spherical harmonics, shape (n_edges, irreps_edge.dim)
            edge_radial: Radial features, shape (n_edges, radial_dim)

        Returns:
            Updated node features, shape (n_nodes, irreps_out.dim)
        """
        src, dst = edge_index
        n_nodes = node_features.shape[0]

        # Get source node features for each edge
        src_features = node_features[src]  # (n_edges, irreps_in.dim)

        # Compute tensor product weights
        weights = self.weight_nn(edge_radial)  # (n_edges, tp.weight_numel)

        # Compute messages via tensor product
        messages = self.tp(src_features, edge_sh, weights)  # (n_edges, irreps_out.dim)

        # Aggregate messages at destination nodes
        aggregated = torch.zeros(
            n_nodes, self.irreps_out.dim, device=node_features.device
        )
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Normalize by number of neighbors
        aggregated = aggregated / self.num_neighbors**0.5

        # Add self-connection
        output = aggregated + self.self_connection(node_features)

        return output


class ScalarActivation(nn.Module):
    """
    Apply activation only to L=0 (scalar) features, leaving others unchanged.

    For equivariant networks, we can only apply nonlinearities to scalars.
    Non-scalar features (L>0) must pass through unchanged to preserve equivariance.
    """

    def __init__(self, irreps: Irreps, activation=torch.nn.functional.silu):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.activation = activation

        # Precompute slices for scalar and non-scalar features
        self.scalar_slices = []
        self.nonscalar_slices = []
        idx = 0
        for mul, ir in self.irreps:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 0:
                self.scalar_slices.append((idx, idx + dim))
            else:
                self.nonscalar_slices.append((idx, idx + dim))
            idx += dim

    def forward(self, x: Tensor) -> Tensor:
        """Apply activation to scalar features only."""
        if not self.scalar_slices:
            return x

        # Clone to avoid in-place modification
        out = x.clone()

        # Apply activation to scalar slices
        for start, end in self.scalar_slices:
            out[..., start:end] = self.activation(x[..., start:end])

        return out


class GeometryEncoder(nn.Module):
    """
    SO(3)-equivariant encoder for molecular geometries.

    Uses E3NN to encode atomic positions and types into equivariant
    features that transform correctly under rotations. This is the
    core component for generalizing across molecular orientations.

    The encoder outputs:
    - L=0 (scalar) features: Local chemical environment
    - L=1 (vector) features: Directional information for field coupling
    - L=2 (tensor) features: d-orbital-like structure

    Args:
        irreps_out: Output irreps (e.g., "32x0e + 16x1o + 8x2e")
        num_layers: Number of convolution layers
        max_radius: Cutoff radius for neighbor finding
        num_basis: Number of radial basis functions
        node_embed_dim: Dimension of initial node embedding
        num_neighbors: Expected number of neighbors
        max_atomic_number: Maximum atomic number to embed
    """

    def __init__(
        self,
        irreps_out: str = "32x0e + 16x1o + 8x2e",
        num_layers: int = 4,
        max_radius: float = 5.0,
        num_basis: int = 8,
        node_embed_dim: int = 64,
        num_neighbors: float = 10.0,
        max_atomic_number: int = 100,
    ):
        super().__init__()
        self.irreps_out = Irreps(irreps_out)
        self.num_layers = num_layers
        self.max_radius = max_radius

        # Edge spherical harmonics irreps (up to L=2 for 2e output)
        self.irreps_edge = Irreps("1x0e + 1x1o + 1x2e")

        # Atomic number embedding
        self.atom_embedding = nn.Embedding(max_atomic_number, node_embed_dim)

        # Initial irreps: just scalars from embedding
        irreps_init = Irreps(f"{node_embed_dim}x0e")

        # Radial basis encoding
        self.radial_encoding = RadialBasisEncoding(
            num_basis=num_basis,
            cutoff=max_radius,
            hidden_dim=64,
            output_dim=64,
        )

        # Build layers with progressive irreps
        self.convolutions = nn.ModuleList()
        self.activations = nn.ModuleList()

        irreps_hidden = self.irreps_out

        for i in range(num_layers):
            if i == 0:
                irreps_in = irreps_init
            else:
                irreps_in = irreps_hidden

            if i == num_layers - 1:
                irreps_layer_out = self.irreps_out
            else:
                irreps_layer_out = irreps_hidden

            self.convolutions.append(
                EquivariantConvolution(
                    irreps_in=irreps_in,
                    irreps_out=irreps_layer_out,
                    irreps_edge=self.irreps_edge,
                    radial_dim=64,
                    num_neighbors=num_neighbors,
                )
            )

            # Activation for intermediate layers (not final layer)
            if i < num_layers - 1:
                self.activations.append(ScalarActivation(irreps_layer_out))
            else:
                self.activations.append(nn.Identity())

    def forward(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_vec: Optional[Tensor] = None,
        edge_len: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode molecular geometry to equivariant features.

        Args:
            positions: Atomic positions, shape (n_atoms, 3)
            atomic_numbers: Atomic numbers, shape (n_atoms,)
            edge_index: Optional precomputed edge indices, shape (2, n_edges)
            edge_vec: Optional precomputed edge vectors, shape (n_edges, 3)
            edge_len: Optional precomputed edge lengths, shape (n_edges,)

        Returns:
            Node features with output irreps, shape (n_atoms, irreps_out.dim)
        """
        n_atoms = positions.shape[0]
        device = positions.device

        # Build graph if not provided
        if edge_index is None:
            graph = build_molecular_graph(positions, atomic_numbers, self.max_radius)
            edge_index = graph.edge_index.to(device)
            edge_vec = graph.edge_vec.to(device)
            edge_len = graph.edge_len.to(device)

        # Initial node features from atomic number embedding
        node_features = self.atom_embedding(atomic_numbers)  # (n_atoms, node_embed_dim)

        # Compute edge spherical harmonics
        edge_unit = edge_vec / (edge_len.unsqueeze(-1) + 1e-8)
        edge_sh = spherical_harmonics(
            self.irreps_edge,
            edge_unit,
            normalize=True,
            normalization="component",
        )

        # Compute radial features
        edge_radial = self.radial_encoding(edge_len)

        # Apply convolution layers
        for conv, act in zip(self.convolutions, self.activations):
            node_features = conv(node_features, edge_index, edge_sh, edge_radial)
            node_features = act(node_features)

        return node_features

    def forward_with_graph(self, graph: MolecularGraph) -> Tensor:
        """
        Encode using a precomputed molecular graph.

        Args:
            graph: MolecularGraph object

        Returns:
            Node features with output irreps
        """
        return self.forward(
            graph.positions,
            graph.atomic_numbers,
            graph.edge_index,
            graph.edge_vec,
            graph.edge_len,
        )

    def get_invariant_features(self, node_features: Tensor) -> Tensor:
        """
        Extract only the L=0 (scalar/invariant) components.

        Args:
            node_features: Full equivariant features

        Returns:
            Invariant features only
        """
        # Get slice for L=0 components
        start = 0
        invariant_features = []
        for mul, ir in self.irreps_out:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 0:
                invariant_features.append(node_features[..., start : start + dim])
            start += dim

        return torch.cat(invariant_features, dim=-1)

    def get_vector_features(self, node_features: Tensor) -> Tensor:
        """
        Extract only the L=1 (vector) components.

        Args:
            node_features: Full equivariant features

        Returns:
            Vector features only, shape (..., n_vectors, 3)
        """
        start = 0
        vector_features = []
        for mul, ir in self.irreps_out:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 1:
                # Reshape to (batch, mul, 3)
                vecs = node_features[..., start : start + dim]
                vecs = vecs.reshape(*vecs.shape[:-1], mul, 3)
                vector_features.append(vecs)
            start += dim

        if not vector_features:
            return torch.zeros(*node_features.shape[:-1], 0, 3, device=node_features.device)

        return torch.cat(vector_features, dim=-2)


def verify_equivariance(encoder: GeometryEncoder, positions: Tensor, atomic_numbers: Tensor) -> float:
    """
    Verify SO(3) equivariance of the encoder.

    Args:
        encoder: GeometryEncoder instance
        positions: Atomic positions
        atomic_numbers: Atomic numbers

    Returns:
        Maximum equivariance error
    """
    # Random rotation matrix
    angles = torch.randn(3) * 2 * torch.pi
    R = o3.angles_to_matrix(*angles)

    # Encode original
    features_original = encoder(positions, atomic_numbers)

    # Encode rotated
    positions_rotated = positions @ R.T
    features_rotated = encoder(positions_rotated, atomic_numbers)

    # Apply Wigner-D rotation to original features
    D = encoder.irreps_out.D_from_matrix(R)
    features_transformed = features_original @ D.T

    # Compute error
    error = (features_rotated - features_transformed).abs().max().item()
    return error
