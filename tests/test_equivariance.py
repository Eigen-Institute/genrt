"""
Tests for SO(3) equivariance of the geometry encoder.

These tests verify that the E3NN-based geometry encoder correctly
transforms under rotations, which is essential for generalizing
across molecular orientations.
"""

import pytest
import torch
import numpy as np
from e3nn import o3

from src.models.geometry_encoder import GeometryEncoder, verify_equivariance


class TestGeometryEncoderEquivariance:
    """Test SO(3) equivariance of GeometryEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a GeometryEncoder instance for testing."""
        return GeometryEncoder(
            irreps_out="8x0e + 4x1o + 2x2e",
            num_layers=2,
            max_radius=5.0,
            num_basis=4,
            node_embed_dim=16,
        )

    def test_forward_shape(self, encoder, h2_geometry):
        """Test that forward pass produces correct output shape."""
        output = encoder(
            h2_geometry["positions"],
            h2_geometry["atomic_numbers"],
        )

        n_atoms = h2_geometry["positions"].shape[0]
        expected_dim = encoder.irreps_out.dim

        assert output.shape == (n_atoms, expected_dim)

    def test_forward_no_nan(self, encoder, h2_geometry):
        """Test that forward pass produces no NaN values."""
        output = encoder(
            h2_geometry["positions"],
            h2_geometry["atomic_numbers"],
        )

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_equivariance_h2(self, encoder, h2_geometry, random_rotation_matrix):
        """Test SO(3) equivariance on H2 molecule."""
        positions = h2_geometry["positions"]
        atomic_numbers = h2_geometry["atomic_numbers"]

        # Get random rotation
        R = random_rotation_matrix()

        # Encode original geometry
        features_original = encoder(positions, atomic_numbers)

        # Rotate geometry and encode
        positions_rotated = positions @ R.T
        features_rotated = encoder(positions_rotated, atomic_numbers)

        # Apply Wigner-D rotation to original features
        D = encoder.irreps_out.D_from_matrix(R)
        features_transformed = features_original @ D.T

        # Check equivariance
        error = (features_rotated - features_transformed).abs().max().item()
        assert error < 1e-4, f"Equivariance error: {error}"

    def test_equivariance_h2o(self, encoder, h2o_geometry, random_rotation_matrix):
        """Test SO(3) equivariance on H2O molecule."""
        positions = h2o_geometry["positions"]
        atomic_numbers = h2o_geometry["atomic_numbers"]

        R = random_rotation_matrix()

        features_original = encoder(positions, atomic_numbers)
        positions_rotated = positions @ R.T
        features_rotated = encoder(positions_rotated, atomic_numbers)

        D = encoder.irreps_out.D_from_matrix(R)
        features_transformed = features_original @ D.T

        error = (features_rotated - features_transformed).abs().max().item()
        assert error < 1e-4, f"Equivariance error: {error}"

    def test_equivariance_multiple_rotations(self, encoder, h2_geometry):
        """Test equivariance with multiple random rotations."""
        positions = h2_geometry["positions"]
        atomic_numbers = h2_geometry["atomic_numbers"]

        errors = []
        for _ in range(10):
            error = verify_equivariance(encoder, positions, atomic_numbers)
            errors.append(error)

        max_error = max(errors)
        mean_error = sum(errors) / len(errors)

        assert max_error < 1e-4, f"Max equivariance error: {max_error}"
        assert mean_error < 1e-5, f"Mean equivariance error: {mean_error}"

    def test_invariant_features_are_invariant(self, encoder, h2_geometry, random_rotation_matrix):
        """Test that L=0 features are truly invariant (unchanged by rotation)."""
        positions = h2_geometry["positions"]
        atomic_numbers = h2_geometry["atomic_numbers"]

        R = random_rotation_matrix()

        # Get features
        features_original = encoder(positions, atomic_numbers)
        positions_rotated = positions @ R.T
        features_rotated = encoder(positions_rotated, atomic_numbers)

        # Extract invariant (L=0) components
        invariants_original = encoder.get_invariant_features(features_original)
        invariants_rotated = encoder.get_invariant_features(features_rotated)

        # Invariants should be unchanged
        error = (invariants_original - invariants_rotated).abs().max().item()
        assert error < 1e-5, f"Invariant features changed by rotation: {error}"

    def test_vector_features_transform_correctly(self, encoder, h2_geometry, random_rotation_matrix):
        """Test that L=1 features transform as vectors."""
        positions = h2_geometry["positions"]
        atomic_numbers = h2_geometry["atomic_numbers"]

        R = random_rotation_matrix()

        # Get features
        features_original = encoder(positions, atomic_numbers)
        positions_rotated = positions @ R.T
        features_rotated = encoder(positions_rotated, atomic_numbers)

        # Extract vector (L=1) components
        vectors_original = encoder.get_vector_features(features_original)
        vectors_rotated = encoder.get_vector_features(features_rotated)

        if vectors_original.numel() > 0:
            # Apply rotation to original vectors
            # vectors shape: (n_atoms, n_vectors, 3)
            vectors_transformed = torch.einsum("...i,ji->...j", vectors_original, R)

            error = (vectors_rotated - vectors_transformed).abs().max().item()
            assert error < 1e-4, f"Vector features transformation error: {error}"

    def test_translation_invariance(self, encoder, h2_geometry):
        """Test that features are invariant to translations."""
        positions = h2_geometry["positions"]
        atomic_numbers = h2_geometry["atomic_numbers"]

        # Random translation
        translation = torch.randn(3) * 10.0

        # Encode original and translated
        features_original = encoder(positions, atomic_numbers)
        features_translated = encoder(positions + translation, atomic_numbers)

        # Features should be identical
        error = (features_original - features_translated).abs().max().item()
        assert error < 1e-5, f"Translation invariance error: {error}"

    def test_permutation_equivariance(self, encoder, h2_geometry):
        """Test that permuting identical atoms gives permuted features."""
        positions = h2_geometry["positions"]
        atomic_numbers = h2_geometry["atomic_numbers"]

        # H2 has two identical atoms, so permutation should work
        # Swap atom order
        positions_permuted = positions.flip(0)
        atomic_numbers_permuted = atomic_numbers.flip(0)

        features_original = encoder(positions, atomic_numbers)
        features_permuted = encoder(positions_permuted, atomic_numbers_permuted)

        # Features should be permuted
        features_original_permuted = features_original.flip(0)
        error = (features_permuted - features_original_permuted).abs().max().item()
        assert error < 1e-5, f"Permutation equivariance error: {error}"


class TestGeometryEncoderDifferentMolecules:
    """Test GeometryEncoder on different molecules."""

    @pytest.fixture
    def encoder(self):
        return GeometryEncoder(
            irreps_out="16x0e + 8x1o + 4x2e",
            num_layers=3,
            max_radius=5.0,
        )

    def test_h2_plus(self, encoder, h2_plus_geometry):
        """Test encoder on H2+."""
        output = encoder(
            h2_plus_geometry["positions"],
            h2_plus_geometry["atomic_numbers"],
        )
        assert not torch.isnan(output).any()

    def test_lih(self, encoder, lih_geometry):
        """Test encoder on LiH."""
        output = encoder(
            lih_geometry["positions"],
            lih_geometry["atomic_numbers"],
        )
        assert not torch.isnan(output).any()

    def test_h2o(self, encoder, h2o_geometry):
        """Test encoder on H2O."""
        output = encoder(
            h2o_geometry["positions"],
            h2o_geometry["atomic_numbers"],
        )
        assert not torch.isnan(output).any()

    def test_different_bond_lengths(self, encoder):
        """Test encoder on H2 with different bond lengths."""
        atomic_numbers = torch.tensor([1, 1], dtype=torch.long)

        bond_lengths = [0.5, 0.74, 1.0, 1.5, 2.0]
        outputs = []

        for bl in bond_lengths:
            positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, bl]], dtype=torch.float32)
            output = encoder(positions, atomic_numbers)
            outputs.append(output)
            assert not torch.isnan(output).any()

        # Features should vary with bond length
        for i in range(len(outputs) - 1):
            diff = (outputs[i] - outputs[i + 1]).abs().sum().item()
            assert diff > 1e-3, "Features should change with bond length"


class TestGeometryEncoderGradients:
    """Test gradient flow through GeometryEncoder."""

    @pytest.fixture
    def encoder(self):
        return GeometryEncoder(
            irreps_out="8x0e + 4x1o",
            num_layers=2,
            max_radius=5.0,
        )

    def test_gradients_exist(self, encoder, h2_geometry):
        """Test that gradients flow through the encoder."""
        positions = h2_geometry["positions"].requires_grad_(True)
        atomic_numbers = h2_geometry["atomic_numbers"]

        output = encoder(positions, atomic_numbers)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are finite
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_position_gradients(self, encoder, h2_geometry):
        """Test that gradients w.r.t. positions exist."""
        positions = h2_geometry["positions"].clone().requires_grad_(True)
        atomic_numbers = h2_geometry["atomic_numbers"]

        output = encoder(positions, atomic_numbers)
        loss = output.sum()
        loss.backward()

        assert positions.grad is not None
        assert not torch.isnan(positions.grad).any()


class TestEquivarianceBuildIntegration:
    """Integration tests for equivariance in a training-like scenario."""

    def test_equivariance_preserved_after_optimization_step(self, h2_geometry):
        """Test that equivariance is preserved after parameter updates."""
        encoder = GeometryEncoder(
            irreps_out="8x0e + 4x1o",
            num_layers=2,
        )

        positions = h2_geometry["positions"]
        atomic_numbers = h2_geometry["atomic_numbers"]

        # Check equivariance before optimization
        error_before = verify_equivariance(encoder, positions, atomic_numbers)
        assert error_before < 1e-4

        # Simulate optimization step
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
        for _ in range(5):
            output = encoder(positions, atomic_numbers)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check equivariance after optimization
        error_after = verify_equivariance(encoder, positions, atomic_numbers)
        assert error_after < 1e-4, f"Equivariance broken after optimization: {error_after}"
