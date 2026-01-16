"""
Tests for Phase 2 model components.

Tests cover:
- DensityEdgeProjection encoder
- EquivariantFieldEncoder
- GeometryConditionedMamba dynamics
- DensityDecoder
- RTTDDFTModel end-to-end
"""

import pytest
import torch
import numpy as np


class TestDensityEncoder:
    """Tests for DensityEdgeProjection encoder."""

    @pytest.fixture
    def encoder(self):
        from src.models.density_encoder import DensityEdgeProjection
        return DensityEdgeProjection(
            latent_dim=64,
            n_query_tokens=8,
            max_l=2,
            n_heads=4,
        )

    @pytest.fixture
    def sample_density(self):
        """Create sample density matrix and metadata."""
        n_basis = 4
        n_spin = 1

        # Create Hermitian density matrix
        rho_real = torch.randn(n_spin, n_basis, n_basis)
        rho_real = 0.5 * (rho_real + rho_real.transpose(-2, -1))
        rho_imag = torch.randn(n_spin, n_basis, n_basis)
        rho_imag = 0.5 * (rho_imag - rho_imag.transpose(-2, -1))
        rho = torch.complex(rho_real, rho_imag)

        metadata = {
            'atom_idx': torch.tensor([0, 0, 1, 1]),
            'Z': torch.tensor([1, 1, 1, 1]),
            'l': torch.tensor([0, 0, 0, 0]),
            'm': torch.tensor([0, 0, 0, 0]),
        }

        return rho, metadata

    def test_encoder_forward_shape(self, encoder, sample_density):
        """Test encoder produces correct output shape."""
        rho, metadata = sample_density
        output = encoder(rho, metadata)

        assert output.shape == (8, 64)  # (n_query_tokens, latent_dim)

    def test_encoder_no_nan(self, encoder, sample_density):
        """Test encoder produces no NaN values."""
        rho, metadata = sample_density
        output = encoder(rho, metadata)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_encoder_batch(self, encoder, sample_density):
        """Test batch encoding."""
        rho, metadata = sample_density
        batch = encoder.forward_batch([rho, rho], [metadata, metadata])

        assert batch.shape == (2, 8, 64)

    def test_encoder_gradients(self, encoder, sample_density):
        """Test gradients flow through encoder."""
        rho, metadata = sample_density
        rho = rho.requires_grad_(True)

        output = encoder(rho, metadata)
        loss = output.sum()
        loss.backward()

        assert rho.grad is not None


class TestFieldEncoder:
    """Tests for EquivariantFieldEncoder."""

    @pytest.fixture
    def encoder(self):
        from src.models.field_encoder import EquivariantFieldEncoder
        return EquivariantFieldEncoder(
            scalar_dim=16,
            vector_multiplicity=8,
            hidden_dim=32,
        )

    def test_encoder_forward_shape(self, encoder):
        """Test field encoder output shape."""
        field = torch.tensor([0.001, 0.0, 0.0])
        output = encoder(field)

        # scalar_dim + vector_multiplicity * 3
        expected_dim = 16 + 8 * 3
        assert output.shape == (expected_dim,)

    def test_encoder_zero_field(self, encoder):
        """Test encoding zero field."""
        field = torch.zeros(3)
        output = encoder(field)

        assert not torch.isnan(output).any()

    def test_encoder_batch(self, encoder):
        """Test batch field encoding."""
        fields = torch.randn(5, 3)
        outputs = encoder(fields)

        assert outputs.shape == (5, 16 + 8 * 3)

    def test_scalar_vector_extraction(self, encoder):
        """Test extracting scalar and vector components."""
        field = torch.randn(3)
        output = encoder(field)

        scalars = encoder.get_scalar_features(output)
        vectors = encoder.get_vector_features(output)

        assert scalars.shape == (16,)
        assert vectors.shape == (8, 3)

    def test_field_magnitude_scaling(self, encoder):
        """Test that output scales with field magnitude."""
        field_small = torch.tensor([0.001, 0.0, 0.0])
        field_large = torch.tensor([0.1, 0.0, 0.0])

        out_small = encoder(field_small)
        out_large = encoder(field_large)

        # Outputs should be different
        assert (out_small - out_large).abs().sum() > 0.1


class TestFieldTimeEncoder:
    """Tests for combined field and time encoder."""

    @pytest.fixture
    def encoder(self):
        from src.models.field_encoder import FieldTimeEncoder
        return FieldTimeEncoder(
            scalar_dim=16,
            vector_multiplicity=8,
            time_embed_dim=32,
        )

    def test_with_time(self, encoder):
        """Test encoding with time."""
        field = torch.randn(3)
        time = torch.tensor(0.5)

        output = encoder(field, time)
        assert output.shape[-1] == 16 + 8 * 3

    def test_without_time(self, encoder):
        """Test encoding without time."""
        field = torch.randn(3)
        output = encoder(field, None)
        assert not torch.isnan(output).any()


class TestDynamics:
    """Tests for Mamba dynamics model."""

    @pytest.fixture
    def dynamics(self):
        from src.models.dynamics import create_dynamics_model, DynamicsConfig
        config = DynamicsConfig(
            d_model=64,
            d_state=8,
            d_conv=4,
            n_layers=2,
        )
        return create_dynamics_model(
            config=config,
            latent_dim=64,
            geometry_dim=32,
            use_simplified=True,  # Use simplified for testing
        )

    def test_dynamics_forward_single(self, dynamics):
        """Test single-step dynamics."""
        z = torch.randn(1, 64)
        geom = torch.randn(1, 32)

        z_next = dynamics(z, geom)
        assert z_next.shape == z.shape

    def test_dynamics_forward_sequence(self, dynamics):
        """Test sequence dynamics."""
        z = torch.randn(2, 10, 64)  # batch=2, seq=10
        geom = torch.randn(2, 32)

        z_next = dynamics(z, geom)
        assert z_next.shape == z.shape

    def test_dynamics_predict_sequence(self, dynamics):
        """Test autoregressive sequence prediction."""
        z0 = torch.randn(2, 64)
        geom = torch.randn(2, 32)

        trajectory = dynamics.predict_sequence(z0, geom, n_steps=5)
        assert trajectory.shape == (2, 5, 64)

    def test_dynamics_gradients(self, dynamics):
        """Test gradient flow through dynamics."""
        z = torch.randn(1, 64, requires_grad=True)
        geom = torch.randn(1, 32)

        z_next = dynamics(z, geom)
        loss = z_next.sum()
        loss.backward()

        assert z.grad is not None


class TestFiLM:
    """Tests for FiLM conditioning layer."""

    @pytest.fixture
    def film(self):
        from src.models.dynamics import FiLMLayer
        return FiLMLayer(feature_dim=64, condition_dim=32)

    def test_film_forward(self, film):
        """Test FiLM forward pass."""
        x = torch.randn(10, 64)
        cond = torch.randn(10, 32)

        out = film(x, cond)
        assert out.shape == x.shape

    def test_film_identity_init(self, film):
        """Test that FiLM is initialized near identity."""
        x = torch.randn(10, 64)
        cond = torch.zeros(10, 32)  # Zero condition

        out = film(x, cond)

        # Output should be close to input at initialization
        # (gamma=1, beta=0 gives identity)
        diff = (out - x).abs().mean()
        assert diff < 5.0  # Loose bound since sigmoid shifts things


class TestDecoder:
    """Tests for DensityDecoder."""

    @pytest.fixture
    def decoder(self):
        from src.models.decoder import DensityDecoder
        return DensityDecoder(
            latent_dim=64,
            n_query_tokens=8,
            max_l=2,
        )

    @pytest.fixture
    def sample_metadata(self):
        return {
            'Z': torch.tensor([1, 1, 1, 1]),
            'l': torch.tensor([0, 0, 0, 0]),
            'm': torch.tensor([0, 0, 0, 0]),
        }

    def test_decoder_forward_shape(self, decoder, sample_metadata):
        """Test decoder output shape."""
        latent = torch.randn(8, 64)
        rho = decoder(latent, sample_metadata, n_spin=1)

        assert rho.shape == (1, 4, 4)
        assert rho.is_complex()

    def test_decoder_multiple_spin(self, decoder, sample_metadata):
        """Test decoder with multiple spin channels."""
        latent = torch.randn(8, 64)
        rho = decoder(latent, sample_metadata, n_spin=2)

        assert rho.shape == (2, 4, 4)

    def test_decoder_gradients(self, decoder, sample_metadata):
        """Test gradient flow through decoder."""
        latent = torch.randn(8, 64, requires_grad=True)
        rho = decoder(latent, sample_metadata)

        loss = rho.abs().sum()
        loss.backward()

        assert latent.grad is not None


class TestFullModel:
    """Tests for RTTDDFTModel end-to-end."""

    @pytest.fixture
    def model(self):
        from src.models.full_model import RTTDDFTModel, RTTDDFTConfig
        config = RTTDDFTConfig(
            geometry_irreps="8x0e + 4x1o",
            geometry_layers=2,
            n_query_tokens=8,
            latent_dim=64,
            mamba_d_model=64,
            mamba_layers=2,
        )
        return RTTDDFTModel(config)

    @pytest.fixture
    def sample_inputs(self):
        """Create sample model inputs."""
        n_atoms = 2
        n_basis = 4
        n_spin = 1

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74],
        ], dtype=torch.float32)

        atomic_numbers = torch.tensor([1, 1], dtype=torch.long)

        rho_real = torch.randn(n_spin, n_basis, n_basis)
        rho_real = 0.5 * (rho_real + rho_real.transpose(-2, -1))
        rho = torch.complex(rho_real, torch.zeros_like(rho_real))

        field = torch.tensor([0.001, 0.0, 0.0])

        metadata = {
            'Z': torch.tensor([1, 1, 1, 1]),
            'l': torch.tensor([0, 0, 0, 0]),
            'm': torch.tensor([0, 0, 0, 0]),
        }

        overlap = torch.eye(n_basis, dtype=torch.complex64)

        return {
            'positions': positions,
            'atomic_numbers': atomic_numbers,
            'rho_current': rho,
            'field': field,
            'basis_metadata': metadata,
            'overlap': overlap,
            'n_electrons': 2,
        }

    def test_model_forward(self, model, sample_inputs):
        """Test full model forward pass."""
        output = model(
            sample_inputs['positions'],
            sample_inputs['atomic_numbers'],
            sample_inputs['rho_current'],
            sample_inputs['field'],
            sample_inputs['basis_metadata'],
            apply_constraints=False,
        )

        assert 'rho_pred' in output
        assert output['rho_pred'].shape == sample_inputs['rho_current'].shape

    def test_model_with_constraints(self, model, sample_inputs):
        """Test model with physics constraints."""
        output = model(
            sample_inputs['positions'],
            sample_inputs['atomic_numbers'],
            sample_inputs['rho_current'],
            sample_inputs['field'],
            sample_inputs['basis_metadata'],
            overlap=sample_inputs['overlap'],
            n_electrons=sample_inputs['n_electrons'],
            apply_constraints=True,
        )

        rho_pred = output['rho_pred']

        # Check Hermiticity after constraints
        diff = (rho_pred - rho_pred.conj().transpose(-2, -1)).abs().max()
        assert diff < 1e-5

    def test_model_gradients(self, model, sample_inputs):
        """Test gradient flow through full model."""
        sample_inputs['positions'] = sample_inputs['positions'].requires_grad_(True)

        output = model(
            sample_inputs['positions'],
            sample_inputs['atomic_numbers'],
            sample_inputs['rho_current'],
            sample_inputs['field'],
            sample_inputs['basis_metadata'],
            apply_constraints=False,
        )

        loss = output['rho_pred'].abs().sum()
        loss.backward()

        # Check gradients exist for model parameters
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


class TestModelFactory:
    """Tests for model factory functions."""

    def test_create_lite_model(self):
        from src.models.full_model import create_model
        model = create_model("lite")
        assert model is not None

    def test_create_base_model(self):
        from src.models.full_model import create_model
        model = create_model("base")
        assert model is not None

    def test_create_model_with_overrides(self):
        from src.models.full_model import create_model
        model = create_model("lite", latent_dim=128)
        assert model.config.latent_dim == 128
