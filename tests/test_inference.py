"""
Tests for inference module.

Tests cover:
- Predictor rollout functionality
- Uncertainty estimation (ensemble, MC dropout)
- Adaptive re-anchoring controller
"""

import pytest
import torch
import torch.nn as nn


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, n_basis: int = 4):
        super().__init__()
        self.n_basis = n_basis
        self.linear = nn.Linear(n_basis * n_basis * 2, n_basis * n_basis * 2)

    def forward(self, batch):
        rho = batch['density']
        batch_size = rho.shape[0]
        n = self.n_basis

        # Simple transformation
        rho_flat = torch.cat([rho.real.reshape(batch_size, -1),
                              rho.imag.reshape(batch_size, -1)], dim=-1)
        out = self.linear(rho_flat)

        # Reshape and make Hermitian
        rho_real = out[:, :n*n].reshape(batch_size, n, n)
        rho_imag = out[:, n*n:].reshape(batch_size, n, n)
        rho_out = rho_real + 1j * rho_imag
        rho_out = 0.5 * (rho_out + rho_out.conj().transpose(-2, -1))

        return rho_out.squeeze(0) if batch_size == 1 else rho_out


class TestPredictor:
    """Tests for prediction functionality."""

    @pytest.fixture
    def model(self):
        return MockModel(n_basis=4)

    @pytest.fixture
    def geometry(self):
        return {
            'positions': torch.randn(2, 3),
            'atomic_numbers': torch.tensor([1, 1]),
        }

    @pytest.fixture
    def initial_density(self):
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        return 0.5 * (rho + rho.conj().T)

    def test_predictor_creation(self, model):
        """Test predictor can be created."""
        from src.inference.predictor import Predictor

        predictor = Predictor(model)
        assert predictor.model is model

    def test_rollout_config(self):
        """Test rollout configuration."""
        from src.inference.predictor import RolloutConfig

        config = RolloutConfig(
            max_steps=100,
            apply_physics_projection=True,
            checkpoint_interval=50
        )
        assert config.max_steps == 100
        assert config.apply_physics_projection
        assert config.checkpoint_interval == 50

    def test_streaming_predictor(self, model, geometry, initial_density):
        """Test streaming predictor."""
        from src.inference.predictor import StreamingPredictor

        predictor = StreamingPredictor(model)
        predictor.initialize(initial_density, geometry)

        n = initial_density.shape[0]
        overlap = torch.eye(n, dtype=torch.complex64)
        field = torch.randn(3)

        # Step forward
        rho_next = predictor.step(field, overlap, n_electrons=2, apply_projection=False)

        assert rho_next.shape == initial_density.shape
        assert predictor._step_count == 1

    def test_streaming_predictor_state(self, model, geometry, initial_density):
        """Test streaming predictor state management."""
        from src.inference.predictor import StreamingPredictor

        predictor = StreamingPredictor(model)
        predictor.initialize(initial_density, geometry)

        n = initial_density.shape[0]
        overlap = torch.eye(n, dtype=torch.complex64)

        # Take some steps
        for _ in range(5):
            predictor.step(torch.randn(3), overlap, n_electrons=2, apply_projection=False)

        # Get state
        state = predictor.get_state()
        assert state['step_count'] == 5

        # Reset and verify
        predictor.reset()
        assert predictor._step_count == 0
        assert predictor._current_density is None


class TestUncertainty:
    """Tests for uncertainty estimation."""

    @pytest.fixture
    def models(self):
        return [MockModel(n_basis=4) for _ in range(3)]

    @pytest.fixture
    def batch(self):
        n = 4
        rho = torch.randn(1, n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
        return {'density': rho, 'field': torch.randn(1, 3)}

    def test_ensemble_uncertainty(self, models, batch):
        """Test ensemble uncertainty estimation."""
        from src.inference.uncertainty import EnsembleUncertainty

        ensemble = EnsembleUncertainty(models)
        result = ensemble(batch)

        assert result.mean is not None
        assert result.std is not None
        assert result.mean.shape == batch['density'].squeeze(0).shape

    def test_ensemble_with_samples(self, models, batch):
        """Test ensemble returns samples when requested."""
        from src.inference.uncertainty import EnsembleUncertainty

        ensemble = EnsembleUncertainty(models)
        result = ensemble(batch, return_samples=True)

        assert result.samples is not None
        assert result.samples.shape[0] == len(models)

    def test_mc_dropout_uncertainty(self, batch):
        """Test MC Dropout uncertainty."""
        from src.inference.uncertainty import MCDropoutUncertainty

        # Create model with dropout
        class ModelWithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(0.1)
                self.linear = nn.Linear(32, 32)

            def forward(self, batch):
                rho = batch['density']
                # Simple pass-through with dropout noise
                return rho + 0.01 * self.dropout(torch.randn_like(rho.real))

        model = ModelWithDropout()
        mc = MCDropoutUncertainty(model, n_samples=5)
        result = mc(batch)

        assert result.mean is not None
        assert result.std is not None

    def test_uncertainty_aggregator(self):
        """Test uncertainty aggregation methods."""
        from src.inference.uncertainty import UncertaintyAggregator

        uncertainty = torch.randn(4, 4).abs()

        # Frobenius
        agg_frob = UncertaintyAggregator(method="frobenius")
        scalar_frob = agg_frob(uncertainty)
        assert scalar_frob.dim() == 0

        # Max
        agg_max = UncertaintyAggregator(method="max")
        scalar_max = agg_max(uncertainty)
        assert scalar_max.item() == uncertainty.max().item()

        # Trace
        agg_trace = UncertaintyAggregator(method="trace")
        scalar_trace = agg_trace(uncertainty)
        expected = uncertainty.diagonal().sum()
        assert torch.isclose(scalar_trace, expected)

    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        from src.inference.uncertainty import UncertaintyAggregator

        agg = UncertaintyAggregator(method="frobenius")

        prediction = torch.randn(4, 4, dtype=torch.complex64)
        uncertainty = torch.ones(4, 4) * 0.1

        rel_uncert = agg.relative_uncertainty(uncertainty, prediction)
        assert rel_uncert.dim() == 0
        assert rel_uncert.item() > 0


class TestReanchor:
    """Tests for adaptive re-anchoring."""

    @pytest.fixture
    def overlap(self):
        n = 4
        return torch.eye(n, dtype=torch.complex64)

    @pytest.fixture
    def density(self):
        n = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        return 0.5 * (rho + rho.conj().T)

    def test_reanchor_thresholds(self):
        """Test threshold configuration."""
        from src.inference.reanchor import ReanchorThresholds

        thresholds = ReanchorThresholds(
            uncertainty=0.1,
            trace=0.05,
            max_ml_steps=200
        )
        assert thresholds.uncertainty == 0.1
        assert thresholds.trace == 0.05
        assert thresholds.max_ml_steps == 200

    def test_controller_continue(self, overlap, density):
        """Test controller continues when constraints satisfied."""
        from src.inference.reanchor import (
            AdaptiveReAnchorController,
            ReanchorThresholds,
            ReanchorReason
        )

        # Create well-behaved density (idempotent projection matrix)
        n = 4
        n_electrons = 2
        # Create proper idempotent density: project onto first n_electrons/2 orbitals
        rho = torch.zeros(n, n, dtype=torch.complex64)
        rho[0, 0] = 1.0
        rho[1, 1] = 1.0  # Trace = 2 = n_electrons

        # Use relaxed thresholds for idempotency (it's a soft constraint)
        thresholds = ReanchorThresholds(max_ml_steps=100, idempotency=1.0)
        controller = AdaptiveReAnchorController(thresholds)

        should_reanchor, reason, diag = controller.should_reanchor(
            rho, overlap, n_electrons
        )

        assert not should_reanchor
        assert reason == ReanchorReason.CONTINUE

    def test_controller_max_steps(self, overlap, density):
        """Test controller triggers on max steps."""
        from src.inference.reanchor import (
            AdaptiveReAnchorController,
            ReanchorThresholds,
            ReanchorReason
        )

        # Use relaxed thresholds so only max_steps triggers
        thresholds = ReanchorThresholds(
            max_ml_steps=5,
            idempotency=10.0,  # Relaxed
            trace=10.0,  # Relaxed
            hermiticity=10.0,  # Relaxed
        )
        controller = AdaptiveReAnchorController(thresholds)

        n_electrons = 2
        # Create valid density (idempotent)
        n = 4
        rho = torch.zeros(n, n, dtype=torch.complex64)
        rho[0, 0] = 1.0
        rho[1, 1] = 1.0  # Trace = 2 = n_electrons

        # Step through until max
        for _ in range(4):
            should_reanchor, reason, _ = controller.should_reanchor(
                rho, overlap, n_electrons
            )
            assert not should_reanchor

        # 5th step should trigger
        should_reanchor, reason, _ = controller.should_reanchor(
            rho, overlap, n_electrons
        )
        assert should_reanchor
        assert reason == ReanchorReason.MAX_STEPS

    def test_controller_trace_violation(self, overlap):
        """Test controller triggers on trace violation."""
        from src.inference.reanchor import (
            AdaptiveReAnchorController,
            ReanchorThresholds,
            ReanchorReason
        )

        thresholds = ReanchorThresholds(trace=0.01)
        controller = AdaptiveReAnchorController(thresholds)

        n = 4
        n_electrons = 2
        # Create density with wrong trace
        rho = torch.eye(n, dtype=torch.complex64) * 2.0  # Trace = 8, should be 2

        should_reanchor, reason, diag = controller.should_reanchor(
            rho, overlap, n_electrons
        )

        assert should_reanchor
        assert reason == ReanchorReason.TRACE_VIOLATION
        assert 'trace_error' in diag

    def test_controller_hermiticity_violation(self, overlap):
        """Test controller triggers on Hermiticity violation."""
        from src.inference.reanchor import (
            AdaptiveReAnchorController,
            ReanchorThresholds,
            ReanchorReason
        )

        thresholds = ReanchorThresholds(hermiticity=0.001, trace=10.0)
        controller = AdaptiveReAnchorController(thresholds)

        n = 4
        n_electrons = 2
        # Create non-Hermitian density
        rho = torch.randn(n, n, dtype=torch.complex64)  # Not Hermitian

        should_reanchor, reason, diag = controller.should_reanchor(
            rho, overlap, n_electrons
        )

        assert should_reanchor
        assert reason == ReanchorReason.HERMITICITY_VIOLATION
        assert 'hermiticity_error' in diag

    def test_controller_uncertainty_trigger(self, overlap):
        """Test controller triggers on high uncertainty."""
        from src.inference.reanchor import (
            AdaptiveReAnchorController,
            ReanchorThresholds,
            ReanchorReason
        )

        thresholds = ReanchorThresholds(uncertainty=0.01)
        controller = AdaptiveReAnchorController(thresholds)

        n = 4
        n_electrons = 2
        rho = torch.eye(n, dtype=torch.complex64) * (n_electrons / n)

        # Create high uncertainty
        uncertainty = torch.ones(n, n) * 0.5  # Much higher than threshold

        should_reanchor, reason, diag = controller.should_reanchor(
            rho, overlap, n_electrons, uncertainty=uncertainty
        )

        assert should_reanchor
        assert reason == ReanchorReason.HIGH_UNCERTAINTY

    def test_reanchor_history(self, overlap):
        """Test re-anchoring history tracking."""
        from src.inference.reanchor import (
            AdaptiveReAnchorController,
            ReanchorThresholds,
            ReanchorReason
        )

        thresholds = ReanchorThresholds(max_ml_steps=3)
        controller = AdaptiveReAnchorController(thresholds)

        n = 4
        n_electrons = 2
        rho = torch.eye(n, dtype=torch.complex64) * (n_electrons / n)

        # Trigger a few re-anchors
        for _ in range(2):
            for _ in range(3):
                should, reason, diag = controller.should_reanchor(
                    rho, overlap, n_electrons
                )
            controller.record_reanchor(reason, diag)

        history = controller.get_history()
        assert history.n_reanchors == 2
        assert history.total_ml_steps == 6  # 3 steps x 2 reanchors

        summary = history.summary()
        assert summary['n_reanchors'] == 2
        assert 'speedup_factor' in summary

    def test_scheduled_reanchor(self, overlap):
        """Test scheduled re-anchoring."""
        from src.inference.reanchor import (
            ScheduledReanchorController,
            ReanchorReason
        )

        controller = ScheduledReanchorController(interval=5)

        n = 4
        n_electrons = 2
        rho = torch.eye(n, dtype=torch.complex64) * (n_electrons / n)

        # Should not trigger before interval
        for _ in range(4):
            should, reason, _ = controller.should_reanchor(rho, overlap, n_electrons)
            assert not should

        # Should trigger at interval
        should, reason, _ = controller.should_reanchor(rho, overlap, n_electrons)
        assert should
        assert reason == ReanchorReason.MAX_STEPS


class TestIntegration:
    """Integration tests for inference pipeline."""

    def test_imports(self):
        """Test all inference modules can be imported."""
        from src.inference import (
            Predictor,
            StreamingPredictor,
            BundledPredictor,
            EnsembleUncertainty,
            MCDropoutUncertainty,
            AdaptiveReAnchorController,
            HybridSimulator,
        )

    def test_predictor_with_physics_projection(self):
        """Test predictor with physics projection."""
        from src.inference.predictor import Predictor
        from src.physics.projections import PhysicsProjection

        model = MockModel(n_basis=4)
        projection = PhysicsProjection(
            apply_hermitian=True,
            apply_trace=True,
            apply_mcweeney=False
        )

        predictor = Predictor(model, physics_projection=projection)
        assert predictor.physics_projection is not None

    def test_bundled_predictor_creation(self):
        """Test bundled predictor creation."""
        from src.inference.predictor import BundledPredictor

        model = MockModel(n_basis=4)
        predictor = BundledPredictor(model, bundle_size=2)

        assert predictor.bundle_size == 2
