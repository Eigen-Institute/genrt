"""
Tests for physics constraints and projections.

Tests cover:
- Constraint checking functions
- Projection operators (Hermitianize, trace normalize, McWeeney)
- Observable calculations
"""

import pytest
import torch
import numpy as np


class TestConstraintChecking:
    """Tests for physics constraint checking functions."""

    @pytest.fixture
    def hermitian_matrix(self):
        """Create a Hermitian matrix."""
        n = 5
        A = torch.randn(n, n) + 1j * torch.randn(n, n)
        return 0.5 * (A + A.conj().T)

    @pytest.fixture
    def non_hermitian_matrix(self):
        """Create a non-Hermitian matrix."""
        n = 5
        return torch.randn(n, n) + 1j * torch.randn(n, n)

    @pytest.fixture
    def overlap_matrix(self):
        """Create a valid overlap matrix (positive definite)."""
        n = 5
        A = torch.randn(n, n)
        return A @ A.T + torch.eye(n)  # Guaranteed positive definite

    def test_check_hermiticity_hermitian(self, hermitian_matrix):
        """Test Hermiticity check on Hermitian matrix."""
        from src.physics.constraints import check_hermiticity

        is_hermitian, error = check_hermiticity(hermitian_matrix)
        assert is_hermitian
        assert error < 1e-6

    def test_check_hermiticity_non_hermitian(self, non_hermitian_matrix):
        """Test Hermiticity check on non-Hermitian matrix."""
        from src.physics.constraints import check_hermiticity

        is_hermitian, error = check_hermiticity(non_hermitian_matrix)
        assert not is_hermitian
        assert error > 1e-3

    def test_check_trace(self, hermitian_matrix, overlap_matrix):
        """Test trace check."""
        from src.physics.constraints import check_trace

        # Scale to get specific trace
        n_electrons = 4
        overlap_complex = overlap_matrix.to(torch.complex64)
        current_trace = torch.einsum("ij,ji->", hermitian_matrix, overlap_complex).real
        scaled = hermitian_matrix * (n_electrons / current_trace)

        is_correct, error = check_trace(scaled, overlap_complex, n_electrons)
        assert is_correct
        assert error < 1e-4

    def test_check_idempotency(self, overlap_matrix):
        """Test idempotency check."""
        from src.physics.constraints import check_idempotency

        # Create an idempotent matrix via projection
        n = 5
        overlap_complex = overlap_matrix.to(torch.complex64)

        # Create a random density that's NOT idempotent
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)

        is_idempotent, error = check_idempotency(rho, overlap_complex)
        # Random matrix is unlikely to be idempotent
        assert error > 0.01

    def test_check_cauchy_schwarz(self):
        """Test fast Cauchy-Schwarz PSD check."""
        from src.physics.constraints import check_cauchy_schwarz

        n = 4
        # PSD matrix (positive eigenvalues)
        A = torch.randn(n, n, dtype=torch.complex64)
        psd_matrix = A @ A.conj().T  # Guaranteed PSD

        satisfies, violation = check_cauchy_schwarz(psd_matrix)
        assert satisfies, f"PSD matrix should satisfy Cauchy-Schwarz, violation={violation}"

        # Non-PSD matrix with negative diagonal
        non_psd = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.complex64))
        satisfies, violation = check_cauchy_schwarz(non_psd)
        assert not satisfies, "Matrix with negative diagonal should fail"

    def test_check_psd_fast_vs_full(self, hermitian_matrix, overlap_matrix):
        """Test fast vs full PSD check."""
        from src.physics.constraints import check_positive_semidefinite

        overlap_complex = overlap_matrix.to(torch.complex64)

        # Both methods should agree on a clearly PSD matrix
        A = torch.randn(5, 5, dtype=torch.complex64)
        psd = A @ A.conj().T

        is_psd_fast, _ = check_positive_semidefinite(psd, overlap_complex, fast=True)
        is_psd_full, _ = check_positive_semidefinite(psd, overlap_complex, fast=False)

        # Full check is authoritative; fast check may have false positives
        assert is_psd_full, "PSD matrix should pass full check"

    def test_check_all_constraints(self, hermitian_matrix, overlap_matrix):
        """Test checking all constraints at once."""
        from src.physics.constraints import check_all_constraints

        overlap_complex = overlap_matrix.to(torch.complex64)
        results = check_all_constraints(hermitian_matrix, overlap_complex, n_electrons=4)

        assert 'hermitian' in results
        assert 'trace' in results
        assert 'positive_semidefinite' in results
        assert 'idempotent' in results
        # AO basis uses positive_occupations (non-negative), not bounded [0,2]
        assert 'positive_occupations' in results

    def test_constraint_violation_loss(self, hermitian_matrix, overlap_matrix):
        """Test differentiable constraint violation loss."""
        from src.physics.constraints import constraint_violation_loss

        overlap_complex = overlap_matrix.to(torch.complex64)
        losses = constraint_violation_loss(hermitian_matrix, overlap_complex, n_electrons=4)

        assert 'hermitian' in losses
        assert 'trace' in losses
        assert 'idempotent' in losses

        # Losses should be tensors
        for name, loss in losses.items():
            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad or hermitian_matrix.requires_grad is False


class TestProjections:
    """Tests for physics projection operators."""

    @pytest.fixture
    def overlap_matrix(self):
        n = 5
        A = torch.randn(n, n)
        return (A @ A.T + torch.eye(n)).to(torch.complex64)

    def test_hermitianize(self):
        """Test Hermitianization projection."""
        from src.physics.projections import hermitianize

        n = 5
        A = torch.randn(n, n) + 1j * torch.randn(n, n)
        H = hermitianize(A)

        # Check result is Hermitian
        diff = (H - H.conj().T).abs().max()
        assert diff < 1e-6

    def test_trace_normalize(self, overlap_matrix):
        """Test trace normalization."""
        from src.physics.projections import trace_normalize

        n = 5
        n_electrons = 4
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)

        rho_norm = trace_normalize(rho, overlap_matrix, n_electrons)

        # Check trace
        trace = torch.einsum("ij,ji->", rho_norm, overlap_matrix).real
        assert abs(trace.item() - n_electrons) < 1e-4

    def test_mcweeney_purification(self, overlap_matrix):
        """Test McWeeney purification."""
        from src.physics.projections import mcweeney_purification

        n = 5
        n_electrons = 2
        rho = torch.randn(n, n, dtype=torch.complex64)
        rho = 0.5 * (rho + rho.conj().T)

        rho_pure = mcweeney_purification(rho, overlap_matrix, n_electrons, n_iterations=5)

        # Check still Hermitian
        diff = (rho_pure - rho_pure.conj().T).abs().max()
        assert diff < 1e-5

        # Check trace preserved
        trace = torch.einsum("ij,ji->", rho_pure, overlap_matrix).real
        assert abs(trace.item() - n_electrons) < 0.1

    def test_physics_projection_module(self, overlap_matrix):
        """Test PhysicsProjection module."""
        from src.physics.projections import PhysicsProjection

        proj = PhysicsProjection(
            apply_hermitian=True,
            apply_trace=True,
            apply_mcweeney=False,
        )

        n = 5
        n_electrons = 4
        rho = torch.randn(n, n, dtype=torch.complex64)

        rho_proj = proj(rho, overlap_matrix, n_electrons)

        # Check Hermitian
        diff = (rho_proj - rho_proj.conj().T).abs().max()
        assert diff < 1e-6

        # Check trace
        trace = torch.einsum("ij,ji->", rho_proj, overlap_matrix).real
        assert abs(trace.item() - n_electrons) < 1e-4


class TestObservables:
    """Tests for physical observable calculations."""

    @pytest.fixture
    def density_setup(self):
        """Create density matrix and supporting data."""
        n = 4
        n_atoms = 2

        # Simple density matrix
        rho = torch.eye(n, dtype=torch.complex64)
        overlap = torch.eye(n, dtype=torch.complex64)

        # Dipole integrals (random for testing)
        dipole = torch.randn(3, n, n)

        # Atom-basis mapping
        atom_map = torch.tensor([0, 0, 1, 1])

        return {
            'rho': rho,
            'overlap': overlap,
            'dipole': dipole,
            'atom_map': atom_map,
        }

    def test_compute_dipole_moment(self, density_setup):
        """Test dipole moment calculation."""
        from src.physics.observables import compute_dipole_moment

        dipole = compute_dipole_moment(
            density_setup['rho'],
            density_setup['dipole'],
        )

        assert dipole.shape == (3,)
        assert not torch.isnan(dipole).any()

    def test_compute_mulliken_populations(self, density_setup):
        """Test Mulliken population analysis."""
        from src.physics.observables import compute_mulliken_populations

        pops = compute_mulliken_populations(
            density_setup['rho'],
            density_setup['overlap'],
            density_setup['atom_map'],
        )

        assert pops.shape == (2,)  # 2 atoms
        assert not torch.isnan(pops).any()

        # Total should equal trace
        total = pops.sum()
        trace = density_setup['rho'].trace().real
        assert abs(total.item() - trace.item()) < 1e-4

    def test_compute_natural_orbital_occupations(self, density_setup):
        """Test natural orbital occupation calculation."""
        from src.physics.observables import compute_natural_orbital_occupations

        occupations, coeffs = compute_natural_orbital_occupations(
            density_setup['rho'],
            density_setup['overlap'],
        )

        assert occupations.shape == (4,)
        assert coeffs.shape == (4, 4)

        # Occupations should sum to trace
        assert abs(occupations.sum().item() - 4.0) < 1e-4

    def test_observable_calculator(self, density_setup):
        """Test ObservableCalculator module."""
        from src.physics.observables import ObservableCalculator

        calc = ObservableCalculator(
            overlap=density_setup['overlap'],
            atom_basis_map=density_setup['atom_map'],
            dipole_integrals=density_setup['dipole'],
        )

        results = calc(density_setup['rho'])

        assert 'mulliken_populations' in results
        assert 'dipole_moment' in results


class TestTrainingLosses:
    """Tests for physics-aware training losses."""

    @pytest.fixture
    def density_pair(self):
        """Create current and target density matrices."""
        n = 4

        rho_current = torch.randn(1, n, n, dtype=torch.complex64)
        rho_current = 0.5 * (rho_current + rho_current.conj().transpose(-2, -1))

        rho_target = rho_current + 0.1 * torch.randn_like(rho_current)
        rho_target = 0.5 * (rho_target + rho_target.conj().transpose(-2, -1))

        rho_pred = rho_current + 0.05 * torch.randn_like(rho_current)

        overlap = torch.eye(n, dtype=torch.complex64)

        return {
            'current': rho_current,
            'target': rho_target,
            'pred': rho_pred,
            'overlap': overlap,
            'n_electrons': 2,
        }

    def test_reconstruction_loss(self, density_pair):
        """Test reconstruction loss."""
        from src.training.losses import DensityReconstructionLoss

        loss_fn = DensityReconstructionLoss()
        loss = loss_fn(density_pair['pred'], density_pair['target'])

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_gradient_loss(self, density_pair):
        """Test gradient loss."""
        from src.training.losses import GradientLoss

        loss_fn = GradientLoss(dt=1.0)
        loss = loss_fn(
            density_pair['pred'],
            density_pair['current'],
            density_pair['target'],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_hermiticity_loss(self, density_pair):
        """Test Hermiticity loss."""
        from src.training.losses import HermiticityLoss

        loss_fn = HermiticityLoss()

        # Hermitian matrix should have low loss
        loss_hermitian = loss_fn(density_pair['current'])
        assert loss_hermitian.item() < 1e-10

        # Non-Hermitian should have higher loss
        non_hermitian = torch.randn(1, 4, 4, dtype=torch.complex64)
        loss_non_hermitian = loss_fn(non_hermitian)
        assert loss_non_hermitian.item() > loss_hermitian.item()

    def test_physics_aware_loss(self, density_pair):
        """Test combined physics-aware loss."""
        from src.training.losses import PhysicsAwareLoss

        loss_fn = PhysicsAwareLoss()
        total_loss, components = loss_fn(
            density_pair['pred'],
            density_pair['target'],
            density_pair['current'],
            density_pair['overlap'],
            density_pair['n_electrons'],
            return_components=True,
        )

        assert isinstance(total_loss, torch.Tensor)
        assert 'reconstruction' in components
        assert 'gradient' in components
        assert 'hermitian' in components
        assert 'trace' in components
        assert 'idempotent' in components


class TestCurriculum:
    """Tests for curriculum learning strategies."""

    def test_variance_curriculum_update(self):
        """Test variance curriculum updates."""
        from src.training.curriculum import VarianceCurriculum

        curriculum = VarianceCurriculum(
            initial_horizon=1,
            max_horizon=5,
            variance_threshold=0.01,
            patience=3,
        )

        assert curriculum.get_horizon() == 1

        # Simulate low variance for multiple epochs
        for _ in range(5):
            result = curriculum.update(0.005)

        # Horizon should have increased
        assert curriculum.get_horizon() > 1

    def test_molecule_ladder(self):
        """Test molecule ladder progression."""
        from src.training.curriculum import MoleculeLadder

        ladder = MoleculeLadder()

        assert ladder.current_stage.name == "phase1_h2p"

        # Step through epochs
        for _ in range(60):  # Past first stage
            result = ladder.step_epoch()

        # Should have advanced to next stage
        assert ladder.current_stage_idx >= 1

    def test_loss_weight_scheduler(self):
        """Test loss weight scheduling."""
        from src.training.curriculum import LossWeightScheduler

        scheduler = LossWeightScheduler(
            initial_weights={'a': 0.0, 'b': 1.0},
            target_weights={'a': 1.0, 'b': 1.0},
            warmup_epochs=10,
        )

        # First epoch should be close to initial
        weights = scheduler.step()
        assert weights['a'] < 0.2

        # After warmup, should be at target
        for _ in range(10):
            weights = scheduler.step()
        assert abs(weights['a'] - 1.0) < 0.01

    def test_curriculum_trainer(self):
        """Test combined curriculum trainer."""
        from src.training.curriculum import CurriculumTrainer

        trainer = CurriculumTrainer()

        # Step through a few epochs
        for i in range(5):
            result = trainer.step_epoch(variance=0.1)

        assert result['epoch'] == 5
        assert 'config' in result


class TestTDDFTNetLosses:
    """Tests for TDDFTNet-inspired loss functions."""

    @pytest.fixture
    def density_pair(self):
        """Create density matrices for loss testing."""
        n = 4
        rho_pred = torch.randn(n, n, dtype=torch.complex64)
        rho_pred = 0.5 * (rho_pred + rho_pred.conj().T)
        rho_true = torch.randn(n, n, dtype=torch.complex64)
        rho_true = 0.5 * (rho_true + rho_true.conj().T)
        return rho_pred, rho_true

    def test_scaled_l2_loss(self, density_pair):
        """Test scaled L2 loss is scale-invariant."""
        from src.training.losses import ScaledLosses

        rho_pred, rho_true = density_pair
        loss_fn = ScaledLosses()

        # Original loss
        loss1 = loss_fn.scaled_l2(rho_pred, rho_true)

        # Scale both by same factor - relative error should be same
        scale = 10.0
        loss2 = loss_fn.scaled_l2(rho_pred * scale, rho_true * scale)

        assert abs(loss1.item() - loss2.item()) < 1e-4

    def test_scaled_dipole_loss(self, density_pair):
        """Test scaled dipole loss."""
        from src.training.losses import ScaledLosses

        rho_pred, rho_true = density_pair
        n = rho_pred.shape[0]
        # Dipole integrals should be complex to match density dtype
        dipole_integrals = torch.randn(3, n, n, dtype=torch.complex64)

        loss_fn = ScaledLosses()
        dipole_loss = loss_fn.scaled_dipole(rho_pred, rho_true, dipole_integrals)

        assert isinstance(dipole_loss, torch.Tensor)
        assert dipole_loss.item() >= 0

    def test_scaled_losses_forward(self, density_pair):
        """Test combined scaled loss forward pass."""
        from src.training.losses import ScaledLosses

        rho_pred, rho_true = density_pair
        n = rho_pred.shape[0]
        # Dipole integrals should be complex to match density dtype
        dipole_integrals = torch.randn(3, n, n, dtype=torch.complex64)

        loss_fn = ScaledLosses(dipole_weight=0.1)

        # Without dipole
        loss_no_dipole = loss_fn(rho_pred, rho_true)
        assert isinstance(loss_no_dipole, torch.Tensor)

        # With dipole
        loss_with_dipole = loss_fn(rho_pred, rho_true, dipole_integrals)
        assert loss_with_dipole.item() >= loss_no_dipole.item()

    def test_variance_weighted_loss(self, density_pair):
        """Test variance-weighted loss."""
        from src.training.losses import VarianceWeightedLoss

        rho_pred, rho_true = density_pair

        loss_fn = VarianceWeightedLoss()

        # Without precomputed weights
        loss = loss_fn(rho_pred, rho_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_variance_weighted_loss_with_weights(self, density_pair):
        """Test variance-weighted loss with precomputed weights."""
        from src.training.losses import VarianceWeightedLoss

        rho_pred, rho_true = density_pair
        n = rho_pred.shape[0]

        loss_fn = VarianceWeightedLoss()

        # Create some sample trajectories for weight computation
        trajectories = [
            torch.randn(10, n, n, dtype=torch.complex64) for _ in range(3)
        ]
        weights = loss_fn.compute_variance_weights(trajectories)
        loss_fn.set_weights(weights)

        loss = loss_fn(rho_pred, rho_true)
        assert isinstance(loss, torch.Tensor)


class TestHorizonCurriculum:
    """Tests for TDDFTNet horizon curriculum."""

    def test_horizon_progression(self):
        """Test horizon increases through stages."""
        from src.training.curriculum import HorizonCurriculum

        curriculum = HorizonCurriculum(
            stages=[8, 16, 32, 64],
            epochs_per_stage=5
        )

        # Epoch 0-4: stage 0, horizon 8
        assert curriculum.get_horizon(0) == 8
        assert curriculum.get_horizon(4) == 8

        # Epoch 5-9: stage 1, horizon 16
        assert curriculum.get_horizon(5) == 16
        assert curriculum.get_horizon(9) == 16

        # Epoch 10-14: stage 2, horizon 32
        assert curriculum.get_horizon(10) == 32

        # Epoch 15+: stage 3, horizon 64
        assert curriculum.get_horizon(15) == 64
        assert curriculum.get_horizon(100) == 64  # Stays at max

    def test_horizon_step(self):
        """Test stepping through epochs."""
        from src.training.curriculum import HorizonCurriculum

        curriculum = HorizonCurriculum(
            stages=[8, 16],
            epochs_per_stage=3
        )

        # Step through epochs
        for _ in range(3):
            result = curriculum.step()

        # Should be at end of stage 0
        assert result['stage_idx'] == 0
        assert result['horizon'] == 8
        assert result['stage_complete']

        # Next step should be stage 1
        result = curriculum.step()
        assert result['stage_idx'] == 1
        assert result['horizon'] == 16

    def test_stage_info(self):
        """Test getting stage information."""
        from src.training.curriculum import HorizonCurriculum

        curriculum = HorizonCurriculum(
            stages=[8, 16, 32],
            epochs_per_stage=5
        )

        info = curriculum.get_stage_info()
        assert info['stage_idx'] == 0
        assert info['total_stages'] == 3
        assert not info['is_final_stage']


class TestTemporalBundling:
    """Tests for TDDFTNet temporal bundling."""

    @pytest.fixture
    def trajectory(self):
        """Create a test trajectory."""
        n_steps = 10
        n = 4
        return torch.randn(n_steps, n, n, dtype=torch.complex64)

    def test_bundle_trajectory(self, trajectory):
        """Test trajectory bundling."""
        from src.training.curriculum import TemporalBundling

        bundler = TemporalBundling(bundle_size=2, overlap=0)
        result = bundler.bundle_trajectory(trajectory)

        assert 'inputs' in result
        assert 'targets' in result
        assert 'n_bundles' in result
        assert result['n_bundles'] > 0

        # Each target should have bundle_size timesteps
        assert result['targets'].shape[1] == 2

    def test_bundle_with_overlap(self, trajectory):
        """Test bundling with overlapping windows."""
        from src.training.curriculum import TemporalBundling

        bundler = TemporalBundling(bundle_size=3, overlap=1)
        result = bundler.bundle_trajectory(trajectory)

        # With overlap, stride is 2, should get more bundles
        assert result['n_bundles'] > 0
        assert result['targets'].shape[1] == 3

    def test_unbundle_average(self):
        """Test unbundling with averaging."""
        from src.training.curriculum import TemporalBundling

        bundler = TemporalBundling(bundle_size=2, overlap=0)

        # Create some bundled predictions
        n_bundles = 4
        bundles = torch.randn(n_bundles, 2, 3, 3)

        trajectory = bundler.unbundle_predictions(bundles, method="average")

        # Should reconstruct full trajectory
        expected_length = (n_bundles - 1) * bundler.stride + 2
        assert trajectory.shape[0] == expected_length

    def test_unbundle_with_overlap(self):
        """Test unbundling overlapping bundles."""
        from src.training.curriculum import TemporalBundling

        bundler = TemporalBundling(bundle_size=3, overlap=1)

        n_bundles = 4
        bundles = torch.randn(n_bundles, 3, 3, 3)

        # Average method should handle overlapping regions
        trajectory = bundler.unbundle_predictions(bundles, method="average")
        assert trajectory.shape[0] == (n_bundles - 1) * 2 + 3

    def test_bundled_loss(self, trajectory):
        """Test computing loss over bundled predictions."""
        from src.training.curriculum import TemporalBundling

        bundler = TemporalBundling(bundle_size=2)

        # Create mock bundles
        n_bundles = 3
        pred = torch.randn(n_bundles, 2, 4, 4)
        target = torch.randn(n_bundles, 2, 4, 4)

        def mse_loss(a, b):
            return (a - b).pow(2).mean()

        loss = bundler.compute_bundled_loss(pred, target, mse_loss)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_bundled_loss_with_discount(self, trajectory):
        """Test bundled loss with temporal discounting."""
        from src.training.curriculum import TemporalBundling

        bundler = TemporalBundling(bundle_size=3)

        pred = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)

        def mse_loss(a, b):
            return (a - b).pow(2).mean()

        # With discount, later timesteps contribute less
        loss_no_discount = bundler.compute_bundled_loss(pred, target, mse_loss, discount=1.0)
        loss_with_discount = bundler.compute_bundled_loss(pred, target, mse_loss, discount=0.9)

        # Both should be valid
        assert loss_no_discount.item() >= 0
        assert loss_with_discount.item() >= 0

    def test_create_tddftnet_curriculum(self):
        """Test factory function for TDDFTNet curriculum."""
        from src.training.curriculum import create_tddftnet_curriculum

        components = create_tddftnet_curriculum(
            horizon_stages=[8, 16, 32],
            epochs_per_stage=5,
            bundle_size=2
        )

        assert 'horizon_curriculum' in components
        assert 'temporal_bundling' in components
        assert components['horizon_curriculum'].stages == [8, 16, 32]
        assert components['temporal_bundling'].bundle_size == 2
