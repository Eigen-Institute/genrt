"""
Tests for training module.

Tests cover:
- Trainer configuration
- Training loop basics
- Checkpointing
- Curriculum integration
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, n_basis: int = 4):
        super().__init__()
        self.n_basis = n_basis
        self.linear = nn.Linear(n_basis * n_basis * 2, n_basis * n_basis * 2)

    def forward(self, batch):
        rho = batch['density']
        if rho.dim() == 4:  # (batch, spin, n, n)
            rho = rho[:, 0]  # Take first spin
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

        return rho_out


def create_dummy_dataloader(
    n_samples: int = 10,
    n_basis: int = 4,
    trajectory_length: int = 16,
    batch_size: int = 2,
):
    """Create a dummy data loader for testing."""

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples, n_basis, trajectory_length):
            self.n_samples = n_samples
            self.n_basis = n_basis
            self.traj_len = trajectory_length

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            n = self.n_basis
            traj_len = self.traj_len

            density = torch.randn(traj_len + 1, n, n, dtype=torch.complex64)
            density = 0.5 * (density + density.conj().transpose(-2, -1))

            return {
                'density': density,
                'field': torch.randn(traj_len + 1, 3),
                'overlap': torch.eye(n, dtype=torch.complex64),
                'n_electrons': torch.tensor(2),
                'positions': torch.randn(2, 3),
                'atomic_numbers': torch.tensor([1, 1]),
            }

    dataset = DictDataset(n_samples, n_basis, trajectory_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestTrainerConfig:
    """Tests for trainer configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.training.trainer import TrainerConfig

        config = TrainerConfig()

        assert config.learning_rate == 3e-4
        assert config.max_epochs == 100
        assert config.lambda_gradient == 10.0
        assert config.use_amp is True

    def test_custom_config(self):
        """Test custom configuration."""
        from src.training.trainer import TrainerConfig

        config = TrainerConfig(
            learning_rate=1e-4,
            max_epochs=50,
            lambda_gradient=5.0,
            use_amp=False,
        )

        assert config.learning_rate == 1e-4
        assert config.max_epochs == 50
        assert config.lambda_gradient == 5.0
        assert config.use_amp is False


class TestTrainingState:
    """Tests for training state."""

    def test_initial_state(self):
        """Test initial training state."""
        from src.training.trainer import TrainingState

        state = TrainingState()

        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_val_loss == float('inf')
        assert len(state.train_losses) == 0

    def test_state_updates(self):
        """Test state can be updated."""
        from src.training.trainer import TrainingState

        state = TrainingState()
        state.epoch = 5
        state.global_step = 100
        state.best_val_loss = 0.5
        state.train_losses.append(0.6)

        assert state.epoch == 5
        assert state.global_step == 100
        assert state.best_val_loss == 0.5
        assert len(state.train_losses) == 1


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def model(self):
        return MockModel(n_basis=4)

    @pytest.fixture
    def dataloader(self):
        return create_dummy_dataloader(
            n_samples=10,
            n_basis=4,
            trajectory_length=8,
            batch_size=2,
        )

    @pytest.fixture
    def config(self):
        from src.training.trainer import TrainerConfig
        return TrainerConfig(
            max_epochs=2,
            log_every_n_steps=1,
            save_every_n_epochs=1,
            use_amp=False,
            device="cpu",
            trajectory_length=8,
        )

    def test_trainer_creation(self, model, dataloader, config):
        """Test trainer can be created."""
        from src.training.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            trainer = Trainer(
                model=model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
            )

            assert trainer.model is model
            assert trainer.config == config

    def test_trainer_single_epoch(self, model, dataloader, config):
        """Test trainer can run a single epoch."""
        from src.training.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            config.max_epochs = 1

            trainer = Trainer(
                model=model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
            )

            state = trainer.train()

            assert state.epoch == 0  # 0-indexed after first epoch
            assert state.global_step > 0
            assert len(state.train_losses) == 1

    def test_trainer_with_validation(self, model, dataloader, config):
        """Test trainer with validation."""
        from src.training.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            config.max_epochs = 1

            trainer = Trainer(
                model=model,
                train_loader=dataloader,
                val_loader=dataloader,  # Use same loader for simplicity
                config=config,
            )

            state = trainer.train()

            assert len(state.val_losses) > 0

    def test_checkpoint_save_load(self, model, dataloader, config):
        """Test checkpoint saving and loading."""
        from src.training.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            config.max_epochs = 1

            trainer = Trainer(
                model=model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
            )

            # Train and save
            trainer.train()

            # Check checkpoint exists
            checkpoint_path = Path(tmpdir) / "final.pt"
            assert checkpoint_path.exists()

            # Create new trainer and load
            new_model = MockModel(n_basis=4)
            new_trainer = Trainer(
                model=new_model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
            )

            new_trainer.load_checkpoint(checkpoint_path)

            # States should match
            assert new_trainer.state.epoch == trainer.state.epoch


class TestMultiStepTrainer:
    """Tests for MultiStepTrainer with temporal bundling."""

    @pytest.fixture
    def model(self):
        return MockModel(n_basis=4)

    @pytest.fixture
    def dataloader(self):
        return create_dummy_dataloader(
            n_samples=10,
            n_basis=4,
            trajectory_length=16,
            batch_size=2,
        )

    @pytest.fixture
    def config(self):
        from src.training.trainer import TrainerConfig
        return TrainerConfig(
            max_epochs=1,
            log_every_n_steps=1,
            use_amp=False,
            device="cpu",
            trajectory_length=16,
        )

    def test_multistep_trainer_creation(self, model, dataloader, config):
        """Test MultiStepTrainer can be created."""
        from src.training.trainer import MultiStepTrainer
        from src.training.curriculum import TemporalBundling

        bundling = TemporalBundling(bundle_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            trainer = MultiStepTrainer(
                model=model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
                temporal_bundling=bundling,
            )

            assert trainer.temporal_bundling is bundling

    def test_multistep_trainer_train(self, model, dataloader, config):
        """Test MultiStepTrainer can train."""
        from src.training.trainer import MultiStepTrainer
        from src.training.curriculum import TemporalBundling

        bundling = TemporalBundling(bundle_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            trainer = MultiStepTrainer(
                model=model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
                temporal_bundling=bundling,
            )

            state = trainer.train()

            assert state.global_step > 0
            assert len(state.train_losses) > 0


class TestTrainerIntegration:
    """Integration tests for training components."""

    def test_trainer_with_curriculum(self):
        """Test trainer with curriculum learning."""
        from src.training.trainer import Trainer, TrainerConfig
        from src.training.curriculum import create_default_curriculum

        model = MockModel(n_basis=4)
        dataloader = create_dummy_dataloader(
            n_samples=10,
            n_basis=4,
            trajectory_length=8,
            batch_size=2,
        )

        config = TrainerConfig(
            max_epochs=2,
            use_amp=False,
            device="cpu",
            trajectory_length=8,
        )

        curriculum = create_default_curriculum()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            trainer = Trainer(
                model=model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
                curriculum=curriculum,
            )

            state = trainer.train()

            assert state.global_step > 0

    def test_trainer_with_physics_projection(self):
        """Test trainer with physics projection."""
        from src.training.trainer import Trainer, TrainerConfig
        from src.physics.projections import PhysicsProjection

        model = MockModel(n_basis=4)
        dataloader = create_dummy_dataloader(
            n_samples=10,
            n_basis=4,
            trajectory_length=8,
            batch_size=2,
        )

        config = TrainerConfig(
            max_epochs=1,
            use_amp=False,
            device="cpu",
            trajectory_length=8,
        )

        projection = PhysicsProjection(
            apply_hermitian=True,
            apply_trace=True,
            apply_mcweeney=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            trainer = Trainer(
                model=model,
                train_loader=dataloader,
                val_loader=None,
                config=config,
                physics_projection=projection,
            )

            state = trainer.train()

            assert state.global_step > 0

    def test_imports(self):
        """Test all training modules can be imported."""
        from src.training import (
            Trainer,
            MultiStepTrainer,
            TrainerConfig,
            TrainingState,
            PhysicsAwareLoss,
            HorizonCurriculum,
            TemporalBundling,
        )
