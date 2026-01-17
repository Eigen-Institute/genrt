#!/usr/bin/env python
"""
Training script for RT-TDDFT ML accelerator.

Usage:
    python scripts/train.py --config configs/training/phase1_h2p.yaml
    python scripts/train.py --data data/processed/h2p --epochs 100
    python scripts/train.py --resume checkpoints/epoch_50.pt

Features:
    - Configurable via YAML or command-line arguments
    - Curriculum learning (molecule ladder + horizon curriculum)
    - Physics-aware losses with variance weighting
    - Mixed precision training
    - Wandb integration (optional)
    - Checkpointing and resumption
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import Trainer, MultiStepTrainer, TrainerConfig
from src.training.losses import PhysicsAwareLoss, LossWeights, ScaledLosses
from src.training.curriculum import (
    CurriculumTrainer,
    HorizonCurriculum,
    TemporalBundling,
    MoleculeLadder,
    LossWeightScheduler,
    create_default_curriculum,
    create_tddftnet_curriculum,
)
from src.physics.projections import PhysicsProjection

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RT-TDDFT ML accelerator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data", type=str, default="data/processed",
        help="Path to processed training data"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of data for validation"
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="RTTDDFTModel",
        help="Model class name"
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to model configuration YAML"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Gradient clipping norm"
    )
    parser.add_argument(
        "--trajectory-length", type=int, default=64,
        help="Training trajectory length"
    )

    # Curriculum
    parser.add_argument(
        "--curriculum", type=str, default="default",
        choices=["none", "default", "tddftnet"],
        help="Curriculum learning strategy"
    )
    parser.add_argument(
        "--horizon-stages", type=int, nargs="+", default=[16, 32, 48, 64],
        help="Horizon curriculum stages"
    )
    parser.add_argument(
        "--bundle-size", type=int, default=2,
        help="Temporal bundling size (0 to disable)"
    )

    # Loss weights
    parser.add_argument(
        "--lambda-recon", type=float, default=1.0,
        help="Reconstruction loss weight"
    )
    parser.add_argument(
        "--lambda-grad", type=float, default=10.0,
        help="Gradient loss weight"
    )
    parser.add_argument(
        "--lambda-herm", type=float, default=1.0,
        help="Hermiticity loss weight"
    )
    parser.add_argument(
        "--lambda-trace", type=float, default=5.0,
        help="Trace loss weight"
    )
    parser.add_argument(
        "--lambda-idem", type=float, default=0.5,
        help="Idempotency loss weight"
    )
    parser.add_argument(
        "--use-scaled-loss", action="store_true",
        help="Use scale-invariant losses (TDDFTNet)"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to train on"
    )
    parser.add_argument(
        "--amp", action="store_true", default=True,
        help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--no-amp", action="store_false", dest="amp",
        help="Disable automatic mixed precision"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loader workers"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--save-every", type=int, default=5,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )

    # Logging
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="Log every N steps"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="rt-tddft-ml",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name"
    )

    # Config file (overrides command-line)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training configuration YAML"
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    return parser.parse_args()


def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: Dict) -> nn.Module:
    """Create model from configuration."""
    from src.models.full_model import RTTDDFTModel, RTTDDFTConfig

    model_config = config.get('model', {})
    geometry_config = model_config.get('geometry', {})
    density_config = model_config.get('density', {})
    dynamics_config = model_config.get('dynamics', {})

    # Build RTTDDFTConfig from YAML structure
    rttddft_config = RTTDDFTConfig(
        # Geometry encoder
        geometry_irreps=geometry_config.get('irreps', "32x0e + 16x1o + 8x2e"),
        geometry_layers=geometry_config.get('num_layers', 4),
        max_radius=geometry_config.get('max_radius', 5.0),
        num_radial_basis=geometry_config.get('num_basis', 8),
        # Density encoder
        latent_dim=density_config.get('latent_dim', 256),
        n_query_tokens=density_config.get('n_query_tokens', 32),
        max_l=density_config.get('max_l', 2),
        # Dynamics (Mamba)
        mamba_d_model=dynamics_config.get('d_model', 256),
        mamba_d_state=dynamics_config.get('d_state', 16),
        mamba_layers=dynamics_config.get('n_layers', 6),
        mamba_dropout=dynamics_config.get('dropout', 0.1),
    )

    return RTTDDFTModel(rttddft_config)


def create_dataloaders(
    data_path: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    trajectory_length: int,
) -> tuple:
    """Create training and validation data loaders."""
    from src.data.dataset import UnifiedTrajectoryDataset, collate_fixed_basis

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # Find all HDF5 files
    h5_files = list(data_path.glob("**/*.h5")) + list(data_path.glob("**/*.hdf5"))

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_path}")

    logger.info(f"Found {len(h5_files)} trajectory files")

    # Create dataset (UnifiedTrajectoryDataset handles multiple trajectory files)
    dataset = UnifiedTrajectoryDataset(
        trajectory_paths=h5_files,
        sequence_length=trajectory_length,
    )

    # Split into train/val
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train samples: {n_train}, Val samples: {n_val}")

    # Create data loaders with custom collate function for fixed basis size
    # (H2 and H2+ both have nbf=4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fixed_basis,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fixed_basis,
    ) if n_val > 0 else None

    return train_loader, val_loader, dataset


def create_dummy_dataloaders(
    batch_size: int,
    n_samples: int = 100,
    n_basis: int = 4,
    n_spin: int = 1,
    trajectory_length: int = 64,
) -> tuple:
    """Create dummy data loaders for testing without real data."""
    from torch.utils.data import TensorDataset

    logger.warning("Using dummy data for testing - no real data found")

    # Create random density matrices
    # Shape: (n_samples, n_spin, n_basis, n_basis)
    density_current = torch.randn(n_samples, n_spin, n_basis, n_basis, dtype=torch.complex64)
    density_current = 0.5 * (density_current + density_current.conj().transpose(-2, -1))  # Hermitian
    density_next = torch.randn(n_samples, n_spin, n_basis, n_basis, dtype=torch.complex64)
    density_next = 0.5 * (density_next + density_next.conj().transpose(-2, -1))  # Hermitian

    field = torch.randn(n_samples, 3)
    overlap = torch.eye(n_basis, dtype=torch.complex64).unsqueeze(0).expand(n_samples, -1, -1)
    n_electrons = torch.full((n_samples,), 2, dtype=torch.float32)

    # Geometry (H2-like: 2 atoms)
    positions = torch.zeros(n_samples, 2, 3)
    positions[:, 1, 2] = 0.74  # H-H bond length
    atomic_numbers = torch.tensor([[1, 1]]).expand(n_samples, -1)

    # Create dataset with dict-like access
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            self.length = data['density_current'].shape[0]

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    data = {
        'density_current': density_current,
        'density_next': density_next,
        'field': field,
        'overlap': overlap,
        'n_electrons': n_electrons,
        'positions': positions,
        'atomic_numbers': atomic_numbers,
    }

    dataset = DictDataset(data)

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset


def setup_wandb(args) -> Optional[object]:
    """Setup Weights & Biases logging."""
    if not args.wandb:
        return None

    try:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        logger.info("Weights & Biases logging enabled")
        return wandb
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
        return None


def wandb_callback(wandb_run):
    """Create callback for W&B logging."""
    def callback(trainer, epoch, train_loss):
        if wandb_run is not None:
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'learning_rate': trainer.scheduler.get_last_lr()[0],
            }
            if trainer.state.val_losses:
                metrics['val_loss'] = trainer.state.val_losses[-1]
            wandb_run.log(metrics)
    return callback


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    # Set seed
    set_seed(args.seed)

    # Setup W&B
    wandb_run = setup_wandb(args)

    # Create data loaders
    try:
        train_loader, val_loader, dataset = create_dataloaders(
            data_path=args.data,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            trajectory_length=args.trajectory_length,
        )
        logger.info(f"Loaded {len(dataset)} samples from {args.data}")
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Could not load data: {e}")
        train_loader, val_loader, dataset = create_dummy_dataloaders(
            batch_size=args.batch_size,
            trajectory_length=args.trajectory_length,
        )

    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create loss function
    loss_weights = LossWeights(
        reconstruction=args.lambda_recon,
        gradient=args.lambda_grad,
        hermitian=args.lambda_herm,
        trace=args.lambda_trace,
        idempotent=args.lambda_idem,
    )

    if args.use_scaled_loss:
        loss_fn = ScaledLosses()
        logger.info("Using scaled losses (TDDFTNet)")
    else:
        loss_fn = PhysicsAwareLoss(weights=loss_weights)
        logger.info(f"Using physics-aware loss with weights: {loss_weights}")

    # Create physics projection
    physics_projection = PhysicsProjection(
        apply_hermitian=True,
        apply_trace=True,
        apply_mcweeney=False,  # Don't use during training (too expensive)
    )

    # Create curriculum
    curriculum = None
    temporal_bundling = None
    horizon_curriculum = None

    if args.curriculum == "default":
        curriculum = create_default_curriculum()
        logger.info("Using default curriculum")
    elif args.curriculum == "tddftnet":
        tddftnet = create_tddftnet_curriculum(
            horizon_stages=args.horizon_stages,
            bundle_size=args.bundle_size,
        )
        horizon_curriculum = tddftnet['horizon_curriculum']
        temporal_bundling = tddftnet['temporal_bundling'] if args.bundle_size > 0 else None
        logger.info(f"Using TDDFTNet curriculum: horizons={args.horizon_stages}, bundle={args.bundle_size}")

    # Create trainer config
    trainer_config = TrainerConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        max_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epochs=args.save_every,
        log_every_n_steps=args.log_every,
        use_amp=args.amp,
        device=args.device,
        trajectory_length=args.trajectory_length,
        lambda_reconstruction=args.lambda_recon,
        lambda_gradient=args.lambda_grad,
        lambda_hermitian=args.lambda_herm,
        lambda_trace=args.lambda_trace,
        lambda_idempotent=args.lambda_idem,
    )

    # Create callbacks
    callbacks = []
    if wandb_run is not None:
        callbacks.append(wandb_callback(wandb_run))

    # Create trainer
    if temporal_bundling is not None:
        trainer = MultiStepTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            loss_fn=loss_fn,
            physics_projection=physics_projection,
            curriculum=curriculum,
            temporal_bundling=temporal_bundling,
            horizon_curriculum=horizon_curriculum,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            loss_fn=loss_fn,
            physics_projection=physics_projection,
            curriculum=curriculum,
            callbacks=callbacks,
        )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")

    # Train
    logger.info("Starting training...")
    state = trainer.train()

    # Final summary
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Final epoch: {state.epoch + 1}")
    logger.info(f"Best validation loss: {state.best_val_loss:.6f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
