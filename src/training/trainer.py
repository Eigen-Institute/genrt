"""
Training loop for RT-TDDFT ML accelerator.

Implements:
- Multi-step trajectory training
- Physics-aware loss computation
- Curriculum learning integration
- Gradient accumulation and mixed precision
- Checkpointing and logging
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Training duration
    max_epochs: int = 100
    max_steps: Optional[int] = None
    warmup_steps: int = 1000

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    save_every_n_steps: Optional[int] = None
    keep_last_n_checkpoints: int = 3

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1

    # Hardware
    use_amp: bool = True
    device: str = "cuda"

    # Trajectory training
    trajectory_length: int = 64
    teacher_forcing_ratio: float = 0.5  # Probability of using ground truth

    # Loss weights (from guide.md)
    lambda_reconstruction: float = 1.0
    lambda_gradient: float = 10.0
    lambda_hermitian: float = 1.0
    lambda_trace: float = 5.0
    lambda_idempotent: float = 0.5


@dataclass
class TrainingState:
    """Current state of training."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


class Trainer:
    """
    Trainer for RT-TDDFT model.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainerConfig,
        loss_fn: Optional[nn.Module] = None,
        physics_projection: Optional[nn.Module] = None,
        curriculum: Optional[object] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            loss_fn: Loss function (uses PhysicsAwareLoss if None)
            physics_projection: Physics projection module
            curriculum: Curriculum learning controller
            optimizer: Optimizer (creates AdamW if None)
            scheduler: LR scheduler
            callbacks: List of callback functions
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.physics_projection = physics_projection
        self.curriculum = curriculum
        self.callbacks = callbacks or []

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup loss function
        if loss_fn is None:
            from .losses import PhysicsAwareLoss, LossWeights
            loss_fn = PhysicsAwareLoss(
                weights=LossWeights(
                    reconstruction=config.lambda_reconstruction,
                    gradient=config.lambda_gradient,
                    hermitian=config.lambda_hermitian,
                    trace=config.lambda_trace,
                    idempotent=config.lambda_idempotent,
                )
            )
        self.loss_fn = loss_fn.to(self.device)

        # Setup optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer

        # Setup scheduler
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.max_epochs,
                eta_min=config.learning_rate * 0.01,
            )
        self.scheduler = scheduler

        # Setup mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Training state
        self.state = TrainingState()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def train(self) -> TrainingState:
        """
        Run full training loop.

        Returns:
            Final training state
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        try:
            for epoch in range(self.state.epoch, self.config.max_epochs):
                self.state.epoch = epoch

                # Update curriculum if available
                curriculum_config = None
                if self.curriculum is not None:
                    curriculum_config = self.curriculum.step_epoch()
                    logger.info(f"Curriculum config: {curriculum_config}")

                # Train epoch
                train_loss = self._train_epoch(curriculum_config)
                self.state.train_losses.append(train_loss)

                # Validate
                if self.val_loader is not None and (epoch + 1) % self.config.eval_every_n_epochs == 0:
                    val_loss = self._validate()
                    self.state.val_losses.append(val_loss)

                    # Save best model
                    if val_loss < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss
                        self._save_checkpoint("best")
                        logger.info(f"New best validation loss: {val_loss:.6f}")

                # Step scheduler
                self.scheduler.step()

                # Save checkpoint
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(f"epoch_{epoch + 1}")

                # Log epoch summary
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} - "
                    f"Time: {elapsed / 60:.1f}min"
                )

                # Run callbacks
                for callback in self.callbacks:
                    callback(self, epoch, train_loss)

                # Check max steps
                if self.config.max_steps and self.state.global_step >= self.config.max_steps:
                    logger.info(f"Reached max steps: {self.config.max_steps}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint("interrupted")

        # Save final checkpoint
        self._save_checkpoint("final")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")

        return self.state

    def _train_epoch(self, curriculum_config: Optional[Dict] = None) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Get trajectory length from curriculum or config
            traj_len = self.config.trajectory_length
            if curriculum_config and 'prediction_horizon' in curriculum_config:
                traj_len = curriculum_config['prediction_horizon']

            # Compute loss
            loss, metrics = self._training_step(batch, traj_len)

            # Backward pass
            if self.config.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.config.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1
            self.state.global_step += 1

            # Logging
            if self.state.global_step % self.config.log_every_n_steps == 0:
                logger.info(
                    f"Step {self.state.global_step} - Loss: {loss.item():.6f} - "
                    f"Components: {metrics}"
                )

            # Step checkpoint
            if self.config.save_every_n_steps and \
               self.state.global_step % self.config.save_every_n_steps == 0:
                self._save_checkpoint(f"step_{self.state.global_step}")

        return total_loss / max(n_batches, 1)

    def _training_step(
        self,
        batch: Dict[str, Tensor],
        trajectory_length: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Single training step with trajectory unrolling.

        Args:
            batch: Batch of training data
            trajectory_length: Number of steps to unroll

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract batch components
        density_trajectory = batch['density']  # (batch, n_steps, n_basis, n_basis)
        field_sequence = batch['field']  # (batch, n_steps, 3)
        overlap = batch['overlap']  # (batch, n_basis, n_basis)
        n_electrons = batch['n_electrons']  # (batch,) or scalar

        batch_size = density_trajectory.shape[0]
        n_steps = min(density_trajectory.shape[1] - 1, trajectory_length)

        # Get single overlap matrix (same for all batch elements typically)
        # Use first element for loss computation
        overlap_single = overlap[0] if overlap.dim() == 3 else overlap

        # Initialize
        rho_current = density_trajectory[:, 0]
        hidden_states = None
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {'reconstruction': 0, 'gradient': 0, 'hermitian': 0, 'trace': 0, 'idempotent': 0}

        # Get n_electrons value
        n_elec = n_electrons[0].item() if isinstance(n_electrons, Tensor) else n_electrons

        with autocast(enabled=self.config.use_amp):
            for t in range(n_steps):
                # Get targets
                rho_target = density_trajectory[:, t + 1]
                field = field_sequence[:, t]

                # Forward pass
                model_input = {
                    'density': rho_current,
                    'field': field,
                    'overlap': overlap,
                    **{k: v for k, v in batch.items()
                       if k not in ['density', 'field', 'overlap', 'n_electrons']},
                }

                if hasattr(self.model, 'forward_with_hidden'):
                    rho_pred, hidden_states, _ = self.model.forward_with_hidden(
                        model_input, hidden_states
                    )
                else:
                    rho_pred = self.model(model_input)

                # Apply physics projection per sample (if available)
                if self.physics_projection is not None:
                    projected = []
                    for i in range(batch_size):
                        overlap_i = overlap[i] if overlap.dim() == 3 else overlap
                        proj_i = self.physics_projection(rho_pred[i], overlap_i, n_elec)
                        projected.append(proj_i)
                    rho_pred = torch.stack(projected)

                # Compute loss per sample and average
                step_loss = torch.tensor(0.0, device=self.device)
                step_components = {'reconstruction': 0, 'gradient': 0, 'hermitian': 0, 'trace': 0, 'idempotent': 0}

                for i in range(batch_size):
                    overlap_i = overlap[i] if overlap.dim() == 3 else overlap
                    sample_loss, components = self.loss_fn(
                        rho_pred[i], rho_target[i], rho_current[i], overlap_i, n_elec,
                        return_components=True
                    )
                    step_loss = step_loss + sample_loss
                    for k, v in components.items():
                        step_components[k] += v.item() if isinstance(v, Tensor) else v

                step_loss = step_loss / batch_size
                for k in step_components:
                    step_components[k] /= batch_size

                total_loss = total_loss + step_loss

                # Accumulate metrics
                for k, v in step_components.items():
                    metrics[k] += v

                # Teacher forcing: sometimes use ground truth
                if torch.rand(1).item() < self.config.teacher_forcing_ratio:
                    rho_current = rho_target
                else:
                    rho_current = rho_pred.detach()

        # Average over steps
        total_loss = total_loss / n_steps
        for k in metrics:
            metrics[k] /= n_steps

        return total_loss, metrics

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation loop."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss, _ = self._training_step(batch, self.config.trajectory_length)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def _to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state': {
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_val_loss': self.state.best_val_loss,
            },
            'config': self.config.__dict__,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        # Clean old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        # Keep best, final, and last N epoch checkpoints
        to_keep = set(['best.pt', 'final.pt', 'interrupted.pt'])
        for ckpt in checkpoints[-self.config.keep_last_n_checkpoints:]:
            to_keep.add(ckpt.name)

        for ckpt in checkpoints:
            if ckpt.name not in to_keep:
                ckpt.unlink()

    def load_checkpoint(self, path: Union[str, Path]):
        """Load checkpoint and resume training."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        state = checkpoint['state']
        self.state.epoch = state['epoch']
        self.state.global_step = state['global_step']
        self.state.best_val_loss = state['best_val_loss']

        logger.info(f"Loaded checkpoint from {path} (epoch {self.state.epoch})")


class MultiStepTrainer(Trainer):
    """
    Trainer specialized for multi-step trajectory prediction.

    Uses temporal bundling and curriculum learning on prediction horizon.
    """

    def __init__(
        self,
        *args,
        temporal_bundling=None,
        horizon_curriculum=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temporal_bundling = temporal_bundling
        self.horizon_curriculum = horizon_curriculum

    def _training_step(
        self,
        batch: Dict[str, Tensor],
        trajectory_length: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Training step with optional temporal bundling."""
        if self.temporal_bundling is not None:
            return self._bundled_training_step(batch, trajectory_length)
        return super()._training_step(batch, trajectory_length)

    def _bundled_training_step(
        self,
        batch: Dict[str, Tensor],
        trajectory_length: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Training step using temporal bundling."""
        density_trajectory = batch['density']
        field_sequence = batch['field']
        overlap = batch['overlap']
        n_electrons = batch['n_electrons']

        # Bundle the trajectory
        bundle_result = self.temporal_bundling.bundle_trajectory(
            density_trajectory.transpose(0, 1)  # (n_steps, batch, n, n)
        )

        inputs = bundle_result['inputs']
        targets = bundle_result['targets']
        n_bundles = bundle_result['n_bundles']

        if n_bundles == 0:
            # Fallback to regular training
            return super()._training_step(batch, trajectory_length)

        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {'reconstruction': 0, 'gradient': 0, 'hermitian': 0, 'trace': 0, 'idempotent': 0}
        hidden_states = None

        bundle_size = self.temporal_bundling.bundle_size
        n_elec = n_electrons[0].item() if isinstance(n_electrons, Tensor) else n_electrons

        with autocast(enabled=self.config.use_amp):
            for b in range(min(n_bundles, trajectory_length // bundle_size)):
                rho_input = inputs[b].transpose(0, 1)  # (batch, n, n)
                target_bundle = targets[b]  # (bundle_size, batch, n, n)

                # Predict bundle
                predictions = []
                rho_current = rho_input

                for t in range(bundle_size):
                    field = field_sequence[:, b * bundle_size + t]

                    model_input = {
                        'density': rho_current,
                        'field': field,
                        'overlap': overlap,
                        **{k: v for k, v in batch.items()
                           if k not in ['density', 'field', 'overlap', 'n_electrons']},
                    }

                    if hasattr(self.model, 'forward_with_hidden'):
                        rho_pred, hidden_states, _ = self.model.forward_with_hidden(
                            model_input, hidden_states
                        )
                    else:
                        rho_pred = self.model(model_input)

                    predictions.append(rho_pred)
                    rho_current = rho_pred

                # Compute bundled loss
                for t, (pred, target) in enumerate(zip(predictions, target_bundle)):
                    target = target.transpose(0, 1)  # (batch, n, n)
                    prev = predictions[t-1] if t > 0 else rho_input

                    step_loss, components = self.loss_fn(
                        pred, target, prev, overlap, n_elec,
                        return_components=True
                    )
                    total_loss = total_loss + step_loss

                    for k, v in components.items():
                        metrics[k] += v.item() if isinstance(v, Tensor) else v

        n_steps = n_bundles * bundle_size
        total_loss = total_loss / max(n_steps, 1)
        for k in metrics:
            metrics[k] /= max(n_steps, 1)

        return total_loss, metrics
