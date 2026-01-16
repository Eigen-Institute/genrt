"""
Curriculum learning strategies for RT-TDDFT model training.

Implements variance curriculum and molecule ladder progression
as described in the roadmap and guide.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import math


@dataclass
class CurriculumStage:
    """Definition of a curriculum training stage."""
    name: str
    n_epochs: int
    molecules: List[str]
    max_field_strength: float
    prediction_horizon: int
    noise_std: float = 0.0
    learning_rate: float = 1e-4


class VarianceCurriculum:
    """
    Variance-based curriculum that adjusts training based on model performance.

    Monitors prediction variance and adjusts:
    - Prediction horizon (number of steps ahead)
    - Field strength range
    - Data augmentation intensity
    - Loss weights

    From guide.md: Start with single-step predictions, gradually increase
    to multi-step rollouts as accuracy improves.
    """

    def __init__(
        self,
        initial_horizon: int = 1,
        max_horizon: int = 10,
        variance_threshold: float = 0.01,
        patience: int = 5,
        horizon_increment: int = 1,
    ):
        self.current_horizon = initial_horizon
        self.max_horizon = max_horizon
        self.variance_threshold = variance_threshold
        self.patience = patience
        self.horizon_increment = horizon_increment

        self.variance_history: List[float] = []
        self.epochs_at_current = 0

    def update(self, variance: float) -> Dict[str, any]:
        """
        Update curriculum based on current variance.

        Args:
            variance: Current prediction variance

        Returns:
            Dictionary with curriculum updates
        """
        self.variance_history.append(variance)
        self.epochs_at_current += 1

        updates = {
            'horizon_increased': False,
            'current_horizon': self.current_horizon,
        }

        # Check if we should increase horizon
        if self.epochs_at_current >= self.patience:
            recent_variance = sum(self.variance_history[-self.patience:]) / self.patience

            if recent_variance < self.variance_threshold:
                if self.current_horizon < self.max_horizon:
                    self.current_horizon = min(
                        self.current_horizon + self.horizon_increment,
                        self.max_horizon
                    )
                    self.epochs_at_current = 0
                    updates['horizon_increased'] = True
                    updates['current_horizon'] = self.current_horizon

        return updates

    def get_horizon(self) -> int:
        """Get current prediction horizon."""
        return self.current_horizon


class MoleculeLadder:
    """
    Progressive training across molecules of increasing complexity.

    From roadmap:
    - Phase 1: H2+ (1 electron)
    - Phase 2: H2, LiH (2-4 electrons)
    - Phase 3: H2O, NH3, CH4 (8-10 electrons)
    - Phase 4: Larger molecules (benzene, etc.)
    """

    def __init__(
        self,
        stages: Optional[List[CurriculumStage]] = None,
    ):
        if stages is None:
            stages = self._default_stages()
        self.stages = stages
        self.current_stage_idx = 0
        self.epochs_in_stage = 0

    @staticmethod
    def _default_stages() -> List[CurriculumStage]:
        """Create default molecule ladder stages."""
        return [
            CurriculumStage(
                name="phase1_h2p",
                n_epochs=50,
                molecules=["h2+"],
                max_field_strength=0.001,
                prediction_horizon=1,
                learning_rate=1e-4,
            ),
            CurriculumStage(
                name="phase2_small",
                n_epochs=100,
                molecules=["h2", "lih"],
                max_field_strength=0.005,
                prediction_horizon=3,
                learning_rate=5e-5,
            ),
            CurriculumStage(
                name="phase3_medium",
                n_epochs=150,
                molecules=["h2o", "nh3", "ch4"],
                max_field_strength=0.01,
                prediction_horizon=5,
                noise_std=0.01,
                learning_rate=2e-5,
            ),
            CurriculumStage(
                name="phase4_large",
                n_epochs=200,
                molecules=["c6h6"],
                max_field_strength=0.02,
                prediction_horizon=10,
                noise_std=0.02,
                learning_rate=1e-5,
            ),
        ]

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current training stage."""
        return self.stages[self.current_stage_idx]

    def step_epoch(self) -> Dict[str, any]:
        """
        Step through one epoch.

        Returns:
            Dictionary with stage info and whether stage changed
        """
        self.epochs_in_stage += 1
        stage_changed = False

        if self.epochs_in_stage >= self.current_stage.n_epochs:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.epochs_in_stage = 0
                stage_changed = True

        return {
            'stage_name': self.current_stage.name,
            'stage_idx': self.current_stage_idx,
            'epochs_in_stage': self.epochs_in_stage,
            'stage_changed': stage_changed,
            'molecules': self.current_stage.molecules,
            'learning_rate': self.current_stage.learning_rate,
        }

    def get_training_config(self) -> Dict[str, any]:
        """Get current training configuration."""
        stage = self.current_stage
        return {
            'molecules': stage.molecules,
            'max_field_strength': stage.max_field_strength,
            'prediction_horizon': stage.prediction_horizon,
            'noise_std': stage.noise_std,
            'learning_rate': stage.learning_rate,
        }


class LossWeightScheduler:
    """
    Schedule loss component weights during training.

    Gradually increases physics constraint weights as training progresses
    to first learn the dynamics, then enforce constraints.
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        target_weights: Dict[str, float],
        warmup_epochs: int = 10,
        schedule_type: str = "linear",
    ):
        self.initial_weights = initial_weights
        self.target_weights = target_weights
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0

    def step(self) -> Dict[str, float]:
        """
        Step to next epoch and return current weights.

        Returns:
            Dictionary of loss weights
        """
        self.current_epoch += 1

        if self.current_epoch >= self.warmup_epochs:
            return self.target_weights.copy()

        # Interpolate between initial and target
        progress = self.current_epoch / self.warmup_epochs

        if self.schedule_type == "linear":
            factor = progress
        elif self.schedule_type == "cosine":
            factor = 0.5 * (1 - math.cos(math.pi * progress))
        elif self.schedule_type == "exponential":
            factor = 1 - math.exp(-3 * progress)
        else:
            factor = progress

        weights = {}
        for key in self.initial_weights:
            initial = self.initial_weights[key]
            target = self.target_weights[key]
            weights[key] = initial + factor * (target - initial)

        return weights


class AdaptiveDataSampler:
    """
    Adaptively sample training data based on model performance.

    Focuses more training on difficult examples (high error)
    while maintaining coverage of easier examples.
    """

    def __init__(
        self,
        n_samples: int,
        temperature: float = 1.0,
        min_weight: float = 0.1,
    ):
        self.n_samples = n_samples
        self.temperature = temperature
        self.min_weight = min_weight

        # Track errors per sample
        self.errors = torch.ones(n_samples)
        self.counts = torch.ones(n_samples)

    def update_errors(
        self,
        indices: Tensor,
        errors: Tensor,
    ):
        """Update error estimates for samples."""
        for idx, err in zip(indices, errors):
            self.errors[idx] = 0.9 * self.errors[idx] + 0.1 * err.item()
            self.counts[idx] += 1

    def get_sampling_weights(self) -> Tensor:
        """
        Get sampling weights for each sample.

        Returns:
            Weights tensor, shape (n_samples,)
        """
        # Higher error = higher weight
        weights = torch.softmax(self.errors / self.temperature, dim=0)

        # Ensure minimum weight for all samples
        weights = weights * (1 - self.min_weight) + self.min_weight / self.n_samples

        return weights


class CurriculumTrainer:
    """
    Combines curriculum strategies for complete training loop control.

    Manages:
    - Molecule ladder progression
    - Variance-based horizon adjustment
    - Loss weight scheduling
    - Adaptive sampling
    """

    def __init__(
        self,
        molecule_ladder: Optional[MoleculeLadder] = None,
        variance_curriculum: Optional[VarianceCurriculum] = None,
        loss_scheduler: Optional[LossWeightScheduler] = None,
    ):
        self.molecule_ladder = molecule_ladder or MoleculeLadder()
        self.variance_curriculum = variance_curriculum or VarianceCurriculum()
        self.loss_scheduler = loss_scheduler

        self.epoch = 0
        self.total_steps = 0

    def step_epoch(
        self,
        variance: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Step through one epoch of training.

        Args:
            variance: Optional current prediction variance

        Returns:
            Dictionary with all curriculum state
        """
        self.epoch += 1

        # Update molecule ladder
        ladder_update = self.molecule_ladder.step_epoch()

        # Update variance curriculum
        variance_update = {}
        if variance is not None:
            variance_update = self.variance_curriculum.update(variance)

        # Update loss weights
        loss_weights = {}
        if self.loss_scheduler is not None:
            loss_weights = self.loss_scheduler.step()

        return {
            'epoch': self.epoch,
            'ladder': ladder_update,
            'variance': variance_update,
            'loss_weights': loss_weights,
            'config': self.get_config(),
        }

    def get_config(self) -> Dict[str, any]:
        """Get current complete training configuration."""
        ladder_config = self.molecule_ladder.get_training_config()

        return {
            **ladder_config,
            'prediction_horizon': max(
                ladder_config['prediction_horizon'],
                self.variance_curriculum.get_horizon()
            ),
        }


class HorizonCurriculum:
    """
    Curriculum on prediction horizon (from TDDFTNet).

    Progressively increases the number of timesteps the model must predict
    in a single forward pass. This helps the model first learn short-term
    dynamics before tackling long-range predictions.

    Reference: TDDFTNet (ICLR 2025)
    """

    def __init__(
        self,
        stages: List[int] = None,
        epochs_per_stage: int = 10,
    ):
        """
        Args:
            stages: List of horizon values for each stage (e.g., [16, 32, 48, 64])
            epochs_per_stage: Number of epochs before advancing to next stage
        """
        if stages is None:
            stages = [16, 32, 48, 64]
        self.stages = stages
        self.epochs_per_stage = epochs_per_stage
        self.current_epoch = 0

    def get_horizon(self, epoch: Optional[int] = None) -> int:
        """
        Get prediction horizon for given epoch.

        Args:
            epoch: Epoch number (uses internal counter if None)

        Returns:
            Current prediction horizon
        """
        if epoch is None:
            epoch = self.current_epoch
        stage_idx = min(epoch // self.epochs_per_stage, len(self.stages) - 1)
        return self.stages[stage_idx]

    def step(self) -> Dict[str, any]:
        """
        Advance one epoch and return current state.

        Returns:
            Dictionary with horizon info
        """
        self.current_epoch += 1
        stage_idx = min(
            (self.current_epoch - 1) // self.epochs_per_stage,
            len(self.stages) - 1
        )

        return {
            'epoch': self.current_epoch,
            'stage_idx': stage_idx,
            'horizon': self.stages[stage_idx],
            'epochs_in_stage': (self.current_epoch - 1) % self.epochs_per_stage + 1,
            'stage_complete': (self.current_epoch % self.epochs_per_stage == 0),
        }

    def get_stage_info(self) -> Dict[str, any]:
        """Get information about current stage."""
        stage_idx = min(
            self.current_epoch // self.epochs_per_stage,
            len(self.stages) - 1
        )
        return {
            'stage_idx': stage_idx,
            'horizon': self.stages[stage_idx],
            'total_stages': len(self.stages),
            'is_final_stage': stage_idx == len(self.stages) - 1,
        }


class TemporalBundling:
    """
    Predict multiple timesteps together (from TDDFTNet).

    Instead of predicting one timestep at a time, bundle multiple consecutive
    timesteps and predict them together. This can:
    1. Reduce error accumulation during rollout
    2. Improve training efficiency
    3. Help the model learn coherent multi-step dynamics

    Reference: TDDFTNet (ICLR 2025)
    """

    def __init__(
        self,
        bundle_size: int = 2,
        overlap: int = 0,
    ):
        """
        Args:
            bundle_size: Number of timesteps to predict together
            overlap: Number of overlapping timesteps between consecutive bundles
        """
        self.bundle_size = bundle_size
        self.overlap = overlap
        self.stride = bundle_size - overlap

    def bundle_trajectory(
        self,
        trajectory: Tensor,
        return_targets: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Bundle a trajectory into prediction groups.

        Args:
            trajectory: Trajectory tensor, shape (n_steps, ...) or (batch, n_steps, ...)
            return_targets: If True, also return target bundles

        Returns:
            Dictionary with:
            - 'inputs': Input timesteps for each bundle
            - 'targets': Target bundles to predict (if return_targets=True)
            - 'n_bundles': Number of bundles created
        """
        # Handle batched vs unbatched
        if trajectory.dim() == 3:
            # Shape: (n_steps, n, n) - unbatched
            n_steps = trajectory.shape[0]
            is_batched = False
        else:
            # Shape: (batch, n_steps, n, n) - batched
            n_steps = trajectory.shape[1]
            is_batched = True

        # Calculate number of valid bundles
        # Need at least 1 input step + bundle_size target steps
        n_bundles = (n_steps - 1 - self.bundle_size) // self.stride + 1
        n_bundles = max(0, n_bundles)

        if n_bundles == 0:
            # Trajectory too short for bundling
            return {
                'inputs': trajectory[:1] if not is_batched else trajectory[:, :1],
                'targets': trajectory[1:self.bundle_size+1] if not is_batched
                          else trajectory[:, 1:self.bundle_size+1],
                'n_bundles': 0,
            }

        # Create bundles
        if not is_batched:
            input_indices = [i * self.stride for i in range(n_bundles)]
            inputs = torch.stack([trajectory[i] for i in input_indices])

            if return_targets:
                targets = torch.stack([
                    trajectory[i+1:i+1+self.bundle_size]
                    for i in input_indices
                ])
            else:
                targets = None
        else:
            batch_size = trajectory.shape[0]
            input_indices = [i * self.stride for i in range(n_bundles)]
            inputs = torch.stack([trajectory[:, i] for i in input_indices], dim=1)

            if return_targets:
                targets = torch.stack([
                    trajectory[:, i+1:i+1+self.bundle_size]
                    for i in input_indices
                ], dim=1)
            else:
                targets = None

        result = {
            'inputs': inputs,
            'n_bundles': n_bundles,
        }
        if return_targets:
            result['targets'] = targets

        return result

    def unbundle_predictions(
        self,
        bundles: Tensor,
        method: str = "average",
    ) -> Tensor:
        """
        Convert bundled predictions back to a full trajectory.

        Args:
            bundles: Bundled predictions, shape (n_bundles, bundle_size, ...)
            method: How to handle overlapping predictions:
                   "average" - average overlapping timesteps
                   "first" - use first prediction for each timestep
                   "last" - use last prediction for each timestep

        Returns:
            Reconstructed trajectory
        """
        n_bundles = bundles.shape[0]
        bundle_size = bundles.shape[1]
        remaining_shape = bundles.shape[2:]

        # Calculate total trajectory length
        total_steps = (n_bundles - 1) * self.stride + bundle_size

        # Initialize output and count tensors
        output = torch.zeros(total_steps, *remaining_shape, device=bundles.device,
                            dtype=bundles.dtype)
        counts = torch.zeros(total_steps, device=bundles.device)

        # Accumulate predictions
        for i in range(n_bundles):
            start_idx = i * self.stride
            for j in range(bundle_size):
                if method == "average":
                    output[start_idx + j] += bundles[i, j]
                    counts[start_idx + j] += 1
                elif method == "first":
                    if counts[start_idx + j] == 0:
                        output[start_idx + j] = bundles[i, j]
                        counts[start_idx + j] = 1
                elif method == "last":
                    output[start_idx + j] = bundles[i, j]
                    counts[start_idx + j] = 1

        # Normalize by counts (for averaging)
        if method == "average":
            counts = counts.clamp(min=1)
            # Expand counts for broadcasting
            for _ in range(len(remaining_shape)):
                counts = counts.unsqueeze(-1)
            output = output / counts

        return output

    def compute_bundled_loss(
        self,
        pred_bundles: Tensor,
        target_bundles: Tensor,
        loss_fn: Callable,
        discount: float = 1.0,
    ) -> Tensor:
        """
        Compute loss over bundled predictions with optional temporal discounting.

        Args:
            pred_bundles: Predicted bundles, shape (n_bundles, bundle_size, ...)
            target_bundles: Target bundles, shape (n_bundles, bundle_size, ...)
            loss_fn: Base loss function to apply
            discount: Discount factor for later timesteps in bundle (1.0 = no discount)

        Returns:
            Scalar loss
        """
        bundle_size = pred_bundles.shape[1]
        total_loss = torch.tensor(0.0, device=pred_bundles.device)
        weight_sum = 0.0

        for t in range(bundle_size):
            weight = discount ** t
            weight_sum += weight

            step_loss = loss_fn(pred_bundles[:, t], target_bundles[:, t])
            total_loss = total_loss + weight * step_loss

        return total_loss / weight_sum


def create_default_curriculum() -> CurriculumTrainer:
    """Create curriculum trainer with default settings."""
    # Loss weight schedule: start with mostly reconstruction,
    # gradually add physics constraints
    loss_scheduler = LossWeightScheduler(
        initial_weights={
            'reconstruction': 1.0,
            'gradient': 1.0,
            'hermitian': 0.1,
            'trace': 0.5,
            'idempotent': 0.0,
        },
        target_weights={
            'reconstruction': 1.0,
            'gradient': 10.0,
            'hermitian': 1.0,
            'trace': 5.0,
            'idempotent': 0.5,
        },
        warmup_epochs=20,
        schedule_type="cosine",
    )

    return CurriculumTrainer(
        loss_scheduler=loss_scheduler,
    )


def create_tddftnet_curriculum(
    horizon_stages: List[int] = None,
    epochs_per_stage: int = 10,
    bundle_size: int = 2,
) -> Dict[str, any]:
    """
    Create curriculum components following TDDFTNet approach.

    Args:
        horizon_stages: Horizon values for each stage
        epochs_per_stage: Epochs before advancing horizon
        bundle_size: Number of timesteps to bundle together

    Returns:
        Dictionary with curriculum components
    """
    if horizon_stages is None:
        horizon_stages = [16, 32, 48, 64]

    return {
        'horizon_curriculum': HorizonCurriculum(
            stages=horizon_stages,
            epochs_per_stage=epochs_per_stage,
        ),
        'temporal_bundling': TemporalBundling(
            bundle_size=bundle_size,
        ),
    }
