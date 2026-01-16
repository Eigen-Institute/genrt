"""
Training utilities for the RT-TDDFT ML accelerator.

This module provides:
- Physics-aware loss functions
- Scaled losses (TDDFTNet)
- Variance curriculum training
- Horizon curriculum (TDDFTNet)
- Temporal bundling (TDDFTNet)
- Molecule ladder progression
- Learning rate and weight scheduling
- Training loop and checkpointing
"""

from .losses import (
    LossWeights,
    DensityReconstructionLoss,
    GradientLoss,
    HermiticityLoss,
    TraceLoss,
    IdempotencyLoss,
    PhysicsAwareLoss,
    LatentSpaceLoss,
    TrajectoryLoss,
    ContrastiveLoss,
    ScaledLosses,
    VarianceWeightedLoss,
    create_loss_function,
)
from .curriculum import (
    CurriculumStage,
    VarianceCurriculum,
    MoleculeLadder,
    LossWeightScheduler,
    AdaptiveDataSampler,
    CurriculumTrainer,
    HorizonCurriculum,
    TemporalBundling,
    create_default_curriculum,
    create_tddftnet_curriculum,
)
from .trainer import (
    TrainerConfig,
    TrainingState,
    Trainer,
    MultiStepTrainer,
)

__all__ = [
    # Losses
    "LossWeights",
    "DensityReconstructionLoss",
    "GradientLoss",
    "HermiticityLoss",
    "TraceLoss",
    "IdempotencyLoss",
    "PhysicsAwareLoss",
    "LatentSpaceLoss",
    "TrajectoryLoss",
    "ContrastiveLoss",
    "ScaledLosses",
    "VarianceWeightedLoss",
    "create_loss_function",
    # Curriculum
    "CurriculumStage",
    "VarianceCurriculum",
    "MoleculeLadder",
    "LossWeightScheduler",
    "AdaptiveDataSampler",
    "CurriculumTrainer",
    "HorizonCurriculum",
    "TemporalBundling",
    "create_default_curriculum",
    "create_tddftnet_curriculum",
    # Trainer
    "TrainerConfig",
    "TrainingState",
    "Trainer",
    "MultiStepTrainer",
]
