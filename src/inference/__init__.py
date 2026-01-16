"""
Inference and re-anchoring for the RT-TDDFT ML accelerator.

This module provides:
- Autoregressive rollout prediction with hidden state management
- Ensemble and MC Dropout uncertainty estimation
- Adaptive re-anchoring controller for hybrid ML/DFT simulations
"""

from .predictor import (
    RolloutConfig,
    RolloutResult,
    Predictor,
    StreamingPredictor,
    BundledPredictor,
)
from .uncertainty import (
    UncertaintyEstimate,
    EnsembleUncertainty,
    MCDropoutUncertainty,
    QuantileUncertainty,
    UncertaintyAggregator,
    create_ensemble,
)
from .reanchor import (
    ReanchorReason,
    ReanchorThresholds,
    ReanchorEvent,
    ReanchorHistory,
    AdaptiveReAnchorController,
    HybridSimulator,
    ScheduledReanchorController,
    estimate_optimal_interval,
)

__all__ = [
    # Predictor
    "RolloutConfig",
    "RolloutResult",
    "Predictor",
    "StreamingPredictor",
    "BundledPredictor",
    # Uncertainty
    "UncertaintyEstimate",
    "EnsembleUncertainty",
    "MCDropoutUncertainty",
    "QuantileUncertainty",
    "UncertaintyAggregator",
    "create_ensemble",
    # Re-anchoring
    "ReanchorReason",
    "ReanchorThresholds",
    "ReanchorEvent",
    "ReanchorHistory",
    "AdaptiveReAnchorController",
    "HybridSimulator",
    "ScheduledReanchorController",
    "estimate_optimal_interval",
]
