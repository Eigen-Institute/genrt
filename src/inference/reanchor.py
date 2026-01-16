"""
Adaptive re-anchoring controller for RT-TDDFT inference.

Monitors ML predictions and triggers DFT re-anchoring when:
- Uncertainty exceeds threshold
- Physics constraints are violated
- Maximum ML steps reached

This enables arbitrary-length simulations with bounded error.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReanchorReason(Enum):
    """Reasons for triggering re-anchoring."""
    CONTINUE = "continue"
    HIGH_UNCERTAINTY = "high_uncertainty"
    TRACE_VIOLATION = "trace_violation"
    HERMITICITY_VIOLATION = "hermiticity_violation"
    IDEMPOTENCY_VIOLATION = "idempotency_violation"
    PSD_VIOLATION = "psd_violation"
    MAX_STEPS = "max_steps"
    ENERGY_DRIFT = "energy_drift"
    MANUAL = "manual"


@dataclass
class ReanchorThresholds:
    """Thresholds for re-anchoring triggers."""
    uncertainty: float = 0.05
    trace: float = 0.02
    hermiticity: float = 0.01
    idempotency: float = 0.1
    psd: float = -0.01  # Minimum eigenvalue threshold
    energy_drift: float = 0.01  # Relative energy change
    max_ml_steps: int = 500


@dataclass
class ReanchorEvent:
    """Record of a re-anchoring event."""
    step: int
    reason: ReanchorReason
    diagnostics: Dict[str, float]
    density_before: Optional[Tensor] = None
    density_after: Optional[Tensor] = None


@dataclass
class ReanchorHistory:
    """History of re-anchoring events during simulation."""
    events: List[ReanchorEvent] = field(default_factory=list)
    total_ml_steps: int = 0
    total_dft_steps: int = 0

    def add_event(self, event: ReanchorEvent):
        """Add re-anchoring event to history."""
        self.events.append(event)
        self.total_dft_steps += 1

    @property
    def n_reanchors(self) -> int:
        """Number of re-anchoring events."""
        return len(self.events)

    @property
    def speedup_factor(self) -> float:
        """Effective speedup from ML prediction."""
        total_steps = self.total_ml_steps + self.total_dft_steps
        if total_steps == 0:
            return 1.0
        return total_steps / max(self.total_dft_steps, 1)

    def summary(self) -> Dict[str, any]:
        """Get summary statistics."""
        reason_counts = {}
        for event in self.events:
            reason = event.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            'n_reanchors': self.n_reanchors,
            'total_ml_steps': self.total_ml_steps,
            'total_dft_steps': self.total_dft_steps,
            'speedup_factor': self.speedup_factor,
            'reason_counts': reason_counts,
        }


class AdaptiveReAnchorController:
    """
    Controller for adaptive re-anchoring during inference.

    Monitors prediction quality and triggers DFT re-anchoring
    when necessary to maintain accuracy over long rollouts.

    Reference: guide_updated.md Section 8.3 (Month 8)
    """

    def __init__(
        self,
        thresholds: Optional[ReanchorThresholds] = None,
        uncertainty_aggregator: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        Args:
            thresholds: Re-anchoring thresholds
            uncertainty_aggregator: Function to aggregate uncertainty to scalar
        """
        if thresholds is None:
            thresholds = ReanchorThresholds()
        self.thresholds = thresholds

        if uncertainty_aggregator is None:
            uncertainty_aggregator = lambda x: x.abs().mean()
        self.uncertainty_aggregator = uncertainty_aggregator

        self.step_counter = 0
        self.history = ReanchorHistory()
        self._reference_energy = None

    def should_reanchor(
        self,
        rho: Tensor,
        overlap: Tensor,
        n_electrons: int,
        uncertainty: Optional[Tensor] = None,
        ensemble_predictions: Optional[List[Tensor]] = None,
        energy: Optional[float] = None,
    ) -> Tuple[bool, ReanchorReason, Dict[str, float]]:
        """
        Check if re-anchoring should be triggered.

        Args:
            rho: Current density matrix prediction
            overlap: Overlap matrix
            n_electrons: Number of electrons
            uncertainty: Pre-computed uncertainty (optional)
            ensemble_predictions: List of ensemble predictions (optional)
            energy: Current energy (optional, for drift detection)

        Returns:
            Tuple of (should_reanchor, reason, diagnostics)
        """
        self.step_counter += 1
        diagnostics = {}

        # Check uncertainty
        if ensemble_predictions is not None:
            rho_stack = torch.stack(ensemble_predictions)
            rho_mean = rho_stack.mean(dim=0)
            uncertainty = rho_stack.std(dim=0)
        elif uncertainty is None:
            uncertainty = torch.zeros_like(rho.real)
            rho_mean = rho
        else:
            rho_mean = rho

        uncertainty_scalar = self.uncertainty_aggregator(uncertainty).item()
        diagnostics['uncertainty'] = uncertainty_scalar

        if uncertainty_scalar > self.thresholds.uncertainty:
            return True, ReanchorReason.HIGH_UNCERTAINTY, diagnostics

        # Check trace conservation
        trace = torch.einsum('ij,ji->', rho_mean, overlap).real.item()
        trace_error = abs(trace - n_electrons)
        diagnostics['trace'] = trace
        diagnostics['trace_error'] = trace_error

        if trace_error > self.thresholds.trace:
            return True, ReanchorReason.TRACE_VIOLATION, diagnostics

        # Check Hermiticity
        herm_error = (rho_mean - rho_mean.conj().T).abs().max().item()
        diagnostics['hermiticity_error'] = herm_error

        if herm_error > self.thresholds.hermiticity:
            return True, ReanchorReason.HERMITICITY_VIOLATION, diagnostics

        # Check idempotency (soft constraint)
        rho_S_rho = rho_mean @ overlap @ rho_mean
        idem_error = (rho_S_rho - rho_mean).abs().max().item()
        diagnostics['idempotency_error'] = idem_error

        if idem_error > self.thresholds.idempotency:
            return True, ReanchorReason.IDEMPOTENCY_VIOLATION, diagnostics

        # Check energy drift (if reference available)
        if energy is not None:
            diagnostics['energy'] = energy
            if self._reference_energy is not None:
                energy_drift = abs(energy - self._reference_energy) / abs(self._reference_energy + 1e-10)
                diagnostics['energy_drift'] = energy_drift

                if energy_drift > self.thresholds.energy_drift:
                    return True, ReanchorReason.ENERGY_DRIFT, diagnostics

        # Check max steps
        diagnostics['steps'] = self.step_counter

        if self.step_counter >= self.thresholds.max_ml_steps:
            return True, ReanchorReason.MAX_STEPS, diagnostics

        return False, ReanchorReason.CONTINUE, diagnostics

    def record_reanchor(
        self,
        reason: ReanchorReason,
        diagnostics: Dict[str, float],
        density_before: Optional[Tensor] = None,
        density_after: Optional[Tensor] = None,
    ):
        """
        Record a re-anchoring event.

        Args:
            reason: Why re-anchoring was triggered
            diagnostics: Diagnostic values at trigger time
            density_before: ML prediction before re-anchoring
            density_after: DFT result after re-anchoring
        """
        event = ReanchorEvent(
            step=self.history.total_ml_steps + self.step_counter,
            reason=reason,
            diagnostics=diagnostics,
            density_before=density_before,
            density_after=density_after,
        )
        self.history.add_event(event)
        self.history.total_ml_steps += self.step_counter

        logger.info(
            f"Re-anchoring triggered at step {event.step}: {reason.value}"
        )

        # Reset step counter
        self.step_counter = 0

    def set_reference_energy(self, energy: float):
        """Set reference energy for drift detection."""
        self._reference_energy = energy

    def reset(self):
        """Reset controller state."""
        self.step_counter = 0
        self._reference_energy = None

    def get_history(self) -> ReanchorHistory:
        """Get re-anchoring history."""
        return self.history


class HybridSimulator:
    """
    Hybrid ML/DFT simulator with adaptive re-anchoring.

    Combines fast ML predictions with accurate DFT calculations,
    switching between them based on uncertainty and constraint violations.
    """

    def __init__(
        self,
        ml_predictor,
        dft_calculator: Callable,
        reanchor_controller: Optional[AdaptiveReAnchorController] = None,
        ensemble_uncertainty=None,
    ):
        """
        Args:
            ml_predictor: ML model for fast predictions
            dft_calculator: Callable that performs DFT step
            reanchor_controller: Controller for re-anchoring decisions
            ensemble_uncertainty: Optional ensemble for uncertainty estimation
        """
        self.ml_predictor = ml_predictor
        self.dft_calculator = dft_calculator

        if reanchor_controller is None:
            reanchor_controller = AdaptiveReAnchorController()
        self.controller = reanchor_controller

        self.ensemble = ensemble_uncertainty

    def simulate(
        self,
        initial_density: Tensor,
        geometry: Dict[str, Tensor],
        field_sequence: Tensor,
        overlap: Tensor,
        n_electrons: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, any]:
        """
        Run hybrid simulation with adaptive re-anchoring.

        Args:
            initial_density: Starting density matrix
            geometry: Molecular geometry
            field_sequence: External field sequence (n_steps, 3)
            overlap: Overlap matrix
            n_electrons: Number of electrons
            progress_callback: Optional callback(step, total, mode)

        Returns:
            Dict with trajectory, history, and diagnostics
        """
        n_steps = field_sequence.shape[0]
        device = initial_density.device

        trajectory = [initial_density]
        diagnostics_history = []
        mode_history = ['dft']  # First step is from DFT (initial condition)

        rho = initial_density
        self.controller.reset()

        # Initialize ML predictor state
        if hasattr(self.ml_predictor, 'initialize'):
            self.ml_predictor.initialize(initial_density, geometry)

        for t in range(n_steps):
            field = field_sequence[t]

            # Get ML prediction
            if hasattr(self.ml_predictor, 'step'):
                rho_ml = self.ml_predictor.step(
                    field, overlap, n_electrons,
                    apply_projection=True
                )
            else:
                rho_ml = self._ml_step(rho, geometry, field, overlap, n_electrons)

            # Get uncertainty if available
            uncertainty = None
            ensemble_preds = None
            if self.ensemble is not None:
                estimate = self.ensemble(rho, geometry, field)
                uncertainty = estimate.std
                ensemble_preds = estimate.samples

            # Check if re-anchoring needed
            should_reanchor, reason, diag = self.controller.should_reanchor(
                rho_ml, overlap, n_electrons,
                uncertainty=uncertainty,
                ensemble_predictions=ensemble_preds,
            )

            diagnostics_history.append(diag)

            if should_reanchor:
                # Perform DFT calculation
                rho_dft = self.dft_calculator(rho, geometry, field)

                # Record event
                self.controller.record_reanchor(
                    reason, diag,
                    density_before=rho_ml,
                    density_after=rho_dft,
                )

                # Use DFT result and reinitialize ML
                rho = rho_dft
                mode_history.append('dft')

                if hasattr(self.ml_predictor, 'initialize'):
                    self.ml_predictor.initialize(rho_dft, geometry)
            else:
                # Use ML prediction
                rho = rho_ml
                mode_history.append('ml')

            trajectory.append(rho)

            if progress_callback is not None:
                progress_callback(t + 1, n_steps, mode_history[-1])

        return {
            'trajectory': torch.stack(trajectory),
            'diagnostics': diagnostics_history,
            'mode_history': mode_history,
            'reanchor_history': self.controller.get_history(),
        }

    def _ml_step(
        self,
        rho: Tensor,
        geometry: Dict[str, Tensor],
        field: Tensor,
        overlap: Tensor,
        n_electrons: int,
    ) -> Tensor:
        """Single ML prediction step."""
        batch = {
            'density': rho.unsqueeze(0),
            'field': field.unsqueeze(0),
            **{k: v.unsqueeze(0) if v.dim() > 0 else v for k, v in geometry.items()},
        }
        rho_pred = self.ml_predictor(batch)
        if isinstance(rho_pred, tuple):
            rho_pred = rho_pred[0]
        return rho_pred.squeeze(0)


class ScheduledReanchorController:
    """
    Simple scheduled re-anchoring at fixed intervals.

    Useful as a baseline or when uncertainty estimation is not available.
    """

    def __init__(self, interval: int = 100):
        """
        Args:
            interval: Number of ML steps between re-anchoring
        """
        self.interval = interval
        self.step_counter = 0
        self.history = ReanchorHistory()

    def should_reanchor(self, *args, **kwargs) -> Tuple[bool, ReanchorReason, Dict]:
        """Check if scheduled re-anchoring is due."""
        self.step_counter += 1

        if self.step_counter >= self.interval:
            return True, ReanchorReason.MAX_STEPS, {'steps': self.step_counter}

        return False, ReanchorReason.CONTINUE, {'steps': self.step_counter}

    def record_reanchor(self, reason: ReanchorReason, diagnostics: Dict, **kwargs):
        """Record re-anchoring event."""
        event = ReanchorEvent(
            step=self.history.total_ml_steps + self.step_counter,
            reason=reason,
            diagnostics=diagnostics,
        )
        self.history.add_event(event)
        self.history.total_ml_steps += self.step_counter
        self.step_counter = 0

    def reset(self):
        """Reset controller."""
        self.step_counter = 0

    def get_history(self) -> ReanchorHistory:
        """Get history."""
        return self.history


def estimate_optimal_interval(
    validation_trajectory: Tensor,
    ml_predictor,
    overlap: Tensor,
    n_electrons: int,
    geometry: Dict[str, Tensor],
    target_error: float = 0.01,
    max_interval: int = 1000,
) -> int:
    """
    Estimate optimal re-anchoring interval from validation data.

    Args:
        validation_trajectory: Ground truth trajectory
        ml_predictor: ML predictor to evaluate
        overlap: Overlap matrix
        n_electrons: Number of electrons
        geometry: Molecular geometry
        target_error: Target relative error threshold
        max_interval: Maximum interval to test

    Returns:
        Recommended re-anchoring interval
    """
    n_steps = validation_trajectory.shape[0]

    for interval in range(10, min(max_interval, n_steps), 10):
        # Simulate with this interval
        errors = []
        rho = validation_trajectory[0]

        for t in range(interval):
            if t >= n_steps - 1:
                break

            # Predict
            batch = {
                'density': rho.unsqueeze(0),
                'field': torch.zeros(1, 3),  # Assume zero field for estimation
                **{k: v.unsqueeze(0) for k, v in geometry.items()},
            }
            rho_pred = ml_predictor(batch)
            if isinstance(rho_pred, tuple):
                rho_pred = rho_pred[0]
            rho = rho_pred.squeeze(0)

            # Compute error
            rho_true = validation_trajectory[t + 1]
            rel_error = (rho - rho_true).abs().mean() / (rho_true.abs().mean() + 1e-10)
            errors.append(rel_error.item())

        max_error = max(errors) if errors else 0

        if max_error > target_error:
            # Return previous interval (last one that was acceptable)
            return max(10, interval - 10)

    return max_interval
