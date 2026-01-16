"""
Rollout prediction for RT-TDDFT inference.

Provides autoregressive prediction with:
- Hidden state management for Mamba SSM
- Batched rollout for efficiency
- Optional physics projection at each step
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class RolloutConfig:
    """Configuration for rollout prediction."""
    max_steps: int = 1000
    apply_physics_projection: bool = True
    store_hidden_states: bool = False
    store_latents: bool = False
    checkpoint_interval: int = 100  # Store checkpoints every N steps


@dataclass
class RolloutResult:
    """Result of a rollout prediction."""
    densities: Tensor  # (n_steps, n_basis, n_basis) complex
    times: Tensor  # (n_steps,) float
    dipoles: Optional[Tensor] = None  # (n_steps, 3) if computed
    energies: Optional[Tensor] = None  # (n_steps,) if computed
    hidden_states: Optional[List] = None  # If stored
    latents: Optional[Tensor] = None  # (n_steps, latent_dim) if stored
    checkpoints: Optional[Dict[int, Tensor]] = None  # Step -> density


class Predictor(nn.Module):
    """
    Autoregressive rollout predictor for RT-TDDFT.

    Manages hidden states and provides efficient batched inference.
    """

    def __init__(
        self,
        model: nn.Module,
        physics_projection: Optional[nn.Module] = None,
        observable_calculator: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: The trained RTTDDFTModel
            physics_projection: Optional physics projection module
            observable_calculator: Optional module to compute observables
        """
        super().__init__()
        self.model = model
        self.physics_projection = physics_projection
        self.observable_calculator = observable_calculator

    @torch.no_grad()
    def rollout(
        self,
        initial_density: Tensor,
        geometry: Dict[str, Tensor],
        field_sequence: Tensor,
        overlap: Tensor,
        n_electrons: int,
        config: Optional[RolloutConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> RolloutResult:
        """
        Perform autoregressive rollout prediction.

        Args:
            initial_density: Starting density matrix, shape (n_basis, n_basis) complex
            geometry: Dict with 'positions', 'atomic_numbers', 'edge_index', etc.
            field_sequence: External field at each timestep, shape (n_steps, 3)
            overlap: Overlap matrix, shape (n_basis, n_basis)
            n_electrons: Number of electrons
            config: Rollout configuration
            progress_callback: Optional callback(current_step, total_steps)

        Returns:
            RolloutResult with predicted trajectory
        """
        if config is None:
            config = RolloutConfig()

        n_steps = min(field_sequence.shape[0], config.max_steps)
        device = initial_density.device
        dtype = initial_density.dtype

        # Initialize storage
        densities = [initial_density]
        hidden_states_history = [] if config.store_hidden_states else None
        latents_history = [] if config.store_latents else None
        checkpoints = {} if config.checkpoint_interval > 0 else None

        # Current state
        rho = initial_density
        hidden_states = None

        # Precompute geometry encoding (static for trajectory)
        geometry_encoding = self._encode_geometry(geometry)

        for t in range(n_steps):
            # Get field for this timestep
            field = field_sequence[t]

            # Forward pass
            rho_next, hidden_states, latent = self._step(
                rho, geometry, geometry_encoding, field, hidden_states
            )

            # Apply physics projection if enabled
            if config.apply_physics_projection and self.physics_projection is not None:
                rho_next = self.physics_projection(rho_next, overlap, n_electrons)

            # Store results
            densities.append(rho_next)

            if config.store_hidden_states and hidden_states is not None:
                hidden_states_history.append(
                    [h.clone() for h in hidden_states]
                )

            if config.store_latents and latent is not None:
                latents_history.append(latent.clone())

            if checkpoints is not None and (t + 1) % config.checkpoint_interval == 0:
                checkpoints[t + 1] = rho_next.clone()

            # Update current density
            rho = rho_next

            # Progress callback
            if progress_callback is not None:
                progress_callback(t + 1, n_steps)

        # Stack results
        densities = torch.stack(densities)
        times = torch.arange(n_steps + 1, device=device, dtype=torch.float32)

        # Compute observables if calculator available
        dipoles = None
        energies = None
        if self.observable_calculator is not None:
            dipoles, energies = self._compute_observables(densities)

        # Stack latents if stored
        latents = None
        if latents_history:
            latents = torch.stack(latents_history)

        return RolloutResult(
            densities=densities,
            times=times,
            dipoles=dipoles,
            energies=energies,
            hidden_states=hidden_states_history,
            latents=latents,
            checkpoints=checkpoints,
        )

    def _encode_geometry(self, geometry: Dict[str, Tensor]) -> Tensor:
        """Encode geometry features (computed once per trajectory)."""
        if hasattr(self.model, 'geometry_encoder'):
            return self.model.geometry_encoder(
                geometry['positions'],
                geometry['atomic_numbers'],
                geometry.get('edge_index'),
            )
        return None

    def _step(
        self,
        rho: Tensor,
        geometry: Dict[str, Tensor],
        geometry_encoding: Optional[Tensor],
        field: Tensor,
        hidden_states: Optional[List[Tensor]],
    ) -> Tuple[Tensor, Optional[List[Tensor]], Optional[Tensor]]:
        """Single prediction step."""
        # Build batch-like input for model
        batch = self._make_batch(rho, geometry, field)

        # Forward through model
        if hasattr(self.model, 'forward_with_hidden'):
            rho_next, new_hidden, latent = self.model.forward_with_hidden(
                batch, hidden_states, geometry_encoding
            )
        else:
            # Fallback: model doesn't expose hidden state
            rho_next = self.model(batch)
            new_hidden = None
            latent = None

        return rho_next, new_hidden, latent

    def _make_batch(
        self,
        rho: Tensor,
        geometry: Dict[str, Tensor],
        field: Tensor,
    ) -> Dict[str, Tensor]:
        """Create batch dictionary for model input."""
        batch = {
            'density': rho.unsqueeze(0),  # Add batch dimension
            'field': field.unsqueeze(0),
            **{k: v.unsqueeze(0) if v.dim() > 0 else v for k, v in geometry.items()},
        }
        return batch

    def _compute_observables(
        self,
        densities: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Compute observables for trajectory."""
        if self.observable_calculator is None:
            return None, None

        dipoles = []
        energies = []

        for rho in densities:
            result = self.observable_calculator(rho)
            if 'dipole_moment' in result:
                dipoles.append(result['dipole_moment'])
            if 'energy' in result:
                energies.append(result['energy'])

        dipoles = torch.stack(dipoles) if dipoles else None
        energies = torch.stack(energies) if energies else None

        return dipoles, energies

    @torch.no_grad()
    def rollout_batched(
        self,
        initial_densities: Tensor,
        geometries: List[Dict[str, Tensor]],
        field_sequences: Tensor,
        overlaps: Tensor,
        n_electrons: List[int],
        config: Optional[RolloutConfig] = None,
    ) -> List[RolloutResult]:
        """
        Batched rollout for multiple trajectories.

        Args:
            initial_densities: (batch, n_basis, n_basis) complex
            geometries: List of geometry dicts
            field_sequences: (batch, n_steps, 3)
            overlaps: (batch, n_basis, n_basis)
            n_electrons: List of electron counts
            config: Rollout configuration

        Returns:
            List of RolloutResult, one per batch element
        """
        batch_size = initial_densities.shape[0]
        results = []

        # For now, process sequentially (can be parallelized if needed)
        for i in range(batch_size):
            result = self.rollout(
                initial_densities[i],
                geometries[i],
                field_sequences[i],
                overlaps[i],
                n_electrons[i],
                config,
            )
            results.append(result)

        return results


class StreamingPredictor(Predictor):
    """
    Streaming predictor for real-time inference.

    Maintains internal state between calls for continuous prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        physics_projection: Optional[nn.Module] = None,
        observable_calculator: Optional[nn.Module] = None,
    ):
        super().__init__(model, physics_projection, observable_calculator)
        self._current_density = None
        self._hidden_states = None
        self._geometry_encoding = None
        self._step_count = 0

    def initialize(
        self,
        initial_density: Tensor,
        geometry: Dict[str, Tensor],
    ):
        """Initialize predictor with starting state."""
        self._current_density = initial_density
        self._hidden_states = None
        self._geometry_encoding = self._encode_geometry(geometry)
        self._geometry = geometry
        self._step_count = 0

    def step(
        self,
        field: Tensor,
        overlap: Tensor,
        n_electrons: int,
        apply_projection: bool = True,
    ) -> Tensor:
        """
        Advance one timestep.

        Args:
            field: External field vector, shape (3,)
            overlap: Overlap matrix
            n_electrons: Number of electrons
            apply_projection: Whether to apply physics projection

        Returns:
            Predicted density matrix
        """
        if self._current_density is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        rho_next, self._hidden_states, _ = self._step(
            self._current_density,
            self._geometry,
            self._geometry_encoding,
            field,
            self._hidden_states,
        )

        if apply_projection and self.physics_projection is not None:
            rho_next = self.physics_projection(rho_next, overlap, n_electrons)

        self._current_density = rho_next
        self._step_count += 1

        return rho_next

    def get_state(self) -> Dict[str, any]:
        """Get current predictor state for checkpointing."""
        return {
            'density': self._current_density,
            'hidden_states': self._hidden_states,
            'step_count': self._step_count,
        }

    def set_state(self, state: Dict[str, any]):
        """Restore predictor state from checkpoint."""
        self._current_density = state['density']
        self._hidden_states = state['hidden_states']
        self._step_count = state['step_count']

    def reset(self):
        """Reset predictor state."""
        self._current_density = None
        self._hidden_states = None
        self._geometry_encoding = None
        self._step_count = 0


class BundledPredictor(Predictor):
    """
    Predictor using temporal bundling (from TDDFTNet).

    Predicts multiple timesteps together for improved stability.
    """

    def __init__(
        self,
        model: nn.Module,
        bundle_size: int = 2,
        physics_projection: Optional[nn.Module] = None,
        observable_calculator: Optional[nn.Module] = None,
    ):
        super().__init__(model, physics_projection, observable_calculator)
        self.bundle_size = bundle_size

    @torch.no_grad()
    def rollout(
        self,
        initial_density: Tensor,
        geometry: Dict[str, Tensor],
        field_sequence: Tensor,
        overlap: Tensor,
        n_electrons: int,
        config: Optional[RolloutConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> RolloutResult:
        """
        Rollout using temporal bundling.

        Predicts bundle_size steps at once, using the last prediction
        as input for the next bundle.
        """
        if config is None:
            config = RolloutConfig()

        n_steps = min(field_sequence.shape[0], config.max_steps)
        device = initial_density.device

        densities = [initial_density]
        rho = initial_density
        hidden_states = None

        geometry_encoding = self._encode_geometry(geometry)

        t = 0
        while t < n_steps:
            # Get fields for this bundle
            bundle_end = min(t + self.bundle_size, n_steps)
            bundle_fields = field_sequence[t:bundle_end]

            # Predict bundle
            bundle_preds, hidden_states = self._predict_bundle(
                rho, geometry, geometry_encoding, bundle_fields, hidden_states
            )

            # Apply physics projection to each prediction
            for pred in bundle_preds:
                if config.apply_physics_projection and self.physics_projection is not None:
                    pred = self.physics_projection(pred, overlap, n_electrons)
                densities.append(pred)

            # Use last prediction as input for next bundle
            rho = bundle_preds[-1]
            t = bundle_end

            if progress_callback is not None:
                progress_callback(t, n_steps)

        densities = torch.stack(densities)
        times = torch.arange(len(densities), device=device, dtype=torch.float32)

        dipoles, energies = None, None
        if self.observable_calculator is not None:
            dipoles, energies = self._compute_observables(densities)

        return RolloutResult(
            densities=densities,
            times=times,
            dipoles=dipoles,
            energies=energies,
        )

    def _predict_bundle(
        self,
        rho: Tensor,
        geometry: Dict[str, Tensor],
        geometry_encoding: Optional[Tensor],
        fields: Tensor,
        hidden_states: Optional[List[Tensor]],
    ) -> Tuple[List[Tensor], Optional[List[Tensor]]]:
        """Predict a bundle of timesteps."""
        predictions = []
        current_rho = rho

        for field in fields:
            next_rho, hidden_states, _ = self._step(
                current_rho, geometry, geometry_encoding, field, hidden_states
            )
            predictions.append(next_rho)
            current_rho = next_rho

        return predictions, hidden_states
