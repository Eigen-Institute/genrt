"""
Physics-aware loss functions for RT-TDDFT model training.

Loss function structure from guide.md:
L = L_recon + 10.0*L_grad + 1.0*L_herm + 5.0*L_trace + 0.5*L_idem

Where:
- L_recon: Reconstruction loss (Frobenius norm)
- L_grad: Time derivative accuracy
- L_herm: Hermiticity violation
- L_trace: Trace conservation violation
- L_idem: Idempotency violation
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossWeights:
    """Weights for different loss components."""
    reconstruction: float = 1.0
    gradient: float = 10.0
    hermitian: float = 1.0
    trace: float = 5.0
    idempotent: float = 0.5
    orthogonal: float = 0.0  # Optional orthogonality loss


class DensityReconstructionLoss(nn.Module):
    """
    Reconstruction loss for density matrix prediction.

    Computes Frobenius norm between predicted and target density matrices.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, rho_pred: Tensor, rho_target: Tensor) -> Tensor:
        """
        Compute reconstruction loss.

        Args:
            rho_pred: Predicted density, shape (..., n, n) complex
            rho_target: Target density, shape (..., n, n) complex

        Returns:
            Scalar loss
        """
        diff = rho_pred - rho_target
        loss = diff.abs().pow(2).sum(dim=(-2, -1))

        if self.normalize:
            norm = rho_target.abs().pow(2).sum(dim=(-2, -1)) + 1e-8
            loss = loss / norm

        return loss.mean()


class GradientLoss(nn.Module):
    """
    Time derivative accuracy loss.

    Ensures the model correctly predicts the rate of change
    of the density matrix, not just the final state.
    """

    def __init__(self, dt: float = 1.0):
        super().__init__()
        self.dt = dt

    def forward(
        self,
        rho_pred: Tensor,
        rho_current: Tensor,
        rho_target: Tensor,
    ) -> Tensor:
        """
        Compute gradient loss.

        Args:
            rho_pred: Predicted next density
            rho_current: Current density
            rho_target: Target next density

        Returns:
            Scalar loss
        """
        # Predicted derivative
        drho_pred = (rho_pred - rho_current) / self.dt

        # Target derivative
        drho_target = (rho_target - rho_current) / self.dt

        # MSE on derivatives
        diff = drho_pred - drho_target
        loss = diff.abs().pow(2).mean()

        return loss


class HermiticityLoss(nn.Module):
    """
    Loss for enforcing Hermiticity: ρ = ρ†.
    """

    def forward(self, rho: Tensor) -> Tensor:
        """
        Compute Hermiticity violation.

        Args:
            rho: Density matrix, shape (..., n, n) complex

        Returns:
            Scalar loss
        """
        rho_dagger = rho.conj().transpose(-2, -1)
        violation = rho - rho_dagger
        return violation.abs().pow(2).mean()


class TraceLoss(nn.Module):
    """
    Loss for enforcing trace conservation: Tr(ρS) = n_electrons.
    """

    def forward(
        self,
        rho: Tensor,
        overlap: Tensor,
        n_electrons: int,
    ) -> Tensor:
        """
        Compute trace violation.

        Args:
            rho: Density matrix, shape (..., n, n) complex
            overlap: Overlap matrix, shape (n, n)
            n_electrons: Target number of electrons

        Returns:
            Scalar loss
        """
        trace = torch.einsum("...ij,ji->...", rho, overlap).real
        violation = (trace - n_electrons).pow(2)
        return violation.mean()


class IdempotencyLoss(nn.Module):
    """
    Loss for enforcing idempotency: ρSρ = ρ.

    For closed-shell systems, the density matrix should be idempotent
    in non-orthogonal basis.
    """

    def forward(
        self,
        rho: Tensor,
        overlap: Tensor,
    ) -> Tensor:
        """
        Compute idempotency violation.

        Args:
            rho: Density matrix, shape (..., n, n) complex
            overlap: Overlap matrix, shape (n, n)

        Returns:
            Scalar loss
        """
        rho_S_rho = rho @ overlap @ rho
        violation = rho_S_rho - rho
        return violation.abs().pow(2).mean()


class PhysicsAwareLoss(nn.Module):
    """
    Combined physics-aware loss function.

    Implements the full loss from guide.md:
    L = L_recon + 10.0*L_grad + 1.0*L_herm + 5.0*L_trace + 0.5*L_idem

    Args:
        weights: Loss component weights
        dt: Time step for gradient loss
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        dt: float = 1.0,
    ):
        super().__init__()

        if weights is None:
            weights = LossWeights()
        self.weights = weights

        self.reconstruction_loss = DensityReconstructionLoss()
        self.gradient_loss = GradientLoss(dt=dt)
        self.hermitian_loss = HermiticityLoss()
        self.trace_loss = TraceLoss()
        self.idempotency_loss = IdempotencyLoss()

    def forward(
        self,
        rho_pred: Tensor,
        rho_target: Tensor,
        rho_current: Tensor,
        overlap: Tensor,
        n_electrons: int,
        return_components: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute combined loss.

        Args:
            rho_pred: Predicted next density
            rho_target: Target next density
            rho_current: Current density
            overlap: Overlap matrix
            n_electrons: Number of electrons
            return_components: If True, also return individual loss components

        Returns:
            Total loss, optionally with component dictionary
        """
        components = {}

        # Reconstruction loss
        l_recon = self.reconstruction_loss(rho_pred, rho_target)
        components['reconstruction'] = l_recon

        # Gradient loss
        l_grad = self.gradient_loss(rho_pred, rho_current, rho_target)
        components['gradient'] = l_grad

        # Hermiticity loss
        l_herm = self.hermitian_loss(rho_pred)
        components['hermitian'] = l_herm

        # Trace loss
        l_trace = self.trace_loss(rho_pred, overlap, n_electrons)
        components['trace'] = l_trace

        # Idempotency loss
        l_idem = self.idempotency_loss(rho_pred, overlap)
        components['idempotent'] = l_idem

        # Weighted sum
        total_loss = (
            self.weights.reconstruction * l_recon
            + self.weights.gradient * l_grad
            + self.weights.hermitian * l_herm
            + self.weights.trace * l_trace
            + self.weights.idempotent * l_idem
        )

        if return_components:
            return total_loss, components
        return total_loss


class LatentSpaceLoss(nn.Module):
    """
    Loss in latent space for faster training.

    Instead of computing losses in full density matrix space,
    compute them on the latent representation.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        z_pred: Tensor,
        z_target: Tensor,
    ) -> Tensor:
        """
        Compute latent space MSE.

        Args:
            z_pred: Predicted latent, shape (..., latent_dim)
            z_target: Target latent, shape (..., latent_dim)

        Returns:
            Scalar loss
        """
        return (z_pred - z_target).pow(2).mean()


class TrajectoryLoss(nn.Module):
    """
    Loss for multi-step trajectory prediction.

    Weights losses at different time horizons, with optional
    discounting for later steps.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        discount: float = 0.95,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.discount = discount

    def forward(
        self,
        trajectory_pred: Tensor,
        trajectory_target: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute trajectory loss.

        Args:
            trajectory_pred: Predicted trajectory, shape (n_steps, ...)
            trajectory_target: Target trajectory, shape (n_steps, ...)
            **kwargs: Additional arguments for base loss

        Returns:
            Scalar loss
        """
        n_steps = trajectory_pred.shape[0]
        total_loss = torch.tensor(0.0, device=trajectory_pred.device)
        weight_sum = 0.0

        for t in range(n_steps):
            weight = self.discount ** t
            weight_sum += weight

            step_loss = self.base_loss(
                trajectory_pred[t],
                trajectory_target[t],
                **kwargs,
            )
            total_loss = total_loss + weight * step_loss

        return total_loss / weight_sum


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning robust latent representations.

    Encourages similar densities to have similar latents,
    and different densities to have different latents.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_anchor: Tensor,
        z_positive: Tensor,
        z_negatives: Tensor,
    ) -> Tensor:
        """
        Compute contrastive loss.

        Args:
            z_anchor: Anchor latent, shape (batch, latent_dim)
            z_positive: Positive sample latent, shape (batch, latent_dim)
            z_negatives: Negative sample latents, shape (batch, n_neg, latent_dim)

        Returns:
            Scalar loss
        """
        # Similarity with positive
        pos_sim = torch.sum(z_anchor * z_positive, dim=-1) / self.temperature

        # Similarity with negatives
        neg_sim = torch.sum(
            z_anchor.unsqueeze(1) * z_negatives,
            dim=-1
        ) / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(z_anchor.shape[0], dtype=torch.long, device=z_anchor.device)

        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class ScaledLosses(nn.Module):
    """
    Scaled losses from TDDFTNet - normalize by target magnitude.

    Prevents large targets from dominating training. Uses:
    - Scaled-L2: sqrt(Σ(pred-true)² / Σ(true)²)
    - Scaled dipole error for physical observables

    Reference: TDDFTNet (ICLR 2025)
    """

    def __init__(self, dipole_weight: float = 0.1, eps: float = 1e-10):
        super().__init__()
        self.dipole_weight = dipole_weight
        self.eps = eps

    def scaled_l2(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        Scaled L2 loss: sqrt(Σ(pred-true)² / Σ(true)²)

        Scale-invariant metric that prevents large-magnitude
        targets from dominating the loss.
        """
        diff_sq = (pred - true).abs().pow(2).sum()
        true_sq = true.abs().pow(2).sum() + self.eps
        return torch.sqrt(diff_sq / true_sq)

    def scaled_mae(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        Scaled MAE: Σ|pred-true| / Σ|true|
        """
        diff = (pred - true).abs().sum()
        true_abs = true.abs().sum() + self.eps
        return diff / true_abs

    def scaled_dipole(
        self,
        rho_pred: Tensor,
        rho_true: Tensor,
        dipole_integrals: Tensor,
    ) -> Tensor:
        """
        Scaled dipole error for density matrices.

        Computes dipole moment μ = Tr(ρ·D) and returns relative error.

        Args:
            rho_pred: Predicted density, shape (..., n, n) complex
            rho_true: True density, shape (..., n, n) complex
            dipole_integrals: Dipole matrices, shape (3, n, n)

        Returns:
            Scaled dipole error
        """
        # Compute dipole moments: μ = Tr(ρ·D)
        dipole_pred = torch.einsum("...ij,cji->...c", rho_pred, dipole_integrals).real
        dipole_true = torch.einsum("...ij,cji->...c", rho_true, dipole_integrals).real

        # Relative error
        diff_norm = (dipole_pred - dipole_true).norm()
        true_norm = dipole_true.norm() + self.eps

        return diff_norm / true_norm

    def forward(
        self,
        rho_pred: Tensor,
        rho_true: Tensor,
        dipole_integrals: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute scaled loss.

        Args:
            rho_pred: Predicted density
            rho_true: True density
            dipole_integrals: Optional dipole matrices for dipole loss

        Returns:
            Total scaled loss
        """
        loss = self.scaled_l2(rho_pred, rho_true)

        if dipole_integrals is not None:
            dipole_loss = self.scaled_dipole(rho_pred, rho_true, dipole_integrals)
            loss = loss + self.dipole_weight * dipole_loss

        return loss


class VarianceWeightedLoss(nn.Module):
    """
    Variance-weighted reconstruction loss.

    Weights density matrix elements inversely by their variance,
    preventing the model from ignoring low-variance (but important)
    off-diagonal elements.

    w_ij = 1 / (variance_ij + eps)
    L = Σ w_ij * |ρ_pred_ij - ρ_true_ij|²
    """

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps
        self.register_buffer('variance_weights', None)

    def compute_variance_weights(self, trajectories: list[Tensor]) -> Tensor:
        """
        Compute variance weights from training trajectories.

        Args:
            trajectories: List of density trajectories, each (n_steps, n_basis, n_basis)

        Returns:
            Variance weights, shape (n_basis, n_basis)
        """
        # Stack all densities
        all_rho = torch.cat([t.reshape(-1, t.shape[-2], t.shape[-1]) for t in trajectories])

        # Compute element-wise variance
        variance = all_rho.var(dim=0).abs()

        # Inverse variance weighting
        weights = 1.0 / (variance + self.eps)

        # Normalize
        weights = weights / weights.sum()

        return weights

    def set_weights(self, weights: Tensor):
        """Set precomputed variance weights."""
        self.register_buffer('variance_weights', weights)

    def forward(self, rho_pred: Tensor, rho_true: Tensor) -> Tensor:
        """
        Compute variance-weighted loss.

        Args:
            rho_pred: Predicted density, shape (..., n, n)
            rho_true: True density, shape (..., n, n)

        Returns:
            Weighted loss
        """
        diff_sq = (rho_pred - rho_true).abs().pow(2)

        if self.variance_weights is not None:
            # Apply weights
            weighted = self.variance_weights * diff_sq
            return weighted.sum(dim=(-2, -1)).mean()
        else:
            # Fall back to unweighted
            return diff_sq.mean()


def create_loss_function(
    loss_type: str = "physics_aware",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss ("physics_aware", "mse", "latent")
        **kwargs: Additional arguments

    Returns:
        Loss module
    """
    if loss_type == "physics_aware":
        return PhysicsAwareLoss(**kwargs)
    elif loss_type == "mse":
        return DensityReconstructionLoss(**kwargs)
    elif loss_type == "latent":
        return LatentSpaceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
