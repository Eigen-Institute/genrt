"""
Ensemble uncertainty estimation for RT-TDDFT predictions.

Provides uncertainty quantification through:
- Deep ensemble predictions
- MC Dropout approximation
- Calibration metrics
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates."""
    mean: Tensor  # Mean prediction
    std: Tensor  # Standard deviation
    epistemic: Optional[Tensor] = None  # Model uncertainty
    aleatoric: Optional[Tensor] = None  # Data uncertainty
    samples: Optional[Tensor] = None  # Raw ensemble samples


class EnsembleUncertainty(nn.Module):
    """
    Ensemble-based uncertainty estimation.

    Uses multiple trained models to estimate prediction uncertainty
    via disagreement between ensemble members.
    """

    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = "mean",
    ):
        """
        Args:
            models: List of trained model instances
            aggregation: How to aggregate predictions ("mean", "median")
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.aggregation = aggregation

    @torch.no_grad()
    def forward(
        self,
        *args,
        return_samples: bool = False,
        **kwargs,
    ) -> UncertaintyEstimate:
        """
        Get ensemble prediction with uncertainty.

        Args:
            *args, **kwargs: Arguments passed to each model
            return_samples: Whether to include raw ensemble samples

        Returns:
            UncertaintyEstimate with mean, std, and optionally samples
        """
        predictions = []

        for model in self.models:
            pred = model(*args, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]  # Take first output if model returns tuple
            predictions.append(pred)

        predictions = torch.stack(predictions)  # (n_models, ...)

        # Aggregate
        if self.aggregation == "mean":
            mean = predictions.mean(dim=0)
        elif self.aggregation == "median":
            mean = predictions.median(dim=0).values
        else:
            mean = predictions.mean(dim=0)

        # Compute uncertainty as standard deviation
        std = predictions.std(dim=0)

        return UncertaintyEstimate(
            mean=mean,
            std=std,
            epistemic=std,  # For ensembles, std is epistemic uncertainty
            samples=predictions if return_samples else None,
        )

    def predict_trajectory(
        self,
        initial_density: Tensor,
        geometry: Dict[str, Tensor],
        field_sequence: Tensor,
        overlap: Tensor,
        n_electrons: int,
        physics_projection: Optional[nn.Module] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict trajectory with uncertainty estimates.

        Args:
            initial_density: Starting density
            geometry: Geometry dict
            field_sequence: Field sequence (n_steps, 3)
            overlap: Overlap matrix
            n_electrons: Number of electrons
            physics_projection: Optional physics projection

        Returns:
            Tuple of (mean_trajectory, uncertainty_trajectory)
        """
        n_steps = field_sequence.shape[0]
        device = initial_density.device

        mean_traj = [initial_density]
        std_traj = [torch.zeros_like(initial_density.real)]

        # Use mean prediction for autoregressive rollout
        rho = initial_density
        hidden_states = [None] * self.n_models

        for t in range(n_steps):
            field = field_sequence[t]

            # Get predictions from all models
            predictions = []
            new_hidden_states = []

            for i, model in enumerate(self.models):
                batch = self._make_batch(rho, geometry, field)

                if hasattr(model, 'forward_with_hidden'):
                    pred, h_new, _ = model.forward_with_hidden(
                        batch, hidden_states[i]
                    )
                else:
                    pred = model(batch)
                    h_new = None

                if physics_projection is not None:
                    pred = physics_projection(pred, overlap, n_electrons)

                predictions.append(pred)
                new_hidden_states.append(h_new)

            hidden_states = new_hidden_states
            predictions = torch.stack(predictions)

            # Mean and std
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)

            mean_traj.append(mean)
            std_traj.append(std.abs())  # Take magnitude for complex

            # Use mean for next step
            rho = mean

        return torch.stack(mean_traj), torch.stack(std_traj)

    def _make_batch(
        self,
        rho: Tensor,
        geometry: Dict[str, Tensor],
        field: Tensor,
    ) -> Dict[str, Tensor]:
        """Create batch dictionary."""
        batch = {
            'density': rho.unsqueeze(0),
            'field': field.unsqueeze(0),
            **{k: v.unsqueeze(0) if v.dim() > 0 else v for k, v in geometry.items()},
        }
        return batch

    def calibration_error(
        self,
        predictions: Tensor,
        uncertainties: Tensor,
        targets: Tensor,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics for uncertainty estimates.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates (std)
            targets: Ground truth values
            n_bins: Number of bins for calibration

        Returns:
            Dict with calibration metrics
        """
        errors = (predictions - targets).abs()

        # Expected calibration error
        ece = 0.0
        bin_edges = torch.linspace(0, uncertainties.max(), n_bins + 1)

        for i in range(n_bins):
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_conf = uncertainties[mask].mean()
                bin_acc = errors[mask].mean()
                ece += mask.sum().float() / len(uncertainties) * abs(bin_conf - bin_acc)

        # Negative log-likelihood (assuming Gaussian)
        nll = 0.5 * (
            torch.log(2 * math.pi * uncertainties.pow(2)) +
            errors.pow(2) / uncertainties.pow(2)
        ).mean()

        # Coefficient of variation
        cv = uncertainties.mean() / (predictions.abs().mean() + 1e-10)

        return {
            'ece': ece.item(),
            'nll': nll.item(),
            'mean_uncertainty': uncertainties.mean().item(),
            'cv': cv.item(),
        }


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout uncertainty estimation.

    Uses dropout at inference time to approximate Bayesian inference.
    More efficient than ensembles but may be less accurate.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        dropout_rate: Optional[float] = None,
    ):
        """
        Args:
            model: Trained model with dropout layers
            n_samples: Number of MC samples
            dropout_rate: Override dropout rate (if None, uses model's rate)
        """
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                if self.dropout_rate is not None:
                    module.p = self.dropout_rate

    def _disable_dropout(self):
        """Restore dropout layers to eval mode."""
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        *args,
        return_samples: bool = False,
        **kwargs,
    ) -> UncertaintyEstimate:
        """
        Get MC Dropout prediction with uncertainty.

        Args:
            *args, **kwargs: Arguments passed to model
            return_samples: Whether to include raw samples

        Returns:
            UncertaintyEstimate
        """
        self._enable_dropout()

        samples = []
        for _ in range(self.n_samples):
            pred = self.model(*args, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            samples.append(pred)

        self._disable_dropout()

        samples = torch.stack(samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        return UncertaintyEstimate(
            mean=mean,
            std=std,
            epistemic=std,
            samples=samples if return_samples else None,
        )


class QuantileUncertainty(nn.Module):
    """
    Quantile regression-based uncertainty estimation.

    Predicts multiple quantiles to estimate uncertainty bounds.
    Requires specially trained model with quantile outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        quantiles: List[float] = None,
    ):
        """
        Args:
            model: Model trained with quantile loss
            quantiles: Quantile values to predict (e.g., [0.1, 0.5, 0.9])
        """
        super().__init__()
        self.model = model
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> UncertaintyEstimate:
        """Get quantile predictions."""
        quantile_preds = self.model(*args, **kwargs)

        # Assume model outputs dict with quantile keys or stacked tensor
        if isinstance(quantile_preds, dict):
            lower = quantile_preds[self.quantiles[0]]
            median = quantile_preds[self.quantiles[1]]
            upper = quantile_preds[self.quantiles[-1]]
        else:
            lower = quantile_preds[0]
            median = quantile_preds[len(self.quantiles) // 2]
            upper = quantile_preds[-1]

        # Estimate std from quantile range
        # For normal distribution, 10th-90th percentile is ~2.56 std
        std = (upper - lower) / 2.56

        return UncertaintyEstimate(
            mean=median,
            std=std,
        )


class UncertaintyAggregator:
    """
    Aggregate uncertainty across density matrix elements.

    Provides scalar uncertainty metrics for decision making.
    """

    def __init__(
        self,
        method: str = "frobenius",
        weights: Optional[Tensor] = None,
    ):
        """
        Args:
            method: Aggregation method ("frobenius", "max", "trace", "weighted")
            weights: Optional weights for weighted aggregation
        """
        self.method = method
        self.weights = weights

    def __call__(self, uncertainty: Tensor) -> Tensor:
        """
        Aggregate uncertainty to scalar.

        Args:
            uncertainty: Uncertainty tensor, shape (..., n, n)

        Returns:
            Scalar uncertainty value
        """
        if self.method == "frobenius":
            return uncertainty.pow(2).sum(dim=(-2, -1)).sqrt()
        elif self.method == "max":
            return uncertainty.abs().amax(dim=(-2, -1))
        elif self.method == "trace":
            return uncertainty.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        elif self.method == "weighted" and self.weights is not None:
            return (self.weights * uncertainty.abs()).sum(dim=(-2, -1))
        else:
            return uncertainty.abs().mean(dim=(-2, -1))

    def relative_uncertainty(
        self,
        uncertainty: Tensor,
        prediction: Tensor,
        eps: float = 1e-10,
    ) -> Tensor:
        """
        Compute relative uncertainty.

        Args:
            uncertainty: Absolute uncertainty
            prediction: Prediction value
            eps: Small constant for numerical stability

        Returns:
            Relative uncertainty
        """
        pred_norm = prediction.abs().pow(2).sum(dim=(-2, -1)).sqrt() + eps
        uncert_norm = self(uncertainty)
        return uncert_norm / pred_norm


def create_ensemble(
    model_class: type,
    model_config: Dict,
    n_models: int = 5,
    checkpoint_paths: Optional[List[str]] = None,
) -> EnsembleUncertainty:
    """
    Factory function to create ensemble uncertainty estimator.

    Args:
        model_class: Model class to instantiate
        model_config: Configuration dict for model
        n_models: Number of ensemble members
        checkpoint_paths: Optional list of checkpoint paths to load

    Returns:
        EnsembleUncertainty instance
    """
    models = []

    for i in range(n_models):
        model = model_class(**model_config)

        if checkpoint_paths is not None and i < len(checkpoint_paths):
            state_dict = torch.load(checkpoint_paths[i], map_location='cpu')
            model.load_state_dict(state_dict)

        model.eval()
        models.append(model)

    return EnsembleUncertainty(models)
