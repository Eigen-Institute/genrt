"""
Mamba-based dynamics module with FiLM conditioning.

This module provides the GeometryConditionedMamba class which uses
Mamba State Space Models to predict density matrix evolution, conditioned
on molecular geometry through Feature-wise Linear Modulation (FiLM).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


@dataclass
class DynamicsConfig:
    """Configuration for the dynamics model."""
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 6
    dropout: float = 0.1


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies affine transformation conditioned on external input:
        y = gamma * x + beta

    where gamma and beta are generated from the conditioning input.

    Args:
        feature_dim: Dimension of features to modulate
        condition_dim: Dimension of conditioning input
        use_sigmoid_gamma: If True, apply sigmoid to gamma for bounded modulation
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        use_sigmoid_gamma: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_sigmoid_gamma = use_sigmoid_gamma

        # Generate gamma and beta from condition
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Initialize to identity transform
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: Features to modulate, shape (..., feature_dim)
            condition: Conditioning input, shape (..., condition_dim) or (condition_dim,)

        Returns:
            Modulated features, shape (..., feature_dim)
        """
        # Broadcast condition if needed
        if condition.dim() < x.dim():
            condition = condition.unsqueeze(0).expand(x.shape[0], -1)

        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)

        if self.use_sigmoid_gamma:
            # Bounded modulation: gamma in (0, 2) centered at 1
            gamma = 2.0 * torch.sigmoid(gamma)

        return gamma * x + beta


class MambaBlock(nn.Module):
    """
    Single Mamba block with optional FiLM conditioning.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Expansion factor for inner dimension
        condition_dim: Dimension of conditioning input (None for no conditioning)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        condition_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package not installed. "
                "Install with: pip install mamba-ssm"
            )

        self.d_model = d_model

        # Layer norm before Mamba
        self.norm = nn.LayerNorm(d_model)

        # Mamba layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Optional FiLM conditioning
        self.film = None
        if condition_dim is not None:
            self.film = FiLMLayer(d_model, condition_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of Mamba block.

        Args:
            x: Input sequence, shape (batch, seq_len, d_model)
            condition: Optional conditioning, shape (batch, condition_dim)

        Returns:
            Output sequence, shape (batch, seq_len, d_model)
        """
        # Pre-norm
        residual = x
        x = self.norm(x)

        # Apply FiLM conditioning if available
        if self.film is not None and condition is not None:
            # Expand condition to sequence length
            cond_expanded = condition.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = self.film(x, cond_expanded)

        # Mamba
        x = self.mamba(x)

        # Residual + dropout
        x = residual + self.dropout(x)

        return x


class GeometryConditionedMamba(nn.Module):
    """
    Mamba-based dynamics model conditioned on molecular geometry.

    Uses stacked Mamba blocks with FiLM conditioning from geometry encoder
    outputs to predict density matrix evolution.

    Architecture:
    1. Project input latent to model dimension
    2. Stack of Mamba blocks with geometry conditioning via FiLM
    3. Project output to latent dimension

    Args:
        latent_dim: Dimension of input/output latent representation
        d_model: Internal model dimension
        d_state: Mamba SSM state dimension
        d_conv: Mamba convolution width
        expand: Mamba expansion factor
        n_layers: Number of Mamba blocks
        geometry_dim: Dimension of geometry conditioning
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int = 256,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 6,
        geometry_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(latent_dim, d_model)

        # Geometry conditioning projection
        self.geometry_proj = nn.Sequential(
            nn.Linear(geometry_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Mamba blocks with FiLM
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                condition_dim=d_model,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

    def forward(
        self,
        z: Tensor,
        geometry_features: Tensor,
        field_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict next latent state(s) from current state(s).

        Args:
            z: Latent density representation, shape (batch, seq_len, latent_dim)
               or (batch, latent_dim) for single step
            geometry_features: Geometry conditioning, shape (batch, geometry_dim)
            field_features: Optional field conditioning, shape (batch, field_dim)

        Returns:
            Next latent state(s), shape matches input z
        """
        # Handle single step input
        squeeze_output = False
        if z.dim() == 2:
            z = z.unsqueeze(1)
            squeeze_output = True

        # Project input
        x = self.input_proj(z)

        # Project geometry condition
        condition = self.geometry_proj(geometry_features)

        # Optionally incorporate field features
        if field_features is not None:
            # Simple addition for now - could be more sophisticated
            condition = condition + field_features[..., :condition.shape[-1]]

        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x, condition)

        # Project output
        output = self.output_proj(x)

        if squeeze_output:
            output = output.squeeze(1)

        return output

    def predict_sequence(
        self,
        z0: Tensor,
        geometry_features: Tensor,
        n_steps: int,
        field_sequence: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Autoregressively predict a sequence of latent states.

        Args:
            z0: Initial latent state, shape (batch, latent_dim)
            geometry_features: Geometry conditioning, shape (batch, geometry_dim)
            n_steps: Number of steps to predict
            field_sequence: Optional field for each step, shape (batch, n_steps, field_dim)

        Returns:
            Predicted sequence, shape (batch, n_steps, latent_dim)
        """
        batch_size = z0.shape[0]
        device = z0.device

        predictions = []
        z_current = z0

        for t in range(n_steps):
            # Get field for this step if provided
            field_t = None
            if field_sequence is not None:
                field_t = field_sequence[:, t]

            # Predict next state
            z_next = self.forward(z_current, geometry_features, field_t)

            predictions.append(z_next)
            z_current = z_next

        return torch.stack(predictions, dim=1)


class SimplifiedMamba(nn.Module):
    """
    Simplified Mamba-like architecture for when mamba-ssm is not available.

    Uses standard components (conv + gating) to approximate Mamba behavior.
    This is useful for testing and environments without CUDA support.

    Args:
        d_model: Model dimension
        d_state: State dimension (used for hidden dim calculation)
        d_conv: Convolution kernel size
        expand: Expansion factor
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, 2 * d_inner)

        # Causal convolution
        self.conv = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model)

        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project and split
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv(x_conv)[..., :seq_len]  # Causal: trim right
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)

        # Gating
        x_gated = x_conv * torch.sigmoid(z)

        # Output projection
        return self.out_proj(x_gated)


class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba block using SimplifiedMamba.

    For use when mamba-ssm is not available.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        condition_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimplifiedMamba(d_model, d_state, d_conv, expand)
        self.film = FiLMLayer(d_model, condition_dim) if condition_dim else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.norm(x)
        if self.film is not None and condition is not None:
            cond_expanded = condition.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = self.film(x, cond_expanded)
        x = self.mamba(x)
        return residual + self.dropout(x)


def create_dynamics_model(
    config: Optional[DynamicsConfig] = None,
    latent_dim: int = 256,
    geometry_dim: int = 128,
    use_simplified: bool = False,
) -> GeometryConditionedMamba:
    """
    Factory function to create a dynamics model.

    Args:
        config: Optional configuration dataclass
        latent_dim: Dimension of latent space
        geometry_dim: Dimension of geometry features
        use_simplified: Force use of simplified (non-Mamba) implementation

    Returns:
        GeometryConditionedMamba instance
    """
    if config is None:
        config = DynamicsConfig()

    # Check if we should use simplified version
    if use_simplified or not MAMBA_AVAILABLE:
        # Swap block class in the model
        model = GeometryConditionedMamba(
            latent_dim=latent_dim,
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            n_layers=config.n_layers,
            geometry_dim=geometry_dim,
            dropout=config.dropout,
        )
        # Replace blocks with simplified versions
        model.blocks = nn.ModuleList([
            SimplifiedMambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                condition_dim=config.d_model,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        return model

    return GeometryConditionedMamba(
        latent_dim=latent_dim,
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        n_layers=config.n_layers,
        geometry_dim=geometry_dim,
        dropout=config.dropout,
    )
