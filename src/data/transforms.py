"""
Data augmentation transforms for RT-TDDFT trajectories.

This module provides transforms for augmenting training data, including
geometry noise injection, field perturbations, and composition of
multiple transforms.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Optional, Callable, Union
from abc import ABC, abstractmethod


class BaseTransform(ABC):
    """Abstract base class for transforms."""

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """Apply transform to sample."""
        pass


class ComposeTransforms(BaseTransform):
    """
    Compose multiple transforms into a single transform.

    Args:
        transforms: List of transforms to apply in order
    """

    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class GeometryNoiseTransform(BaseTransform):
    """
    Add Gaussian noise to atomic positions.

    This augmentation helps the model generalize across slightly
    different geometries, simulating thermal fluctuations.

    Args:
        noise_std: Standard deviation of Gaussian noise in Angstroms
        seed: Optional random seed for reproducibility
    """

    def __init__(self, noise_std: float = 0.05, seed: Optional[int] = None):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, sample: dict) -> dict:
        positions = sample["positions"]

        if isinstance(positions, Tensor):
            noise = torch.randn_like(positions) * self.noise_std
            sample["positions"] = positions + noise
        else:
            noise = self.rng.randn(*positions.shape) * self.noise_std
            sample["positions"] = positions + noise

        return sample


class FieldNoiseTransform(BaseTransform):
    """
    Add Gaussian noise to external field vectors.

    This augmentation helps the model handle slight variations
    in the applied field.

    Args:
        noise_std: Standard deviation of Gaussian noise in atomic units
        seed: Optional random seed for reproducibility
    """

    def __init__(self, noise_std: float = 0.001, seed: Optional[int] = None):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, sample: dict) -> dict:
        field = sample["field"]

        if isinstance(field, Tensor):
            noise = torch.randn_like(field) * self.noise_std
            sample["field"] = field + noise
        else:
            noise = self.rng.randn(*field.shape) * self.noise_std
            sample["field"] = field + noise

        return sample


class RandomRotationTransform(BaseTransform):
    """
    Apply random SO(3) rotation to geometry and field.

    This augmentation enforces rotational equivariance by providing
    rotated training examples. Note: If using E3NN, equivariance
    is built into the model, so this may not be necessary.

    Args:
        seed: Optional random seed for reproducibility
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate a random 3x3 rotation matrix using QR decomposition."""
        # Random matrix
        A = self.rng.randn(3, 3)
        # QR decomposition gives orthogonal Q
        Q, R = np.linalg.qr(A)
        # Ensure proper rotation (det = +1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    def __call__(self, sample: dict) -> dict:
        R = self._random_rotation_matrix()
        R_tensor = torch.tensor(R, dtype=torch.float32)

        # Rotate positions
        positions = sample["positions"]
        if isinstance(positions, Tensor):
            sample["positions"] = positions @ R_tensor.T
        else:
            sample["positions"] = positions @ R.T

        # Rotate field
        field = sample["field"]
        if isinstance(field, Tensor):
            sample["field"] = field @ R_tensor.T
        else:
            sample["field"] = field @ R.T

        # Store rotation for potential use
        sample["_rotation_matrix"] = R_tensor

        return sample


class DensityNoiseTransform(BaseTransform):
    """
    Add small noise to density matrix elements.

    This can help with regularization by preventing overfitting
    to exact density values.

    Args:
        noise_std: Standard deviation of noise (relative to element magnitude)
        seed: Optional random seed for reproducibility
    """

    def __init__(self, noise_std: float = 0.001, seed: Optional[int] = None):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, sample: dict) -> dict:
        density = sample["density_current"]

        if isinstance(density, Tensor):
            # Add noise to real and imaginary parts
            noise_real = torch.randn_like(density.real) * self.noise_std
            noise_imag = torch.randn_like(density.imag) * self.noise_std

            # Scale noise by element magnitude
            magnitude = density.abs() + 1e-10
            noisy_density = density + magnitude * (noise_real + 1j * noise_imag)

            # Re-hermitianize
            noisy_density = 0.5 * (noisy_density + noisy_density.conj().transpose(-1, -2))

            sample["density_current"] = noisy_density
        else:
            noise_real = self.rng.randn(*density.shape) * self.noise_std
            noise_imag = self.rng.randn(*density.shape) * self.noise_std
            magnitude = np.abs(density) + 1e-10
            noisy_density = density + magnitude * (noise_real + 1j * noise_imag)
            noisy_density = 0.5 * (noisy_density + noisy_density.conj().T)
            sample["density_current"] = noisy_density

        return sample


class TimeShiftTransform(BaseTransform):
    """
    Random time offset for trajectory sampling.

    Instead of using fixed stride, randomly offset the start point
    to get more diverse samples.

    Args:
        max_shift: Maximum number of steps to shift
        seed: Optional random seed for reproducibility
    """

    def __init__(self, max_shift: int = 5, seed: Optional[int] = None):
        self.max_shift = max_shift
        self.rng = np.random.RandomState(seed)

    def __call__(self, sample: dict) -> dict:
        # This transform is typically applied at the dataset level
        # rather than per-sample, but we include it for completeness
        sample["_time_shift"] = self.rng.randint(0, self.max_shift + 1)
        return sample


class BasisMaskTransform(BaseTransform):
    """
    Randomly mask a fraction of basis functions during training.

    This augmentation encourages the model to not overfit to specific
    basis function patterns.

    Args:
        mask_prob: Probability of masking each basis function
        seed: Optional random seed for reproducibility
    """

    def __init__(self, mask_prob: float = 0.1, seed: Optional[int] = None):
        self.mask_prob = mask_prob
        self.rng = np.random.RandomState(seed)

    def __call__(self, sample: dict) -> dict:
        density = sample["density_current"]
        n_basis = density.shape[-1]

        # Generate mask
        mask_1d = self.rng.random(n_basis) > self.mask_prob

        if isinstance(density, Tensor):
            mask_1d = torch.tensor(mask_1d, device=density.device)
            # Create 2D mask for matrix
            mask_2d = mask_1d.unsqueeze(-1) & mask_1d.unsqueeze(0)
            # Apply mask (zero out masked elements)
            sample["density_current"] = density * mask_2d.unsqueeze(0)
            sample["density_next"] = sample["density_next"] * mask_2d.unsqueeze(0)
        else:
            mask_2d = np.outer(mask_1d, mask_1d)
            sample["density_current"] = density * mask_2d
            sample["density_next"] = sample["density_next"] * mask_2d

        sample["_basis_mask"] = mask_1d

        return sample


class NormalizeFieldTransform(BaseTransform):
    """
    Normalize field to unit maximum amplitude.

    Useful when training on different field strengths.

    Args:
        eps: Small value to avoid division by zero
    """

    def __init__(self, eps: float = 1e-10):
        self.eps = eps

    def __call__(self, sample: dict) -> dict:
        field = sample["field"]

        if isinstance(field, Tensor):
            max_amp = field.abs().max() + self.eps
            sample["field"] = field / max_amp
            sample["_field_scale"] = max_amp
        else:
            max_amp = np.abs(field).max() + self.eps
            sample["field"] = field / max_amp
            sample["_field_scale"] = max_amp

        return sample


def get_training_transforms(
    geometry_noise: float = 0.05,
    field_noise: float = 0.001,
    use_rotation: bool = False,
    seed: Optional[int] = None,
) -> ComposeTransforms:
    """
    Get standard training transforms.

    Args:
        geometry_noise: Std of geometry noise in Angstroms
        field_noise: Std of field noise in atomic units
        use_rotation: Whether to apply random rotations
        seed: Random seed for reproducibility

    Returns:
        Composed transform for training data augmentation
    """
    transforms = []

    if geometry_noise > 0:
        transforms.append(GeometryNoiseTransform(geometry_noise, seed))

    if field_noise > 0:
        transforms.append(FieldNoiseTransform(field_noise, seed))

    if use_rotation:
        transforms.append(RandomRotationTransform(seed))

    return ComposeTransforms(transforms)


def get_validation_transforms() -> ComposeTransforms:
    """
    Get transforms for validation/test data (no augmentation).

    Returns:
        Empty composed transform (identity)
    """
    return ComposeTransforms([])
