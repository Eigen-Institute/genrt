#!/usr/bin/env python
"""
Evaluation script for RT-TDDFT ML models.

Usage:
    # Basic evaluation
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --data data/processed/test

    # Evaluate with specific metrics
    python scripts/evaluate.py --checkpoint model.pt --data test_data --metrics mse mae dipole

    # Evaluate with spectral analysis
    python scripts/evaluate.py --checkpoint model.pt --data test_data --spectrum --dt 0.1

    # Save detailed results
    python scripts/evaluate.py --checkpoint model.pt --data test_data --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import (
    TrajectoryMetrics,
    frobenius_error,
    dipole_error,
    trace_violation,
    hermiticity_violation,
    idempotency_violation,
    compute_trajectory_metrics,
    compute_step_errors,
    compute_absorption_spectrum,
    spectrum_overlap,
    MetricsAccumulator,
)
from src.inference.predictor import Predictor, RolloutConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RT-TDDFT ML model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to evaluation data (HDF5 file or directory)",
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON format)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mse", "mae", "relative_error", "max_error"],
        choices=["mse", "mae", "relative_error", "max_error",
                 "dipole", "trace", "hermiticity", "idempotency", "spectrum"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum rollout steps per trajectory",
    )

    # Spectral analysis
    parser.add_argument(
        "--spectrum",
        action="store_true",
        help="Compute absorption spectrum metrics",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step in atomic units (for spectrum)",
    )
    parser.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        default=None,
        help="Frequency range for spectrum (min max in eV)",
    )

    # Physics projection
    parser.add_argument(
        "--apply-projection",
        action="store_true",
        help="Apply physics projection during rollout",
    )
    parser.add_argument(
        "--mcweeney-iterations",
        type=int,
        default=3,
        help="McWeeney iterations for physics projection",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to load model directly or from state dict
    if "model" in checkpoint:
        model = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        # Need to reconstruct model - this requires knowing the architecture
        # For now, assume the full model is saved
        raise ValueError(
            "Checkpoint contains only state_dict. "
            "Please provide checkpoint with full model."
        )
    else:
        # Assume checkpoint is the model directly
        model = checkpoint

    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_data(data_path: str, max_trajectories: Optional[int] = None):
    """Load evaluation data."""
    data_path = Path(data_path)

    if data_path.is_file() and data_path.suffix in [".h5", ".hdf5"]:
        # Single HDF5 file
        from src.data.trajectory import Trajectory
        trajectory = Trajectory.load(data_path)
        return [trajectory]

    elif data_path.is_dir():
        # Directory of HDF5 files
        from src.data.trajectory import Trajectory
        trajectories = []
        files = list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5"))

        if max_trajectories:
            files = files[:max_trajectories]

        for f in files:
            try:
                trajectories.append(Trajectory.load(f))
            except Exception as e:
                warnings.warn(f"Failed to load {f}: {e}")

        return trajectories

    else:
        raise ValueError(f"Invalid data path: {data_path}")


def evaluate_trajectory(
    model,
    trajectory,
    config: RolloutConfig,
    device: str,
    compute_dipole: bool = False,
    compute_spectrum: bool = False,
    dt: Optional[float] = None,
    freq_range: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Evaluate model on a single trajectory."""

    # Extract data from trajectory
    densities = torch.tensor(trajectory.densities, device=device, dtype=torch.complex64)
    fields = torch.tensor(trajectory.fields, device=device, dtype=torch.float32)
    overlap = torch.tensor(trajectory.overlap, device=device, dtype=torch.complex64)
    n_electrons = trajectory.n_electrons

    geometry = {
        "positions": torch.tensor(trajectory.positions, device=device, dtype=torch.float32),
        "atomic_numbers": torch.tensor(trajectory.atomic_numbers, device=device),
    }

    # Get dipole integrals if available and needed
    dipole_integrals = None
    if compute_dipole or compute_spectrum:
        if hasattr(trajectory, "dipole_integrals") and trajectory.dipole_integrals is not None:
            dipole_integrals = torch.tensor(
                trajectory.dipole_integrals, device=device, dtype=torch.complex64
            )

    # Setup prediction
    predictor = Predictor(model)
    max_steps = config.max_steps or (len(densities) - 1)

    # Run rollout
    with torch.no_grad():
        result = predictor.rollout(
            initial_density=densities[0],
            geometry=geometry,
            field_sequence=fields[:max_steps + 1],
            overlap=overlap,
            n_electrons=n_electrons,
            config=config,
        )

    pred_trajectory = result.densities
    true_trajectory = densities[1:max_steps + 1]

    # Ensure same length
    min_len = min(len(pred_trajectory), len(true_trajectory))
    pred_trajectory = pred_trajectory[:min_len]
    true_trajectory = true_trajectory[:min_len]

    # Compute metrics
    metrics = compute_trajectory_metrics(
        trajectory_pred=pred_trajectory,
        trajectory_true=true_trajectory,
        overlap=overlap,
        n_electrons=n_electrons,
        dipole_integrals=dipole_integrals,
        dt=dt if compute_spectrum else None,
    )

    # Compute step-by-step errors
    step_errors = compute_step_errors(pred_trajectory, true_trajectory)

    results = {
        "mse": metrics.mse,
        "mae": metrics.mae,
        "relative_error": metrics.relative_error,
        "max_error": metrics.max_error,
        "trace_violation": metrics.trace_violation,
        "hermiticity_violation": metrics.hermiticity_violation,
        "n_steps": min_len,
    }

    if metrics.dipole_error is not None:
        results["dipole_error"] = metrics.dipole_error

    if metrics.spectrum_overlap is not None:
        results["spectrum_overlap"] = metrics.spectrum_overlap

    # Add step error statistics
    results["step_errors"] = {
        "final_mse": step_errors["step_mse"][-1].item(),
        "final_relative_error": step_errors["step_relative_error"][-1].item(),
        "cumulative_error": step_errors["cumulative_error"][-1].item(),
        "error_growth_rate": (
            step_errors["step_relative_error"][-1].item() /
            (step_errors["step_relative_error"][0].item() + 1e-10)
        ),
    }

    return results


def evaluate_with_mock_data(
    model,
    args,
    device: str,
) -> Dict[str, Any]:
    """Evaluate with mock data when real data unavailable."""
    print("Note: Using mock data for demonstration")

    # Create mock trajectory data
    n_steps = args.max_steps or 50
    n_basis = 4  # Small for testing

    # Random Hermitian density matrices
    densities = torch.randn(n_steps + 1, n_basis, n_basis, dtype=torch.complex64, device=device)
    densities = 0.5 * (densities + densities.conj().transpose(-2, -1))

    # Normalize trace
    overlap = torch.eye(n_basis, dtype=torch.complex64, device=device)
    n_electrons = 2
    for i in range(len(densities)):
        trace = torch.einsum("ij,ji->", densities[i], overlap).real
        densities[i] = densities[i] * (n_electrons / trace)

    fields = torch.randn(n_steps + 1, 3, device=device)

    geometry = {
        "positions": torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], device=device),
        "atomic_numbers": torch.tensor([1, 1], device=device),
    }

    # Run model prediction
    predictor = Predictor(model)
    config = RolloutConfig(
        max_steps=n_steps,
        apply_physics_projection=args.apply_projection,
    )

    with torch.no_grad():
        result = predictor.rollout(
            initial_density=densities[0],
            geometry=geometry,
            field_sequence=fields,
            overlap=overlap,
            n_electrons=n_electrons,
            config=config,
        )

    pred = result.densities
    true = densities[1:n_steps + 1]

    min_len = min(len(pred), len(true))
    pred = pred[:min_len]
    true = true[:min_len]

    # Compute metrics
    metrics = compute_trajectory_metrics(
        trajectory_pred=pred,
        trajectory_true=true,
        overlap=overlap,
        n_electrons=n_electrons,
    )

    return {
        "mse": metrics.mse,
        "mae": metrics.mae,
        "relative_error": metrics.relative_error,
        "max_error": metrics.max_error,
        "trace_violation": metrics.trace_violation,
        "hermiticity_violation": metrics.hermiticity_violation,
        "n_trajectories": 1,
        "n_steps": min_len,
        "mock_data": True,
    }


def main():
    args = parse_args()

    if not args.quiet:
        print(f"Loading model from {args.checkpoint}")

    # Load model
    try:
        model, checkpoint = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    if not args.quiet:
        print(f"Model loaded on {args.device}")

    # Setup rollout config
    config = RolloutConfig(
        max_steps=args.max_steps,
        apply_physics_projection=args.apply_projection,
        mcweeney_iterations=args.mcweeney_iterations if args.apply_projection else 0,
    )

    # Load data
    try:
        trajectories = load_data(args.data, args.max_trajectories)
        if not args.quiet:
            print(f"Loaded {len(trajectories)} trajectories")
    except Exception as e:
        if args.verbose:
            print(f"Could not load data: {e}")
        # Fall back to mock evaluation
        results = evaluate_with_mock_data(model, args, args.device)

        if not args.quiet:
            print("\n" + "=" * 50)
            print("EVALUATION RESULTS (Mock Data)")
            print("=" * 50)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6e}")
                else:
                    print(f"  {key}: {value}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        return

    # Evaluate each trajectory
    accumulator = MetricsAccumulator()
    all_results = []

    compute_dipole = "dipole" in args.metrics
    compute_spectrum = args.spectrum or "spectrum" in args.metrics

    iterator = trajectories
    if not args.quiet:
        iterator = tqdm(trajectories, desc="Evaluating")

    for traj in iterator:
        try:
            result = evaluate_trajectory(
                model=model,
                trajectory=traj,
                config=config,
                device=args.device,
                compute_dipole=compute_dipole,
                compute_spectrum=compute_spectrum,
                dt=args.dt,
                freq_range=tuple(args.freq_range) if args.freq_range else None,
            )

            accumulator.add(result)
            all_results.append(result)

        except Exception as e:
            if args.verbose:
                print(f"Error evaluating trajectory: {e}")

    # Compute summary statistics
    summary = accumulator.compute_summary()

    # Print results
    if not args.quiet:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Evaluated {len(all_results)} trajectories")
        print()

        for metric, stats in summary.items():
            if metric in ["step_errors"]:
                continue
            print(f"{metric}:")
            print(f"  mean:   {stats['mean']:.6e}")
            print(f"  std:    {stats['std']:.6e}")
            print(f"  min:    {stats['min']:.6e}")
            print(f"  max:    {stats['max']:.6e}")
            print(f"  median: {stats['median']:.6e}")
            print()

    # Save results
    if args.output:
        output_data = {
            "summary": summary,
            "n_trajectories": len(all_results),
            "config": {
                "checkpoint": args.checkpoint,
                "data": args.data,
                "max_steps": args.max_steps,
                "apply_projection": args.apply_projection,
                "device": args.device,
            },
        }

        if args.verbose:
            output_data["per_trajectory"] = all_results

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        if not args.quiet:
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
