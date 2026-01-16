# genrt User's Guide

A practical guide to using the RT-TDDFT ML accelerator for density matrix prediction.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training Models](#training-models)
4. [Evaluation](#evaluation)
5. [Inference](#inference)
6. [Working with Notebooks](#working-with-notebooks)

---

## Quick Start

### Environment Setup

```bash
# Activate the virtual environment
source ~/pyvenv/bin/activate

# Verify installation
python -c "import torch; from e3nn import o3; print('Ready!')"
```

### Minimal Training Example

```python
from src.data import Trajectory, TrajectoryDataset
from src.models import RTTDDFTModel
from src.training import Trainer, TrainerConfig

# Load a trajectory
traj = Trajectory.load("data/processed/h2_eq_delta_z.h5")
dataset = TrajectoryDataset(traj)

# Create model
model = RTTDDFTModel(
    max_atomic_number=10,
    hidden_dim=128,
    n_mamba_layers=4,
)

# Train
config = TrainerConfig(epochs=50, learning_rate=1e-4)
trainer = Trainer(model, config)
trainer.fit(dataset)
```

---

## Data Preparation

### Understanding the Data Format

Trajectories are stored in HDF5 files containing:
- **Geometry**: Atomic positions and atomic numbers
- **Density matrices**: Complex matrices evolving over time
- **External field**: Applied electric field at each timestep
- **Metadata**: Molecule name, basis set, XC functional, etc.

### Loading Existing Trajectories

```python
from src.data import Trajectory

# Load a single trajectory
traj = Trajectory.load("data/processed/h2_eq_delta_z.h5")

print(f"Molecule: {traj.molecule}")
print(f"Atoms: {traj.n_atoms}")
print(f"Basis functions: {traj.n_basis}")
print(f"Time steps: {traj.n_steps}")
print(f"Electrons: {traj.n_electrons}")

# Access density matrices (complex)
density = traj.density  # shape: (n_steps, n_spin, n_basis, n_basis)

# Access as PyTorch tensors
positions = traj.get_positions_tensor()
density_tensor = traj.get_density_tensor()
```

### Using the Simulation Index

If you have trajectory metadata in a pandas DataFrame:

```python
from src.data import SimulationIndex

# Load the index
index = SimulationIndex.load("rt_simulations.pkl")

# Explore available data
print(f"Total simulations: {len(index)}")
print(f"Molecules: {index.molecules}")
print(f"Geometries: {index.geometries}")
print(f"Field types: {index.field_types}")

# View summary
print(index.summary())

# Filter trajectories
h2_sims = index.filter(molecule="h2", field_type="delta")
small_basis = index.filter(max_nbf=20)
equilibrium = index.filter(geometry="eq", has_data=True)

# Iterate over records
for record in h2_sims:
    print(f"{record.calc_name}: {record.nbf} basis, {record.tsteps} steps")
```

### Converting NWChem Output

#### From a single directory:

```bash
python scripts/convert_nwchem.py \
    --input data/raw/h2_calculation/ \
    --output data/processed/h2_eq_delta_z.h5 \
    --molecule h2 \
    --n-basis 4 \
    --n-electrons 2
```

#### From a simulation index:

```bash
# Convert all trajectories in the index
python scripts/convert_nwchem.py \
    --index rt_simulations.pkl \
    --output data/processed/

# With filters
python scripts/convert_nwchem.py \
    --index rt_simulations.pkl \
    --output data/processed/ \
    --filter-molecule h2 \
    --filter-geometry eq \
    --max-trajectories 10
```

#### Programmatically:

```python
from src.data import NWChemParser, SimulationIndex

# Parse from simulation record
index = SimulationIndex.load("rt_simulations.pkl")
parser = NWChemParser()

record = index.filter(molecule="h2")[0]
trajectory = parser.parse_trajectory_from_record(record)
trajectory.save("output.h5")
```

### Creating Datasets

```python
from src.data import TrajectoryDataset, UnifiedTrajectoryDataset
from torch.utils.data import DataLoader

# Single trajectory dataset
traj = Trajectory.load("data/processed/h2.h5")
dataset = TrajectoryDataset(traj, stride=1)

# Multiple trajectories (variable basis sizes)
paths = list(Path("data/processed").glob("*.h5"))
unified_dataset = UnifiedTrajectoryDataset(paths)

# Create DataLoader (use custom collate for variable basis)
from src.data.dataset import collate_variable_basis
loader = DataLoader(
    unified_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_variable_basis,
)
```

### Data Augmentation

```python
from src.data import (
    GeometryNoiseTransform,
    FieldNoiseTransform,
    ComposeTransforms,
    get_training_transforms,
)

# Individual transforms
geom_noise = GeometryNoiseTransform(noise_std=0.01)
field_noise = FieldNoiseTransform(noise_std=0.0001)

# Compose transforms
transform = ComposeTransforms([geom_noise, field_noise])

# Or use factory function
transform = get_training_transforms(
    geometry_noise=0.01,
    field_noise=0.0001,
    use_rotation=True,
)

# Apply to dataset
dataset = TrajectoryDataset(traj, transform=transform)
```

---

## Training Models

### Training Phases

The recommended training progression:

#### Phase 1: Single Molecule (H2+)

```bash
python scripts/train.py --config configs/training/phase1_h2p.yaml
```

Or with CLI options:
```bash
python scripts/train.py \
    --data data/processed/h2p*.h5 \
    --epochs 100 \
    --lr 3e-4 \
    --horizon-stages 16 32
```

#### Phase 2: Multi-Molecule

```bash
python scripts/train.py --config configs/training/phase2_multi_mol.yaml
```

#### Phase 3: Full Generalization

```bash
python scripts/train.py --config configs/training/phase3_generalization.yaml
```

### Training Configuration

```python
from src.training import TrainerConfig, Trainer
from src.models import RTTDDFTModel

model = RTTDDFTModel(
    max_atomic_number=10,
    hidden_dim=256,
    n_mamba_layers=6,
)

config = TrainerConfig(
    epochs=100,
    learning_rate=1e-4,
    batch_size=32,

    # Loss weights
    reconstruction_weight=1.0,
    derivative_weight=10.0,
    hermiticity_weight=1.0,
    trace_weight=5.0,
    idempotency_weight=0.5,

    # Curriculum
    use_curriculum=True,
    horizon_stages=[16, 32, 48, 64],

    # Checkpointing
    checkpoint_dir="checkpoints/",
    save_every=10,
)

trainer = Trainer(model, config)
trainer.fit(train_dataset, val_dataset)
```

### Curriculum Learning

```python
from src.training import HorizonCurriculum, TemporalBundling

# Progressive horizon expansion
curriculum = HorizonCurriculum(
    stages=[16, 32, 48, 64],
    epochs_per_stage=25,
)

# Temporal bundling (predict multiple steps)
bundling = TemporalBundling(bundle_size=4)
```

### Monitoring Training

```python
from src.utils import plot_training_curves, create_training_dashboard

# After training
plot_training_curves(
    trainer.train_losses,
    trainer.val_losses,
    save_path="training_curves.png"
)

# Full dashboard
create_training_dashboard(
    train_losses=trainer.train_losses,
    val_losses=trainer.val_losses,
    loss_components=trainer.loss_history,
    save_path="dashboard.png"
)
```

---

## Evaluation

### Running Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pt \
    --data data/processed/test/

# With spectrum analysis
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pt \
    --data data/processed/test/ \
    --spectrum \
    --dt 0.2

# Save results to JSON
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pt \
    --data data/processed/test/ \
    --output results.json
```

### Programmatic Evaluation

```python
from src.utils.metrics import (
    frobenius_error,
    trace_violation,
    hermiticity_violation,
    compute_trajectory_metrics,
)

# Single-step metrics
error = frobenius_error(predicted, target, normalized=True)
trace_err = trace_violation(predicted, overlap, n_electrons)
herm_err = hermiticity_violation(predicted)

# Full trajectory metrics
metrics = compute_trajectory_metrics(
    predicted_trajectory,
    target_trajectory,
    overlap=overlap,
    n_electrons=n_electrons,
)

print(f"MSE: {metrics.mse:.6f}")
print(f"Relative Error: {metrics.relative_error:.4f}")
print(f"Max Trace Violation: {metrics.trace_violation:.6f}")
```

### Spectrum Analysis

```python
from src.utils.metrics import compute_absorption_spectrum, spectrum_overlap

# Compute absorption spectrum from dipole trajectory
freqs, spectrum = compute_absorption_spectrum(
    dipole_trajectory,
    dt=0.2,
    padding_factor=4,
)

# Compare spectra
overlap = spectrum_overlap(spectrum_pred, spectrum_ref)
print(f"Spectrum overlap: {overlap:.4f}")
```

### Visualization

```python
from src.utils import (
    plot_error_accumulation,
    plot_physics_violations,
    plot_absorption_spectrum,
    plot_density_comparison,
)

# Error accumulation over time
plot_error_accumulation(
    step_errors,
    cumulative_errors,
    dt=0.2,
    save_path="error_accumulation.png"
)

# Physics constraint violations
plot_physics_violations(
    trace_violations,
    hermiticity_violations,
    idempotency_violations,
    save_path="violations.png"
)

# Absorption spectrum comparison
plot_absorption_spectrum(
    freqs,
    spectrum_pred,
    spectrum_ref,
    labels=["ML", "Reference"],
    save_path="spectrum.png"
)

# Density matrix comparison
plot_density_comparison(
    density_pred,
    density_ref,
    title="Step 100",
    save_path="density_comparison.png"
)
```

---

## Inference

### Basic Prediction

```python
from src.inference import Predictor

# Load model
model = RTTDDFTModel.load("checkpoints/model.pt")
predictor = Predictor(model)

# Predict trajectory
predicted = predictor.rollout(
    initial_density=rho_0,
    positions=positions,
    atomic_numbers=atomic_numbers,
    field_sequence=fields,
    overlap=overlap,
    n_steps=500,
)
```

### Streaming Prediction (Memory Efficient)

```python
from src.inference import StreamingPredictor

predictor = StreamingPredictor(model)
predictor.initialize(initial_density, positions, atomic_numbers, overlap)

for step in range(n_steps):
    field = get_field_at_step(step)
    density = predictor.step(field)

    # Process density immediately
    dipole = compute_dipole(density, positions)
    save_to_file(density, step)
```

### Uncertainty Estimation

```python
from src.inference import EnsembleUncertainty

# Load ensemble of models
models = [RTTDDFTModel.load(f"checkpoints/model_{i}.pt") for i in range(5)]
ensemble = EnsembleUncertainty(models)

# Predict with uncertainty
mean_density, uncertainty = ensemble.predict_with_uncertainty(
    initial_density, positions, atomic_numbers, field, overlap
)

print(f"Uncertainty: {uncertainty.mean():.6f}")
```

### Adaptive Re-anchoring

For long trajectories, periodically re-anchor to DFT:

```python
from src.inference import AdaptiveReAnchorController

controller = AdaptiveReAnchorController(
    max_ml_steps=100,
    uncertainty_threshold=0.01,
    trace_threshold=0.001,
)

predictor = StreamingPredictor(model)
predictor.initialize(initial_density, positions, atomic_numbers, overlap)

for step in range(n_steps):
    # Check if re-anchoring needed
    action = controller.check(
        density=current_density,
        overlap=overlap,
        n_electrons=n_electrons,
        uncertainty=uncertainty,
    )

    if action == "reanchor":
        # Run DFT to get ground truth
        current_density = run_dft_step(current_density, field)
        predictor.reset_state(current_density)
        controller.record_reanchor(step)
    else:
        # Continue with ML prediction
        current_density = predictor.step(field)
```

---

## Working with Notebooks

### Data Exploration (01_data_exploration.ipynb)

Use this notebook to:
- Load and inspect trajectories
- Visualize molecular geometries
- Plot density matrix evolution
- Check data quality

```python
# Example cells
traj = Trajectory.load("data/processed/h2_eq_delta_z.h5")

# Plot density evolution
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, step in enumerate([0, 100, 200]):
    im = axes[i].imshow(traj.density_real[step, 0].real, cmap='RdBu')
    axes[i].set_title(f"Step {step}")
plt.colorbar(im)
```

### Model Debugging (02_model_debugging.ipynb)

Use this notebook to:
- Test model forward pass
- Check gradient flow
- Verify equivariance properties
- Debug training issues

```python
# Check equivariance
from tests.conftest import random_rotation_matrix
R = random_rotation_matrix()

out_original = model(positions, atomic_numbers, density, field, overlap)
out_rotated = model(positions @ R.T, atomic_numbers, density, field @ R.T, overlap)

# Rotate output and compare
# (uses Wigner-D matrices for proper comparison)
```

### Results Analysis (03_results_analysis.ipynb)

Use this notebook to:
- Compare predicted vs reference trajectories
- Analyze error accumulation
- Generate publication figures
- Compute final metrics

```python
# Load results and create comparison plots
from src.utils import create_training_dashboard

create_training_dashboard(
    train_losses=results['train_loss'],
    val_losses=results['val_loss'],
    loss_components=results['components'],
    learning_rates=results['lr'],
    save_path='figures/dashboard.png'
)
```

---

## Common Issues and Solutions

### Out of Memory

- Reduce batch size
- Use `StreamingPredictor` for long trajectories
- Enable gradient checkpointing

### Training Instability

- Reduce learning rate
- Increase derivative loss weight
- Use gradient clipping
- Start with shorter horizons

### Poor Generalization

- Add more diverse training molecules
- Increase geometry noise augmentation
- Use molecule ladder curriculum
- Check for data leakage in splits

### Slow Training

- Enable mixed precision (`use_amp=True`)
- Use DataLoader with multiple workers
- Pre-compute molecular graphs

---

## API Reference

For detailed API documentation, see the docstrings in:
- `src/data/` - Data loading and processing
- `src/models/` - Neural network architectures
- `src/training/` - Training utilities
- `src/inference/` - Inference and prediction
- `src/utils/` - Metrics and visualization
