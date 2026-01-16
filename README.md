# genrt

**Machine Learning Accelerator for Real-Time Time-Dependent Density Functional Theory (RT-TDDFT)**

A PyTorch-based framework for predicting electron density matrix dynamics using E3NN equivariant neural networks and Mamba sequence modeling.

## Overview

RT-TDDFT simulations are computationally expensive, requiring iterative solution of the time-dependent Kohn-Sham equations at each timestep. This project aims to accelerate these simulations by training neural networks to predict density matrix evolution, potentially achieving 10-100x speedup while maintaining chemical accuracy.

### Key Features

- **SO(3)-Equivariant Geometry Encoding**: E3NN-based encoder that respects rotational symmetry of molecular systems
- **Variable Basis Set Support**: Handle molecules with different numbers of basis functions in unified datasets
- **Physics-Aware Training**: Loss functions that enforce Hermiticity, trace conservation, and idempotency constraints
- **Curriculum Learning**: Progressive training from simple (H2+) to complex molecules
- **Adaptive Re-anchoring**: Hybrid ML/DFT inference with uncertainty-based re-anchoring to ground truth

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/chemicalcraig/genrt.git
cd genrt

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Verify installation
python -c "import torch; from e3nn import o3; print('Ready!')"
```

### Dependencies

Core dependencies include:
- `torch` - Deep learning framework
- `e3nn` - Equivariant neural networks
- `mamba-ssm` - State space models for sequence modeling
- `h5py` - HDF5 file I/O for trajectories
- `numpy`, `scipy` - Numerical computing
- `pandas` - Data manipulation

## Quick Start

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

## Project Structure

```
genrt/
├── configs/                 # YAML configuration files
│   ├── model/              # Model architecture configs
│   └── training/           # Training phase configs
├── data/
│   ├── basis/              # NWChem basis set files
│   ├── raw/                # Raw NWChem outputs (gitignored)
│   └── processed/          # HDF5 trajectories (gitignored)
├── docs/
│   └── users_guide.md      # Comprehensive user documentation
├── notebooks/
│   └── 01_data_exploration.ipynb
├── scripts/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── convert_nwchem.py   # Data conversion
├── src/
│   ├── data/               # Data loading and processing
│   ├── models/             # Neural network architectures
│   ├── physics/            # Physics constraints and projections
│   ├── training/           # Training utilities
│   ├── inference/          # Prediction and re-anchoring
│   └── utils/              # Metrics and visualization
└── tests/                  # Test suite (254 tests)
```

## Data Format

Trajectories are stored in HDF5 files containing:
- **Geometry**: Atomic positions and atomic numbers
- **Density matrices**: Complex matrices evolving over time (n_steps, n_spin, n_basis, n_basis)
- **External field**: Applied electric field at each timestep
- **Overlap matrix**: Basis function overlap matrix
- **Metadata**: Molecule name, basis set, XC functional, etc.

### Working with NWChem Data

```python
from src.data import SimulationIndex, NWChemParser

# Load simulation metadata index
index = SimulationIndex.load("rt_simulations.pkl")
print(f"Available molecules: {index.molecules}")

# Filter and convert trajectories
h2_sims = index.filter(molecule="h2", field_type="delta")
parser = NWChemParser()

for record in h2_sims:
    traj = parser.parse_trajectory_from_record(record)
    traj.save(f"data/processed/{record.calc_name}.h5")
```

## Model Architecture

The model consists of four main components:

1. **GeometryEncoder** (E3NN): Encodes molecular geometry into SO(3)-equivariant node features
2. **DensityEncoder**: Projects variable-size density matrices to fixed-dimension latent space
3. **Dynamics** (Mamba): Predicts temporal evolution of latent density representation
4. **DensityDecoder**: Reconstructs density matrices with physics constraint projections

## Training

Training follows a curriculum approach:

```bash
# Phase 1: Single molecule (H2+)
python scripts/train.py --config configs/training/phase1_h2p.yaml

# Phase 2: Multi-molecule
python scripts/train.py --config configs/training/phase2_multi_mol.yaml

# Phase 3: Full generalization
python scripts/train.py --config configs/training/phase3_generalization.yaml
```

### Loss Function

The physics-aware loss combines reconstruction with constraint terms:

```
L = L_recon + 10.0*L_grad + 1.0*L_herm + 5.0*L_trace + 0.5*L_idem
```

## Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pt \
    --data data/processed/test/

# With spectrum analysis
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pt \
    --data data/processed/test/ \
    --spectrum --dt 0.2
```

## Documentation

- **[User's Guide](docs/users_guide.md)**: Comprehensive documentation with examples
- **[CLAUDE.md](CLAUDE.md)**: Technical reference for AI assistants

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_equivariance.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{genrt2024,
  title={genrt: Machine Learning Accelerator for RT-TDDFT},
  author={Craig, Chemical},
  year={2024},
  url={https://github.com/chemicalcraig/genrt}
}
```

## Acknowledgments

- [e3nn](https://e3nn.org/) - Euclidean neural networks
- [Mamba](https://github.com/state-spaces/mamba) - State space models
- [NWChem](https://nwchemgit.github.io/) - Quantum chemistry software
