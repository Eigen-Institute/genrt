# RT-TDDFT ML Accelerator: Development Guide

**Project:** Universal Quantum Dynamics Model for RT-TDDFT Acceleration  
**Version:** 1.1  
**Target:** Claude Code Implementation  
**Last Updated:** January 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Related Work: TDDFTNet Comparison](#2-related-work-tddftnet-comparison)
3. [Architecture Decisions](#3-architecture-decisions)
4. [Environment Setup](#4-environment-setup)
5. [Project Structure](#5-project-structure)
6. [Data Formats](#6-data-formats)
7. [Implementation Phases](#7-implementation-phases)
8. [Core Components](#8-core-components)
9. [Testing & Validation](#9-testing--validation)
10. [References](#10-references)

---

## 1. Project Overview

### 1.1 Goal

Build a machine learning model that accelerates Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) simulations by learning to predict density matrix evolution. The model must generalize across:

- **Geometries:** Different molecular configurations
- **Molecules:** Different atomic compositions (H₂⁺ → C₆H₆)
- **External fields:** Delta kicks, Gaussian pulses, resonant excitations
- **Basis sets:** Variable number of basis functions (4 → 66+)

### 1.2 Core Challenge

Current approaches use fixed-size neural networks that only work for a single molecular system. We need an architecture where **parameter count is independent of basis set size**.

### 1.3 Solution Summary

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  E3NN Geometry  │────▶│  Mamba Dynamics │────▶│ Physics-Aware   │
│    Encoder      │     │    Module       │     │    Decoder      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
   SO(3) equivariant      Geometry-conditioned    Hermitian + Trace
   molecular features     temporal evolution      constraint projection
```

---

## 2. Related Work: TDDFTNet Comparison

### 2.1 TDDFTNet Overview (ICLR 2025)

TDDFTNet is a concurrent work that models RT-TDDFT by evolving electron density on a real-space 3D grid. Key characteristics:

- **Representation:** Volumetric density ρ(r) on 48×48×48 grid
- **Architecture:** 3D U-Net + Volume-Atom Message Passing
- **Physics:** Density normalization (‖x₀‖² = ρ₀), conservation (Σ⟨x₀, dx_t⟩ = 0)
- **Training:** Curriculum learning (16→32→48→60 steps), temporal bundling
- **Results:** ~500× speedup, 98-99% spectrum overlap on water/ethanol/malondialdehyde

### 3.2 Why Our Approach Differs

| Aspect | TDDFTNet | Our Approach | Rationale |
|--------|----------|--------------|-----------|
| **Representation** | Real-space ρ(r) | Basis-set ρ_μν | Compact, interfaces with NWChem |
| **Variable size** | Fixed grid | Variable nbf | Generalizes across molecules |
| **Temporal memory** | None explicit | Mamba SSM | Phase coherence over 1000s of steps |
| **Equivariance** | Data augmentation | E3NN built-in | Exact symmetry, better generalization |
| **Scalability** | O(N³_grid) | O(N²_basis) sparse | Larger molecules feasible |

### 3.3 Key Insights from TDDFTNet to Incorporate

**Validated by their ablations:**

1. **Latent evolution >> Autoregressive** (Table 2: 57% vs 137% dipole error)
   - Confirms our Mamba choice for maintaining latent state
   
2. **Markov assumption fails for density**
   - "the electron density at one time step cannot fully determine the quantum state"
   - Supports our use of Mamba's selective state space for long-range memory

3. **Physics-aware constraints help** (Table 2: TDDFTNet vs MP-U-Net)
   - Density normalization improved all metrics
   - Our analogue: Hermiticity + trace + idempotency projection

**Training strategies to adopt:**

4. **Curriculum training on prediction horizon**
   ```python
   # Stage 1: 16 steps, Stage 2: 32 steps, Stage 3: 48 steps, Stage 4: 60 steps
   curriculum_steps = [16, 32, 48, 60]
   ```

5. **Temporal bundling** — predict 2 steps together
   ```python
   # Predict (t+1, t+2), use t+2 as input for next propagation
   x_t1, x_t2 = propagator(x_t, geometry)
   next_input = x_t2  # Skip connection past t+1
   ```

6. **Scaled losses** — normalize by target magnitude
   ```python
   # Scaled-L2: sqrt(Σ(pred-true)² / Σ(true)²)
   scaled_l2 = torch.sqrt((pred - true).pow(2).sum() / true.pow(2).sum())
   ```

### 3.4 What TDDFTNet Lacks (Our Advantages)

1. **No explicit long-range memory** — Their latent evolution has no mechanism to remember distant past beyond what's encoded in x_t. Our Mamba SSM explicitly maintains history.

2. **Fixed grid size** — Cannot handle molecules larger than training distribution. Our attention-based projection handles any nbf.

3. **Approximate equivariance** — 90° rotation augmentation is not continuous SO(3). Our E3NN is exactly equivariant.

4. **No re-anchoring strategy** — They don't address error accumulation for very long rollouts (>60 steps). Our adaptive re-anchoring enables arbitrary length simulations.

5. **Real-space only** — Requires Octopus-style codes. Our basis-set approach works with NWChem, Gaussian, ORCA, etc.

---

## 3. Architecture Decisions

### 3.1 Dynamics Module: Mamba SSM (NOT Neural ODE)

**Decision:** Use discrete-step Mamba State Space Model over continuous-time Neural ODEs.

**Rationale:**

| Criterion | Neural ODE | Mamba SSM |
|-----------|------------|-----------|
| Memory | No explicit memory | Selective state space retains history |
| Compute | 20-100 RHS evals/step | 1 forward pass/step |
| Phase coherence | Poor (no long-range memory) | Good (SSM hidden state) |
| Stiff dynamics | Struggles with multi-scale | Handles implicitly |

**Key insight:** Quantum coherences persist for thousands of timesteps. Mamba's selective gating learns *which* history to retain—critical for phase-accurate rollouts.

**Implementation:**
```python
# Recommended: Mamba with Hamiltonian-inspired residual
z_mamba = mamba_block(z_t, h_t, geometry_cond)
z_physics = antisymmetric_layer(z_mamba, field_t)  # Approximates [H, ρ]
z_{t+1} = z_t + dt * (z_mamba + λ * z_physics)
```

### 3.2 Variable Basis Size: Density-Edge Projection

**Decision:** Use orbital-level graph with bilinear density decoding (NOT block-level attention).

**Rationale:**

| Approach | Granularity | Scalability | Physics |
|----------|-------------|-------------|---------|
| Block encoder | Atom pairs | O(N_atoms²) | Loses orbital structure |
| **Density-Edge** | Basis functions | O(N_basis²), sparsifiable | Preserves orbital coupling |

**Implementation:**
```python
# Each basis function → latent vector v_μ
v_mu = orbital_encoder(atom_features[atom_idx], Z, l, m)

# Density elements via equivariant tensor product
rho_ij = tensor_product(v_mu[i], v_mu[j])  # → scalar (L=0)
```

### 3.3 Geometry Conditioning: FiLM (NOT Cross-Attention)

**Decision:** Use Feature-wise Linear Modulation for static geometry conditioning.

**Rationale:** Geometry is constant throughout a trajectory. FiLM is 10× cheaper than cross-attention and more stable (bounded γ via sigmoid).

**Implementation:**
```python
class FiLMConditioner(nn.Module):
    def forward(self, x, geometry_embedding):
        gamma = self.gamma_net(geometry_embedding).sigmoid()
        beta = self.beta_net(geometry_embedding)
        return gamma * x + beta
```

### 3.4 Field Encoding: L=1 Spherical Tensor

**Decision:** Encode external field E(t) as L=1 odd-parity irrep for SO(3) equivariance.

**Rationale:** Light-matter coupling is μ·E where dipole μ transforms as L=1. Preserves selection rules.

**Implementation:**
```python
from e3nn import o3

field_irreps = o3.Irreps("1x1o")  # L=1, odd parity
# Couple with geometry via tensor product
coupled = tensor_product(geometry_features, field_spherical)
```

### 3.5 Loss Function: Derivative Loss + Variance Weighting

**Decision:** Include explicit derivative matching and variance-based element weighting.

**Rationale:** 
- Derivative loss prevents phase drift in oscillatory dynamics
- Variance weighting prevents model from ignoring low-variance (but important) off-diagonal elements

**Implementation:**
```python
L = L_recon + 10.0 * L_derivative + 1.0 * L_hermiticity + 5.0 * L_trace + 0.5 * L_idempotency

# Variance weighting
w_ij = 1.0 / (variance_ij + 1e-10)
L_recon = (w_ij * |ρ_pred - ρ_true|²).sum()
```

---

## 4. Environment Setup

### 3.1 Dependencies

```yaml
# environment.yml
name: rt_tddft_ml
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch>=2.0
  - pytorch-cuda=11.8
  - pip
  - pip:
    # Core ML
    - mamba-ssm>=1.1.0          # CUDA-optimized Mamba
    - e3nn>=0.5.0               # SO(3) equivariant networks
    - torch-geometric>=2.3.0    # Graph neural networks
    - torch-scatter>=2.1.0      # Efficient scatter operations
    
    # Scientific
    - numpy>=1.24
    - scipy>=1.10
    - h5py>=3.8                 # HDF5 for trajectory storage
    
    # Training
    - wandb                     # Experiment tracking
    - einops                    # Tensor operations
    - tqdm
    
    # Optional: DFT interface
    - ase                       # Atomic simulation environment
    - pyscf                     # Python quantum chemistry (for testing)
```

### 3.2 Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate rt_tddft_ml

# Verify CUDA availability for Mamba
python -c "import mamba_ssm; print('Mamba OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify E3NN
python -c "from e3nn import o3; print(f'E3NN irreps: {o3.Irreps(\"1x0e + 1x1o\")}')"
```

### 3.3 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | A100/H100 (40GB+) |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD | 500 GB NVMe |

---

## 5. Project Structure

```
rt_tddft_ml/
├── README.md
├── pyproject.toml
├── environment.yml
│
├── configs/                    # Hydra/YAML configs
│   ├── model/
│   │   ├── geometry_encoder.yaml
│   │   ├── dynamics.yaml
│   │   └── decoder.yaml
│   ├── training/
│   │   ├── phase1_h2p.yaml
│   │   ├── phase2_multi_mol.yaml
│   │   └── phase3_generalization.yaml
│   └── experiment/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                   # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   ├── trajectory.py       # Trajectory container
│   │   ├── nwchem_parser.py    # Parse NWChem outputs
│   │   └── transforms.py       # Data augmentation
│   │
│   ├── models/                 # Neural network modules
│   │   ├── __init__.py
│   │   ├── geometry_encoder.py # E3NN molecular encoder
│   │   ├── density_encoder.py  # Density-Edge projection
│   │   ├── field_encoder.py    # Equivariant field encoding
│   │   ├── dynamics.py         # Mamba + FiLM conditioning
│   │   ├── decoder.py          # Latent → density matrix
│   │   └── full_model.py       # End-to-end model
│   │
│   ├── physics/                # Physics constraints
│   │   ├── __init__.py
│   │   ├── constraints.py      # Hermitian, trace, idempotency
│   │   ├── projections.py      # Hard constraint projection
│   │   └── observables.py      # Dipole, populations, etc.
│   │
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   ├── losses.py           # Loss functions
│   │   ├── curriculum.py       # Variance curriculum
│   │   ├── scheduler.py        # Learning rate scheduling
│   │   └── trainer.py          # Training loop
│   │
│   ├── inference/              # Inference and re-anchoring
│   │   ├── __init__.py
│   │   ├── predictor.py        # Rollout prediction
│   │   ├── uncertainty.py      # Ensemble uncertainty
│   │   └── reanchor.py         # Adaptive re-anchoring
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── complex_tensor.py   # Complex number utilities
│       ├── visualization.py    # Plotting
│       └── metrics.py          # Evaluation metrics
│
├── scripts/
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script
│   ├── generate_data.py        # NWChem data generation
│   └── convert_nwchem.py       # Convert NWChem → HDF5
│
├── tests/
│   ├── test_equivariance.py    # SO(3) equivariance tests
│   ├── test_physics.py         # Physics constraint tests
│   ├── test_data.py            # Data loading tests
│   └── test_model.py           # Model forward pass tests
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_debugging.ipynb
│   └── 03_results_analysis.ipynb
│
└── data/                       # Data directory (gitignored)
    ├── raw/                    # Raw NWChem outputs
    ├── processed/              # HDF5 trajectories
    └── splits/                 # Train/val/test splits
```

---

## 6. Data Formats

### 5.1 Trajectory HDF5 Schema

```python
# File: data/processed/{molecule}_{geometry}_{field}.h5

with h5py.File(filepath, 'r') as f:
    # Metadata
    f.attrs['molecule'] = 'h2o'
    f.attrs['n_atoms'] = 3
    f.attrs['n_electrons'] = 10
    f.attrs['n_basis'] = 13
    f.attrs['n_steps'] = 10000
    f.attrs['dt'] = 0.2  # atomic units
    f.attrs['charge'] = 0
    f.attrs['multiplicity'] = 1
    
    # Geometry (static)
    f['geometry/positions']     # (n_atoms, 3) float64, Angstroms
    f['geometry/atomic_numbers'] # (n_atoms,) int32
    
    # Basis set metadata
    f['basis/atom_index']       # (n_basis,) int32 - which atom
    f['basis/angular_momentum'] # (n_basis,) int32 - l quantum number
    f['basis/magnetic_quantum'] # (n_basis,) int32 - m quantum number
    
    # Static matrices
    f['matrices/overlap']       # (n_basis, n_basis) float64
    f['matrices/core_hamiltonian'] # (n_basis, n_basis) float64, optional
    
    # Time series
    f['dynamics/density_real']  # (n_steps, n_spin, n_basis, n_basis) float32
    f['dynamics/density_imag']  # (n_steps, n_spin, n_basis, n_basis) float32
    f['dynamics/field']         # (n_steps, 3) float32 - E(t) vector
    f['dynamics/time']          # (n_steps,) float32 - time points
    f['dynamics/energy']        # (n_steps,) float32 - total energy
    f['dynamics/dipole']        # (n_steps, 3) float32 - dipole moment
```

### 5.2 Molecular Graph Structure

```python
from torch_geometric.data import Data

graph = Data(
    # Node features (atoms)
    x=atomic_numbers,              # (n_atoms,) int - Z values
    pos=positions,                 # (n_atoms, 3) float - coordinates
    
    # Edge features (atom pairs)
    edge_index=edge_index,         # (2, n_edges) int - connectivity
    edge_attr=edge_vectors,        # (n_edges, 3) float - r_ij vectors
    
    # Basis function mapping
    basis_atom_map=basis_to_atom,  # (n_basis,) int - atom index per basis
    basis_l=angular_momentum,      # (n_basis,) int - l values
    basis_m=magnetic_quantum,      # (n_basis,) int - m values
    
    # Density matrix (current timestep)
    rho_real=rho.real,             # (n_spin, n_basis, n_basis) float
    rho_imag=rho.imag,             # (n_spin, n_basis, n_basis) float
    
    # External field
    field=field_vector,            # (3,) float - E(t)
    
    # Target (next timestep)
    y_real=rho_next.real,
    y_imag=rho_next.imag,
)
```

### 5.3 Model Input/Output Specification

```python
@dataclass
class ModelInput:
    """Input to the full model"""
    # Geometry (static per trajectory)
    positions: torch.Tensor         # (batch, n_atoms, 3)
    atomic_numbers: torch.Tensor    # (batch, n_atoms)
    edge_index: torch.Tensor        # (2, n_edges)
    
    # Basis metadata
    basis_atom_map: torch.Tensor    # (batch, n_basis)
    basis_l: torch.Tensor           # (batch, n_basis)
    
    # Dynamic inputs (per timestep)
    density: torch.Tensor           # (batch, n_spin, n_basis, n_basis) complex
    field: torch.Tensor             # (batch, 3) real
    
    # Static matrices
    overlap: torch.Tensor           # (batch, n_basis, n_basis) real


@dataclass  
class ModelOutput:
    """Output from the full model"""
    density_pred: torch.Tensor      # (batch, n_spin, n_basis, n_basis) complex
    latent_state: torch.Tensor      # (batch, latent_dim) - for next step
    uncertainty: torch.Tensor       # (batch,) - if ensemble
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Months 1-3)

#### Month 1: Data Infrastructure + E3NN Prototype

**Tasks:**
1. Implement `NWChemParser` to extract density matrices from restart files
2. Create `TrajectoryDataset` with HDF5 backend
3. Build `GeometryEncoder` using E3NN

**Files to create:**
```
src/data/nwchem_parser.py
src/data/dataset.py
src/models/geometry_encoder.py
tests/test_equivariance.py
```

**Key implementation:**
```python
# src/models/geometry_encoder.py
from e3nn import o3
from e3nn.nn import FullyConnectedNet

class GeometryEncoder(nn.Module):
    def __init__(
        self,
        irreps_out: str = "32x0e + 16x1o + 8x2e",
        num_layers: int = 4,
        max_radius: float = 5.0,
    ):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)
        
        # Atom embedding
        self.atom_embed = nn.Embedding(118, 64)
        
        # E3NN convolution layers
        self.convs = nn.ModuleList([
            e3nn.nn.models.gate_points_2101.Convolution(
                irreps_node_input="64x0e" if i == 0 else irreps_out,
                irreps_node_output=irreps_out,
                irreps_edge_attr="1x0e + 1x1o",  # distance + unit vector
                num_neighbors=10,
            )
            for i in range(num_layers)
        ])
```

**Verification test:**
```python
# tests/test_equivariance.py
def test_geometry_encoder_equivariance():
    """Verify SO(3) equivariance of geometry encoder"""
    model = GeometryEncoder()
    
    # Original geometry
    pos = torch.randn(5, 3)
    z = torch.tensor([1, 1, 6, 8, 8])  # H, H, C, O, O
    
    # Random rotation
    R = o3.rand_matrix()
    pos_rotated = pos @ R.T
    
    # Encode both
    out_original = model(pos, z)
    out_rotated = model(pos_rotated, z)
    
    # Apply Wigner-D to original
    D = model.irreps_out.D_from_matrix(R)
    out_transformed = out_original @ D.T
    
    # Should match
    assert torch.allclose(out_rotated, out_transformed, atol=1e-5)
```

#### Month 2: Geometry-Conditioned Mamba

**Tasks:**
1. Implement `GeometryConditionedMamba` with FiLM
2. Add `FieldEncoder` for equivariant field handling
3. Train on H2+ with multiple bond lengths

**Files to create:**
```
src/models/dynamics.py
src/models/field_encoder.py
configs/model/dynamics.yaml
```

**Key implementation:**
```python
# src/models/dynamics.py
from mamba_ssm import Mamba

class GeometryConditionedMamba(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        n_layers: int = 6,
        geometry_dim: int = 128,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaFiLMBlock(d_model, d_state, geometry_dim)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, geometry_embedding, hidden_states=None):
        """
        Args:
            x: (batch, seq_len, d_model) - encoded density sequence
            geometry_embedding: (batch, geometry_dim) - from E3NN
            hidden_states: Optional list of hidden states for continuation
        """
        new_hidden_states = []
        for i, layer in enumerate(self.layers):
            h = hidden_states[i] if hidden_states else None
            x, h_new = layer(x, geometry_embedding, h)
            new_hidden_states.append(h_new)
        return x, new_hidden_states


class MambaFiLMBlock(nn.Module):
    def __init__(self, d_model, d_state, geometry_dim):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state)
        self.film = FiLMConditioner(geometry_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, geometry_embedding, hidden_state=None):
        # Mamba with residual
        residual = x
        x = self.norm(x)
        x, h_new = self.mamba(x, hidden_state)
        
        # FiLM conditioning
        x = self.film(x, geometry_embedding)
        
        return x + residual, h_new
```

#### Month 3: Variable Basis Attention

**Tasks:**
1. Implement `DensityEdgeProjection` encoder
2. Implement `DensityDecoder` with equivariant tensor product
3. Integrate with LiH (first multi-electron system)

**Files to create:**
```
src/models/density_encoder.py
src/models/decoder.py
tests/test_variable_basis.py
```

**Key implementation:**
```python
# src/models/density_encoder.py
class DensityEdgeProjection(nn.Module):
    """
    Encode density matrix elements as edge features in orbital graph.
    Handles variable basis set sizes via shared orbital embeddings.
    """
    def __init__(self, latent_dim=256, max_l=2):
        super().__init__()
        
        # Element embedding (full periodic table)
        self.element_embed = nn.Embedding(118, 32)
        
        # Angular momentum embedding
        self.l_embed = nn.Embedding(max_l + 1, 16)
        
        # Orbital encoder: combines geometry + basis metadata
        self.orbital_encoder = nn.Sequential(
            nn.Linear(32 + 16 + 128, latent_dim),  # elem + l + geometry
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Density encoder: (rho_real, rho_imag) → features
        self.rho_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Cross-attention for aggregation
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=8)
        self.query_tokens = nn.Parameter(torch.randn(32, latent_dim))
        
    def forward(self, rho, atom_features, basis_metadata):
        """
        Args:
            rho: (n_basis, n_basis) complex - density matrix
            atom_features: (n_atoms, 128) - from geometry encoder (L=0 only)
            basis_metadata: dict with 'atom_idx', 'l', 'm' for each basis
        """
        n_basis = rho.shape[0]
        
        # Create orbital embeddings
        orbital_features = []
        for i in range(n_basis):
            atom_idx = basis_metadata['atom_idx'][i]
            l = basis_metadata['l'][i]
            
            atom_feat = atom_features[atom_idx]
            elem_feat = self.element_embed(basis_metadata['Z'][atom_idx])
            l_feat = self.l_embed(l)
            
            orb = self.orbital_encoder(torch.cat([atom_feat, elem_feat, l_feat]))
            orbital_features.append(orb)
        
        orbital_features = torch.stack(orbital_features)  # (n_basis, latent)
        
        # Encode density elements
        rho_flat = rho.reshape(-1)  # (n_basis²,)
        rho_features = self.rho_encoder(
            torch.stack([rho_flat.real, rho_flat.imag], dim=-1)
        )  # (n_basis², latent)
        
        # Create keys/values from orbital pairs
        i_idx, j_idx = torch.meshgrid(torch.arange(n_basis), torch.arange(n_basis))
        pair_features = orbital_features[i_idx.flatten()] + orbital_features[j_idx.flatten()]
        kv = pair_features + rho_features  # (n_basis², latent)
        
        # Attention pooling to fixed size
        queries = self.query_tokens.unsqueeze(1)  # (32, 1, latent)
        kv = kv.unsqueeze(1)  # (n_basis², 1, latent)
        
        out, _ = self.cross_attn(queries, kv, kv)
        return out.squeeze(1)  # (32, latent)
```

---

### Phase 2: Cross-Geometry (Months 4-6)

#### Month 4: GNN Message Passing + Equivariant Field

**Tasks:**
1. Add E3NN message passing to density encoder
2. Implement equivariant field-geometry coupling
3. Validate selection rules (dipole transitions)

**Key implementation:**
```python
# src/models/field_encoder.py
class EquivariantFieldEncoder(nn.Module):
    """Encode E(t) as L=1 spherical tensor, couple with geometry"""
    
    def __init__(self, geometry_irreps="32x0e + 16x1o + 8x2e"):
        super().__init__()
        self.field_irreps = o3.Irreps("1x1o")  # L=1, odd parity
        self.geometry_irreps = o3.Irreps(geometry_irreps)
        
        # Output: L=0 (scalar) + L=1 (vector) components
        self.output_irreps = o3.Irreps("16x0e + 16x1o + 8x2e")
        
        # Equivariant tensor product
        self.tp = o3.FullyConnectedTensorProduct(
            self.geometry_irreps,
            self.field_irreps,
            self.output_irreps,
        )
        
    def forward(self, geometry_features, field_cartesian):
        """
        Args:
            geometry_features: (batch, n_atoms, irreps_dim)
            field_cartesian: (batch, 3) - E_x, E_y, E_z
        """
        # E3NN expects (batch, 3) for "1x1o"
        coupled = self.tp(geometry_features, field_cartesian)
        return coupled
```

#### Month 5: Multi-Molecule Training

**Tasks:**
1. Joint training on H2+, H2, LiH
2. Implement variance curriculum
3. Add derivative loss

**Key implementation:**
```python
# src/training/losses.py
class PhysicsAwareLoss(nn.Module):
    def __init__(self, dt, lambda_grad=10.0, lambda_herm=1.0, 
                 lambda_trace=5.0, lambda_idem=0.5):
        super().__init__()
        self.dt = dt
        self.lambdas = {
            'grad': lambda_grad,
            'herm': lambda_herm,
            'trace': lambda_trace,
            'idem': lambda_idem,
        }
        
    def forward(self, rho_pred, rho_true, rho_prev, overlap, n_elec, weights=None):
        # Reconstruction (with optional variance weighting)
        diff = rho_pred - rho_true
        if weights is not None:
            L_recon = (weights * diff.abs()**2).mean()
        else:
            L_recon = diff.abs().pow(2).mean()
        
        # Derivative loss
        drho_pred = (rho_pred - rho_prev) / self.dt
        drho_true = (rho_true - rho_prev) / self.dt
        L_grad = (drho_pred - drho_true).abs().pow(2).mean()
        
        # Hermiticity
        L_herm = (rho_pred - rho_pred.conj().transpose(-1, -2)).abs().mean()
        
        # Trace
        trace = torch.einsum('...ij,...ji->...', rho_pred, overlap).real
        L_trace = (trace - n_elec).abs().mean()
        
        # Idempotency (soft)
        rhoSrho = rho_pred @ overlap @ rho_pred
        L_idem = (rhoSrho - rho_pred).abs().mean()
        
        total = (L_recon + 
                 self.lambdas['grad'] * L_grad +
                 self.lambdas['herm'] * L_herm +
                 self.lambdas['trace'] * L_trace +
                 self.lambdas['idem'] * L_idem)
        
        return total, {
            'recon': L_recon.item(),
            'grad': L_grad.item(),
            'herm': L_herm.item(),
            'trace': L_trace.item(),
            'idem': L_idem.item(),
        }


class ScaledLosses(nn.Module):
    """
    Scaled losses from TDDFTNet - normalize by target magnitude.
    Prevents large targets from dominating training.
    """
    def __init__(self, dipole_weight=0.1):
        super().__init__()
        self.dipole_weight = dipole_weight
        
    def scaled_l2(self, pred, true):
        """Scaled L2: sqrt(Σ(pred-true)² / Σ(true)²)"""
        return torch.sqrt(
            (pred - true).pow(2).sum() / (true.pow(2).sum() + 1e-10)
        )
    
    def scaled_dipole(self, rho_pred, rho_true, positions):
        """
        Scaled dipole error for density matrices.
        dipole = Tr(ρ · r) where r is position operator in AO basis
        """
        # Compute dipole moments
        dipole_pred = torch.einsum('...ij,ijk->...k', rho_pred, positions)
        dipole_true = torch.einsum('...ij,ijk->...k', rho_true, positions)
        
        return (dipole_pred - dipole_true).norm() / (dipole_true.norm() + 1e-10)
    
    def forward(self, rho_pred, rho_true, positions=None):
        l2 = self.scaled_l2(rho_pred, rho_true)
        
        if positions is not None:
            dipole = self.scaled_dipole(rho_pred, rho_true, positions)
            return l2 + self.dipole_weight * dipole
        
        return l2
```

```python
# src/training/curriculum.py
class VarianceCurriculum:
    """Gradually include low-variance elements during training"""
    
    def __init__(self, element_variances, stages=[0.3, 0.6, 1.0]):
        self.variances = element_variances
        self.stages = stages
        
    def get_weights(self, progress):
        """
        Args:
            progress: float in [0, 1] - training progress
        Returns:
            weights: tensor of same shape as density matrix
        """
        if progress < self.stages[0]:
            # Top 50% variance only
            threshold = torch.quantile(self.variances, 0.5)
            mask = self.variances > threshold
        elif progress < self.stages[1]:
            # Top 75%
            threshold = torch.quantile(self.variances, 0.25)
            mask = self.variances > threshold
        else:
            # All elements
            mask = torch.ones_like(self.variances, dtype=torch.bool)
        
        weights = torch.zeros_like(self.variances)
        weights[mask] = 1.0 / (self.variances[mask] + 1e-10)
        weights = weights / weights.sum()  # Normalize
        
        return weights


class HorizonCurriculum:
    """
    Curriculum on prediction horizon (from TDDFTNet).
    Start with short rollouts, gradually increase.
    """
    def __init__(self, stages=[16, 32, 48, 64], epochs_per_stage=10):
        self.stages = stages
        self.epochs_per_stage = epochs_per_stage
        
    def get_horizon(self, epoch):
        stage_idx = min(epoch // self.epochs_per_stage, len(self.stages) - 1)
        return self.stages[stage_idx]


class TemporalBundling:
    """
    Predict multiple timesteps together (from TDDFTNet).
    Improves stability by forcing model to learn consistent multi-step dynamics.
    """
    def __init__(self, bundle_size=2):
        self.bundle_size = bundle_size
        
    def bundle_predictions(self, propagator, x_t, geometry, n_bundles):
        """
        Args:
            propagator: dynamics module
            x_t: initial latent state
            geometry: geometry embedding
            n_bundles: number of bundles to predict
            
        Returns:
            predictions: list of predicted states (length = n_bundles * bundle_size)
        """
        predictions = []
        current = x_t
        
        for _ in range(n_bundles):
            # Predict bundle_size steps at once
            bundle = propagator(current, geometry, n_steps=self.bundle_size)
            predictions.extend(bundle)
            
            # Use LAST prediction as input for next bundle (skip connection)
            current = bundle[-1]
            
        return predictions
```

#### Month 6: Physics Constraints Suite

**Tasks:**
1. Implement hard physics projection layer
2. Add McWeeney purification
3. Benchmark on rungs 1-3

**Key implementation:**
```python
# src/physics/projections.py
class PhysicsProjection(nn.Module):
    """Hard projection to enforce quantum constraints"""
    
    def forward(self, rho_raw, overlap, n_elec, purify=True, n_purify_steps=3):
        # 1. Hermitianize
        rho = 0.5 * (rho_raw + rho_raw.conj().transpose(-1, -2))
        
        # 2. Correct trace
        current_trace = torch.einsum('...ij,...ji->...', rho, overlap).real
        rho = rho * (n_elec / current_trace).unsqueeze(-1).unsqueeze(-1)
        
        # 3. McWeeney purification for idempotency
        if purify:
            for _ in range(n_purify_steps):
                rhoS = rho @ overlap
                rho = 3 * rhoS @ rho - 2 * rhoS @ rho @ overlap @ rho
        
        return rho
```

---

### Phase 3: Generalization (Months 7-9)

#### Month 7: Zero-Shot Evaluation

**Tasks:**
1. Scale to H2O, NH3
2. Implement zero-shot evaluation framework
3. Identify failure modes

#### Month 8: Uncertainty + Re-Anchoring

**Tasks:**
1. Implement ensemble uncertainty
2. Design adaptive re-anchoring triggers
3. Validate calibration

**Key implementation:**
```python
# src/inference/reanchor.py
class AdaptiveReAnchorController:
    def __init__(
        self,
        uncertainty_threshold=0.05,
        trace_threshold=0.02,
        hermiticity_threshold=0.01,
        max_ml_steps=500,
    ):
        self.thresholds = {
            'uncertainty': uncertainty_threshold,
            'trace': trace_threshold,
            'hermiticity': hermiticity_threshold,
        }
        self.max_steps = max_ml_steps
        self.step_counter = 0
        
    def should_reanchor(self, ensemble_preds, overlap, n_elec):
        """
        Returns: (should_anchor, reason, diagnostics)
        """
        self.step_counter += 1
        
        rho_stack = torch.stack(ensemble_preds)
        rho_mean = rho_stack.mean(dim=0)
        
        # Check uncertainty
        uncertainty = rho_stack.std(dim=0).mean().item()
        if uncertainty > self.thresholds['uncertainty']:
            return True, 'HIGH_UNCERTAINTY', {'uncertainty': uncertainty}
        
        # Check trace
        trace = torch.einsum('ij,ji->', rho_mean, overlap).real.item()
        trace_error = abs(trace - n_elec)
        if trace_error > self.thresholds['trace']:
            return True, 'TRACE_VIOLATION', {'trace_error': trace_error}
        
        # Check hermiticity
        herm_error = (rho_mean - rho_mean.conj().T).abs().max().item()
        if herm_error > self.thresholds['hermiticity']:
            return True, 'HERMITICITY_VIOLATION', {'herm_error': herm_error}
        
        # Check max steps
        if self.step_counter >= self.max_steps:
            return True, 'MAX_STEPS', {'steps': self.step_counter}
        
        return False, 'CONTINUE', {}
    
    def reset(self):
        self.step_counter = 0
```

#### Month 9: Meta-Learning

**Tasks:**
1. Implement MAML-style adaptation
2. Demonstrate few-shot (5-10 trajectories) is sufficient
3. Prepare for production

---

### Phase 4: Production (Months 10-12)

#### Month 10: NWChem Integration

**Tasks:**
1. Build NWChem ↔ Python interface
2. Implement warm-start SCF from ML predictions
3. Validate end-to-end speedup

#### Month 11: Benzene Stress Test

**Tasks:**
1. Validate on C6H6 (66 basis functions)
2. Optimize inference (TensorRT, mixed precision)
3. Profile bottlenecks

#### Month 12: Documentation & Release

**Tasks:**
1. Comprehensive benchmarks
2. Documentation
3. Open-source release

---

## 8. Core Components

### 7.1 Full Model Architecture

```python
# src/models/full_model.py
class RTTDDFTModel(nn.Module):
    """End-to-end model for RT-TDDFT acceleration"""
    
    def __init__(self, config):
        super().__init__()
        
        # Geometry encoder (E3NN)
        self.geometry_encoder = GeometryEncoder(
            irreps_out=config.geometry_irreps,
            num_layers=config.geometry_layers,
        )
        
        # Density encoder (variable basis)
        self.density_encoder = DensityEdgeProjection(
            latent_dim=config.latent_dim,
            max_l=config.max_l,
        )
        
        # Field encoder (equivariant)
        self.field_encoder = EquivariantFieldEncoder(
            geometry_irreps=config.geometry_irreps,
        )
        
        # Dynamics (Mamba + FiLM)
        self.dynamics = GeometryConditionedMamba(
            d_model=config.d_model,
            d_state=config.d_state,
            n_layers=config.n_layers,
            geometry_dim=config.geometry_dim,
        )
        
        # Decoder
        self.decoder = DensityDecoder(
            latent_dim=config.latent_dim,
            max_l=config.max_l,
        )
        
        # Physics projection
        self.physics_proj = PhysicsProjection()
        
    def forward(self, batch, hidden_states=None):
        # Encode geometry (static, compute once per trajectory)
        geometry_features = self.geometry_encoder(
            batch.positions, batch.atomic_numbers, batch.edge_index
        )
        geometry_embedding = geometry_features.mean(dim=1)  # Pool to (batch, dim)
        
        # Encode current density
        density_latent = self.density_encoder(
            batch.density, geometry_features, batch.basis_metadata
        )
        
        # Encode field
        field_features = self.field_encoder(geometry_features, batch.field)
        
        # Combine inputs
        x = density_latent + field_features.mean(dim=1)  # (batch, latent)
        x = x.unsqueeze(1)  # (batch, 1, latent) - single timestep
        
        # Dynamics
        x, new_hidden = self.dynamics(x, geometry_embedding, hidden_states)
        
        # Decode
        rho_pred = self.decoder(x.squeeze(1), geometry_features, batch.basis_metadata)
        
        # Physics projection
        rho_pred = self.physics_proj(rho_pred, batch.overlap, batch.n_electrons)
        
        return rho_pred, new_hidden
    
    def rollout(self, initial_density, geometry, field_sequence, n_steps):
        """Autoregressive rollout for inference"""
        predictions = [initial_density]
        rho = initial_density
        hidden = None
        
        for t in range(n_steps):
            batch = self._make_batch(rho, geometry, field_sequence[t])
            rho, hidden = self.forward(batch, hidden)
            predictions.append(rho)
        
        return torch.stack(predictions)
```

### 7.2 Configuration Schema

```yaml
# configs/model/default.yaml
model:
  # Geometry encoder
  geometry_irreps: "32x0e + 16x1o + 8x2e"
  geometry_layers: 4
  max_radius: 5.0
  
  # Latent space
  latent_dim: 256
  max_l: 2
  
  # Dynamics
  d_model: 256
  d_state: 16
  n_layers: 6
  geometry_dim: 128
  
training:
  # Optimizer
  lr: 3e-4
  weight_decay: 0.01
  
  # Loss weights
  lambda_grad: 10.0
  lambda_herm: 1.0
  lambda_trace: 5.0
  lambda_idem: 0.5
  
  # Curriculum
  variance_stages: [0.3, 0.6, 1.0]
  
  # Scheduling
  warmup_epochs: 10
  total_epochs: 100
  
data:
  sequence_length: 256
  batch_size: 32
  dt: 0.2
```

---

## 9. Testing & Validation

### 8.1 Unit Tests

```python
# tests/test_equivariance.py

def test_geometry_encoder_equivariance():
    """E3NN encoder must be SO(3) equivariant"""
    # ... (see Month 1)

def test_field_encoder_equivariance():
    """Field coupling must preserve selection rules"""
    model = EquivariantFieldEncoder()
    
    # X-polarized field on x-aligned molecule
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    field = torch.tensor([1., 0., 0.])
    
    out_x = model(pos, field)
    
    # Rotate 90° around z → y-polarized on y-aligned
    R = rotation_matrix_z(np.pi/2)
    pos_rot = pos @ R.T
    field_rot = field @ R.T
    
    out_y = model(pos_rot, field_rot)
    
    # L=1 components should rotate correspondingly
    # ... verify Wigner-D transformation


def test_density_encoder_variable_basis():
    """Encoder must handle different basis sizes"""
    model = DensityEdgeProjection()
    
    # H2 (4 basis)
    rho_h2 = torch.randn(4, 4) + 1j * torch.randn(4, 4)
    out_h2 = model(rho_h2, ...)
    
    # H2O (13 basis)
    rho_h2o = torch.randn(13, 13) + 1j * torch.randn(13, 13)
    out_h2o = model(rho_h2o, ...)
    
    # Same output shape regardless of input
    assert out_h2.shape == out_h2o.shape
```

### 8.2 Physics Tests

```python
# tests/test_physics.py

def test_hermiticity_projection():
    """Projected density must be Hermitian"""
    proj = PhysicsProjection()
    rho_raw = torch.randn(10, 10) + 1j * torch.randn(10, 10)
    rho_proj = proj(rho_raw, overlap, n_elec)
    
    assert torch.allclose(rho_proj, rho_proj.conj().T)


def test_trace_projection():
    """Projected density must have correct trace"""
    proj = PhysicsProjection()
    n_elec = 10
    rho_proj = proj(rho_raw, overlap, n_elec)
    
    trace = torch.einsum('ij,ji->', rho_proj, overlap).real
    assert torch.allclose(trace, torch.tensor(n_elec, dtype=torch.float))


def test_derivative_loss_phase_coherence():
    """Models with derivative loss should maintain phase"""
    # Train two models: with/without L_grad
    # Compare phase correlation after 1000 steps
    # ... 
```

### 8.3 Integration Tests

```python
# tests/test_integration.py

def test_full_model_forward():
    """Full model forward pass without errors"""
    model = RTTDDFTModel(config)
    batch = create_dummy_batch(n_atoms=3, n_basis=13)
    
    rho_pred, hidden = model(batch)
    
    assert rho_pred.shape == batch.density.shape
    assert not torch.isnan(rho_pred).any()


def test_rollout_stability():
    """Rollout should not diverge for 1000 steps"""
    model = load_trained_model()
    trajectory = load_test_trajectory()
    
    predictions = model.rollout(
        trajectory.density[0],
        trajectory.geometry,
        trajectory.field[:1000],
        n_steps=1000
    )
    
    # Check no NaN/Inf
    assert torch.isfinite(predictions).all()
    
    # Check physics constraints
    for rho in predictions:
        assert check_hermiticity(rho) < 0.01
        assert check_trace_error(rho) < 0.05
```

---

## 10. References

### Papers

**ML for TDDFT:**
1. **TDDFTNet:** Anonymous, "Learning Time-Dependent Density Functional Theory via Geometry and Physics Aware Latent Evolution" ICLR (2025, under review)
   - Key insights: Latent evolution, physics-aware readout, curriculum training

**Equivariant Neural Networks:**
2. **NequIP:** Batzner et al., "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials" (2022)
3. **MACE:** Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks" (2022)
4. **Allegro:** Musaelian et al., "Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics" (2023)

**State Space Models:**
5. **Mamba:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
6. **S4:** Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" (2022)

**RT-TDDFT Methods:**
7. **RT-TDDFT Review:** Lopata & Govind, "Modeling Fast Electron Dynamics with Real-Time TDDFT" JCTC (2011)
8. **Octopus:** Tancogne-Dejean et al., "Octopus, a computational framework for exploring light-driven phenomena" JCP (2020)

**Neural Network Techniques:**
9. **FiLM:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
10. **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

**ML for Quantum Chemistry:**
11. **DeepDFT:** Jørgensen & Bhowmik, "DeepDFT: Neural Message Passing Network for Accurate Charge Density Prediction" (2020)
12. **QHNet:** Yu et al., "Efficient and equivariant graph networks for predicting quantum Hamiltonian" ICML (2023)

### Code References

- E3NN: https://github.com/e3nn/e3nn
- Mamba: https://github.com/state-spaces/mamba
- NWChem: https://github.com/nwchemgit/nwchem
- Octopus: https://octopus-code.org/
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric

### Documentation

- E3NN tutorials: https://e3nn.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- NWChem manual: https://nwchemgit.github.io/

---

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate rt_tddft_ml

# 2. Generate training data (requires NWChem)
python scripts/generate_data.py --molecule h2p --output data/raw/

# 3. Convert to HDF5
python scripts/convert_nwchem.py --input data/raw/ --output data/processed/

# 4. Train Phase 1 model
python scripts/train.py --config configs/training/phase1_h2p.yaml

# 5. Evaluate
python scripts/evaluate.py --checkpoint outputs/phase1/best.pt --test_data data/processed/h2p_test/
```

---

*Last updated: January 2025*
*Contact: RT-TDDFT ML Accelerator Team*
