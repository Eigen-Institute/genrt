# RT-TDDFT Universal Accelerator: 12-Month Research Roadmap

**Principal Research Scientist Execution Plan**  
**Program Duration:** Months 1-12  
**Core Goal:** Break the "fixed-system" constraint via E3NN + Mamba architecture

---

## 0. Differential Diagnosis: Neural ODE vs. Discrete-Step Mamba

Before committing to an architecture for the latent dynamics module, we must rigorously evaluate two competing paradigms for learning temporal evolution. This decision has profound implications for long-term rollout stability—the central technical challenge of this program.

### 0.1 Candidate Architectures

| Aspect | Continuous-Time Neural ODE | Discrete-Step Mamba SSM |
|--------|---------------------------|-------------------------|
| **Formulation** | dz/dt = f_θ(z, t, g) | z_{t+1} = Mamba(z_t, g) + ResNet(z_t) |
| **Time handling** | Arbitrary dt via ODE solver | Fixed dt, learned transition |
| **Memory** | O(1) via adjoint method | O(L) hidden state, but efficient |
| **Physics alignment** | Direct analog to Liouville-von Neumann | Approximates propagator U(dt) |

### 0.2 Critical Analysis

**Neural ODE Strengths:**
- Mathematically elegant: iℏ ∂ρ/∂t = [H,ρ] is naturally continuous
- Variable timestep handling without retraining
- Guaranteed invertibility (for adjoint training)
- Energy conservation can be baked into Hamiltonian formulation

**Neural ODE Weaknesses (Critical for RT-TDDFT):**
1. **Solver overhead:** Each forward pass requires multiple RHS evaluations (typically 20-100 for dopri5). For 10^5 step rollouts, this becomes 2×10^6 - 10^7 neural network calls.
2. **Stiff dynamics:** RT-TDDFT exhibits multi-scale oscillations (electronic ~0.1 fs, nuclear ~10 fs). Adaptive solvers struggle with stiffness, often rejecting steps.
3. **Error accumulation pattern:** ODE solvers compound local truncation error. For oscillatory dynamics, phase errors grow as O(T^{p+1}) where p is solver order, but the *amplitude* of phase error grows unboundedly.
4. **No explicit memory:** Neural ODE has no mechanism to "remember" distant past beyond what's encoded in z(t). Quantum coherences require long-range temporal correlations.

**Empirical evidence from document:** The Gemini notes indicate "phase loss" was observed in ResNet rollouts. Neural ODEs would exacerbate this because they lack explicit memory structures.

**Discrete Mamba Strengths:**
1. **Selective state space:** The gating mechanism learns *which* history to retain—critical for coherent quantum oscillations
2. **O(n) complexity:** Linear scaling in sequence length vs O(n²) for attention
3. **Hardware efficiency:** CUDA-optimized scan operations; existing validated implementation
4. **Implicit multi-scale:** Hidden state naturally captures dynamics at multiple timescales without explicit solver tuning

**Discrete Mamba Weaknesses:**
1. **Fixed dt assumption:** Must retrain or use interpolation for different timesteps
2. **No formal conservation guarantees:** Energy/norm drift possible
3. **Initialization sensitivity:** Hidden state h_0 must be carefully set for new molecules

### 0.3 Long-Term Rollout Stability Analysis

**Test scenario:** 10,000 step rollout (typical RT-TDDFT simulation ≈ 2000 a.u. with dt=0.2)

| Metric | Neural ODE (predicted) | Mamba (from existing data) |
|--------|----------------------|---------------------------|
| Phase coherence at t=1000 | Poor (no memory) | Good (SSM memory) |
| Amplitude stability | Moderate (Hamiltonian helps) | Good (ResNet residual) |
| Compute per rollout | ~500k FLOPs/step | ~50k FLOPs/step |
| Re-anchoring frequency needed | Every ~100 steps | Every ~500-1000 steps |

### 0.4 Verdict: Discrete-Step Mamba with Physics-Informed Modifications

**Winner: Mamba SSM**

**Justification:**
1. **Memory is paramount:** Quantum coherences in RT-TDDFT can persist for thousands of femtoseconds. The selective state space in Mamba provides learnable long-range memory that Neural ODEs fundamentally lack.

2. **Practical efficiency:** A 10× reduction in compute per step translates to exploring more architectures during development and faster inference in production.

3. **Existing validation:** The document confirms Mamba already works for H2+ dynamics. We're extending proven technology, not gambling on unvalidated approaches.

4. **Mitigatable weaknesses:** 
   - Fixed dt → Use dt as a conditioning signal (embed dt and inject via FiLM)
   - No conservation → Add physics projection layer post-Mamba (already in architecture)
   - Initialization → Learn h_0 = f(geometry_embedding)

**Recommended hybrid enhancement:** Add a *Hamiltonian-inspired residual* to the Mamba output:
```python
# Standard Mamba step
z_mamba = mamba_block(z_t, h_t, geometry_cond)

# Physics residual: approximate [H, ρ] structure
z_physics = antisymmetric_layer(z_mamba, field_t)  # Learns commutator-like structure

# Combined update
z_{t+1} = z_t + dt * (z_mamba + λ * z_physics)
```

This preserves Mamba's memory advantages while nudging dynamics toward physical structure.

---

## 1. Executive Timeline (12 Months)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 1: FOUNDATION (Months 1-3)                                            ║
║  ├─ M1: Data infrastructure + E3NN prototype                                 ║
║  ├─ M2: Geometry-conditioned Mamba on H2+ variants                          ║
║  └─ M3: Variable basis attention mechanism                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PHASE 2: CROSS-GEOMETRY (Months 4-6)                                        ║
║  ├─ M4: Block-diagonal density encoder with universal projector             ║
║  ├─ M5: LiH integration + multi-molecule training                           ║
║  └─ M6: Variance curriculum + derivative loss implementation                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PHASE 3: GENERALIZATION (Months 7-9)                                        ║
║  ├─ M7: H2O/NH3 ladder + zero-shot evaluation framework                     ║
║  ├─ M8: Uncertainty quantification + adaptive re-anchoring                  ║
║  └─ M9: Meta-learning for few-shot molecular adaptation                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PHASE 4: PRODUCTION (Months 10-12)                                          ║
║  ├─ M10: NWChem integration + warm-start protocol                           ║
║  ├─ M11: Benzene stress test + performance optimization                     ║
║  └─ M12: Documentation, benchmarking, release preparation                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Data Strategy: The Molecule Ladder

### 2.1 Ladder Definition with Trajectory Requirements

| Rung | Molecule | nbf | Atoms | Geometries | Trajectories/Geo | Field Types | Total Trajectories |
|------|----------|-----|-------|------------|-----------------|-------------|-------------------|
| 1 | H2+ | 4 | 2 | 5 | 20 | δ-kick (3), Gaussian (2) | 100 |
| 2 | H2 | 4 | 2 | 5 | 20 | δ-kick (3), Gaussian (2) | 100 |
| 3 | LiH | 11 | 2 | 7 | 15 | δ-kick (3), Gaussian (2), Resonant (2) | 105 |
| 4 | H2O | 13 | 3 | 10 | 12 | δ-kick (2), Gaussian (3), Resonant (3) | 120 |
| 5 | NH3 | 15 | 4 | 8 | 12 | δ-kick (2), Gaussian (3), Resonant (3) | 96 |
| 6 | CH4 | 17 | 5 | 6 | 10 | δ-kick (2), Gaussian (2), Resonant (2) | 60 |
| 7 | C2H4 | 26 | 6 | 5 | 10 | δ-kick (1), Gaussian (2), Resonant (2) | 50 |
| 8 | C6H6 | 66 | 12 | 3 | 8 | Gaussian (2), Resonant (2) | 24 |

**Total: ~655 trajectories, ~3.3 billion density matrix snapshots (assuming 5000 steps each)**

### 2.2 Geometry Sampling Strategy

For each molecule, sample geometries along:
1. **Equilibrium:** DFT-optimized structure
2. **Bond stretches:** ±10%, ±20% of equilibrium bond lengths
3. **Angle bends:** ±5°, ±10° for molecules with angles
4. **Normal mode displacements:** Along 2-3 lowest frequency modes

### 2.3 Field Protocol Specifications

| Field Type | Mathematical Form | Purpose | Parameters |
|------------|-------------------|---------|------------|
| **δ-kick** | E(t) = E₀·δ(t) | Broadband excitation | E₀ ∈ {0.001, 0.005, 0.01} a.u. |
| **Gaussian pulse** | E(t) = E₀·exp(-(t-t₀)²/2σ²)·cos(ωt) | Frequency-selective | ω ∈ {0.1, 0.2, 0.3} a.u., σ = 10 a.u. |
| **Resonant** | E(t) = E₀·sin(ω_res·t) for t ∈ [0, T] | Target specific transitions | ω_res from LR-TDDFT |

### 2.4 Data Generation Compute Estimate

| Molecule | Time/trajectory (CPU-hrs) | Total CPU-hrs |
|----------|---------------------------|---------------|
| H2+, H2 | 0.5 | 100 |
| LiH | 2 | 210 |
| H2O | 8 | 960 |
| NH3 | 12 | 1,152 |
| CH4 | 20 | 1,200 |
| C2H4 | 40 | 2,000 |
| C6H6 | 200 | 4,800 |

**Total DFT compute: ~10,500 CPU-hours ≈ 440 CPU-days**

---

## 3. Phase-by-Phase Deep Dive

### PHASE 1: FOUNDATION (Months 1-3)

#### Month 1: Data Infrastructure + E3NN Prototype

**Technical Objectives:**
- Establish unified data format supporting variable nbf
- Build molecular graph construction pipeline
- Prototype E3NN geometry encoder

**Implementation Sprint:**

```python
# Week 1-2: Data format and loaders
class UnifiedTrajectoryDataset:
    """
    Schema:
    - geometry: (n_atoms, 3) float32 - xyz coordinates
    - atomic_numbers: (n_atoms,) int - Z values  
    - density_series: (n_steps, n_spin, nbf, nbf) complex128
    - overlap: (nbf, nbf) float64
    - field: (n_steps, n_spin, 3) float32
    - basis_metadata: List[BasisFunctionInfo]  # atom_idx, l, m, exponents
    """
    
# Week 3: Molecular graph builder
def build_molecular_graph(geometry, atomic_numbers, cutoff=5.0):
    """
    Nodes: atoms with features [Z_embed, n_basis_on_atom]
    Edges: pairs within cutoff, features [distance, unit_vector]
    """
    
# Week 4: E3NN encoder skeleton
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate

class GeometryEncoder(torch.nn.Module):
    def __init__(self, irreps_out="32x0e + 16x1o + 8x2e"):
        # L=0 (scalar): local chemical environment
        # L=1 (vector): directional/field coupling  
        # L=2 (tensor): d-orbital structure
        self.irreps_out = o3.Irreps(irreps_out)
```

**Verification Step (Chain-of-Verification):**
```
TEST: SO(3) Equivariance of Geometry Encoder

Protocol:
1. Load H2+ geometry: R = [[0,0,0], [0,0,1.4]]
2. Encode → g_original = E3NN(R)
3. Apply random rotation Q ∈ SO(3) to geometry: R' = Q @ R
4. Encode rotated → g_rotated = E3NN(R')
5. Apply Wigner-D rotation to original: g_transformed = D(Q) @ g_original

PASS CRITERION: ||g_rotated - g_transformed|| < 1e-5

Additional check: Verify L=1 components transform as vectors, L=2 as rank-2 tensors
```

**Mitigation Strategy (Risk: Equivariance Breaking):**
- Use only e3nn primitive operations (TensorProduct, Linear)
- Avoid any element-wise nonlinearities that break equivariance
- Add equivariance test to CI/CD pipeline; fail build if violated

---

#### Month 2: Geometry-Conditioned Mamba on H2+ Variants

**Technical Objectives:**
- Extend existing Mamba to accept geometry conditioning
- Validate on H2+ at 5 different bond lengths
- Implement FiLM conditioning mechanism

**Implementation Sprint:**

```python
# Week 1-2: FiLM-conditioned Mamba block
class GeometryConditionedMamba(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation):
    h_out = γ(g) ⊙ Mamba(h_in) + β(g)
    
    Where γ, β are learned functions of geometry embedding g
    """
    def __init__(self, d_model=256, d_state=16, geometry_dim=128):
        self.mamba = Mamba(d_model, d_state)
        self.gamma_net = nn.Sequential(
            nn.Linear(geometry_dim, d_model),
            nn.Sigmoid()  # γ ∈ (0, 1) for stability
        )
        self.beta_net = nn.Linear(geometry_dim, d_model)
        
    def forward(self, x, geometry_embedding):
        gamma = self.gamma_net(geometry_embedding)
        beta = self.beta_net(geometry_embedding)
        mamba_out = self.mamba(x)
        return gamma * mamba_out + beta
```

**FiLM vs Cross-Attention Decision:**

| Mechanism | Compute | Expressivity | Stability |
|-----------|---------|--------------|-----------|
| FiLM | O(d) | Multiplicative modulation | High (bounded γ) |
| Cross-Attention | O(d²) | Full mixing | Lower (softmax saturation) |

**Choice: Start with FiLM, add cross-attention in Phase 2 if needed**

Rationale: Geometry is *static* throughout a trajectory. We don't need the full expressivity of attention to condition on a fixed vector. FiLM is 10× cheaper and more stable.

**Week 3-4: H2+ bond length sweep**
```
Geometries: r ∈ {1.0, 1.2, 1.4, 1.6, 2.0} Bohr
Training: 4 geometries (80 trajectories)
Holdout: 1 geometry (20 trajectories) for generalization test
```

**Verification Step:**
```
TEST: Geometry Generalization on H2+

Protocol:
1. Train on r ∈ {1.0, 1.2, 1.4, 2.0}
2. Evaluate on held-out r = 1.6
3. Compare MSE to baseline (no geometry conditioning)

PASS CRITERION: 
- Held-out MSE < 0.1 (vs 0.4 reported for fixed-system model)
- Rollout stable for 1000 steps without re-anchoring

Physics check: Verify transition frequencies shift correctly with bond length
(ω ∝ 1/r² for particle-in-a-box approximation)
```

**Mitigation Strategy (Risk: Distribution Shift):**
- Implement geometry interpolation augmentation: randomly sample r' ∈ [r_i, r_j] between training geometries
- Use spherical linear interpolation (slerp) for orientation-diverse augmentation

---

#### Month 3: Variable Basis Attention Mechanism

**Technical Objectives:**
- Solve the "Variable Block Size" problem
- Implement attention-based basis-to-latent projection
- Validate on LiH (nbf=11, first multi-electron system)

**Implementation Sprint: Universal Block Encoder**

**The Variable Block Size Problem:**
Density matrix ρ has shape (nbf, nbf) where nbf varies. Element pairs (e.g., C-H vs C-C) contribute blocks of different sizes:
- H-H block: (1, 1) for STO-3G
- C-H block: (5, 1) 
- C-C block: (5, 5)

**Solution: Hierarchical Attention with Element-Type Embeddings**

```python
class UniversalBlockEncoder(nn.Module):
    """
    Strategy: Don't enumerate all element pairs explicitly.
    Instead, use element embeddings and attention to handle any combination.
    
    Block encoding: ρ_AB → z_AB
    where A, B are atoms with elements Z_A, Z_B
    """
    def __init__(self, max_l=2, latent_dim=64):
        # Element embedding: Z → embedding
        self.element_embed = nn.Embedding(118, 32)  # Full periodic table
        
        # Angular momentum embedding: l → embedding  
        self.angular_embed = nn.Embedding(max_l + 1, 16)  # s, p, d
        
        # Per-orbital encoder (shared across all elements)
        self.orbital_encoder = nn.Sequential(
            nn.Linear(32 + 16 + 2, 64),  # element + angular + (real, imag)
            nn.SiLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Cross-attention to aggregate variable-size blocks
        self.block_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Learnable query tokens (fixed size output regardless of input)
        self.query_tokens = nn.Parameter(torch.randn(16, latent_dim))
        
    def encode_block(self, rho_block, Z_A, Z_B, basis_info_A, basis_info_B):
        """
        Encode a single atom-pair block of the density matrix.
        
        rho_block: (n_basis_A, n_basis_B) complex
        Returns: (latent_dim,) fixed-size representation
        """
        # Embed each basis function
        orbitals_A = []
        for bf in basis_info_A:
            elem_emb = self.element_embed(Z_A)
            ang_emb = self.angular_embed(bf.l)
            orbitals_A.append(torch.cat([elem_emb, ang_emb], dim=-1))
        orbitals_A = torch.stack(orbitals_A)  # (n_basis_A, 48)
        
        orbitals_B = []  # Similar for B
        
        # Encode each matrix element with its orbital context
        flat_elements = []
        for i, orb_a in enumerate(orbitals_A):
            for j, orb_b in enumerate(orbitals_B):
                rho_ij = rho_block[i, j]
                element_features = torch.cat([
                    orb_a, orb_b,
                    torch.tensor([rho_ij.real, rho_ij.imag])
                ])
                encoded = self.orbital_encoder(element_features)
                flat_elements.append(encoded)
        
        # Attention pooling to fixed size
        elements = torch.stack(flat_elements).unsqueeze(0)  # (1, n_A*n_B, latent)
        queries = self.query_tokens.unsqueeze(0)  # (1, 16, latent)
        
        pooled, _ = self.block_attention(queries, elements, elements)
        return pooled.mean(dim=1).squeeze(0)  # (latent_dim,)
```

**Key insight:** By embedding elements and angular momentum separately, we avoid the combinatorial explosion of element-pair-specific encoders. A C-H block and a N-O block share the same encoder, just with different input embeddings.

**Week 3-4: LiH integration**
```
LiH specifics:
- 3 electrons (2 alpha, 1 beta OR 1.5, 1.5 for restricted)
- nbf = 11 (Li: 5 basis functions, H: 6 in typical basis)
- Block structure: Li-Li (5×5), Li-H (5×6), H-H (6×6)
```

**Verification Step:**
```
TEST: Variable Basis Handling

Protocol:
1. Train encoder on H2+ (nbf=4) + LiH (nbf=11) jointly
2. Verify latent space distances:
   - Same molecule, different geometries → small distance
   - Different molecules, similar fields → medium distance
   - Different molecules, different fields → large distance

PASS CRITERION:
- Reconstruction MSE < 0.05 for both molecules
- No NaN/Inf in gradients during training
- Latent space shows meaningful clustering (visualize with t-SNE)

Physics check: Li-H bond dipole moment should be recoverable from latent space
```

**Mitigation Strategy (Risk: Basis Set Diversity):**
- Train with dropout on orbital features (p=0.1) to prevent overfitting to specific basis layouts
- Add basis-set augmentation: randomly mask 10% of basis functions during training

---

### PHASE 2: CROSS-GEOMETRY (Months 4-6)

#### Month 4: Block-Diagonal Density Encoder with GNN Message Passing

**Technical Objectives:**
- Implement full density matrix encoder using GNN edge updates
- Add equivariant field encoding (E(t) as L=1 spherical tensor)
- Achieve cross-geometry transfer within single molecule species

**Implementation Sprint:**

```python
class EquivariantDensityEncoder(nn.Module):
    """
    Treats ρ_μν as edge attributes in a molecular graph.
    Edges connect atoms whose basis functions overlap.
    
    Uses E3NN message passing to update edge features equivariantly.
    """
    def __init__(self, irreps_node="32x0e + 16x1o", irreps_edge="16x0e + 8x1o"):
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        
        # Block encoder from Month 3
        self.block_encoder = UniversalBlockEncoder()
        
        # E3NN convolution for message passing
        self.conv = e3nn.nn.models.gate_points_2101.Convolution(
            irreps_node_input=self.irreps_node,
            irreps_node_output=self.irreps_node,
            irreps_edge_attr="1x0e + 1x1o",  # distance scalar + unit vector
            num_neighbors=4
        )
        
        # Field encoder: Cartesian E(t) → spherical L=1
        self.field_encoder = CartesianToSphericalField()
        
    def forward(self, rho, geometry, field_t, graph):
        """
        rho: (nbf, nbf) complex density matrix
        geometry: (n_atoms, 3) positions
        field_t: (3,) electric field vector
        graph: Molecular graph with edges
        """
        # Step 1: Encode each atom-pair block
        edge_latents = []
        for edge_idx, (i, j) in enumerate(graph.edge_index.T):
            block = extract_block(rho, i, j, graph.basis_map)
            z_ij = self.block_encoder.encode_block(
                block, 
                graph.atomic_numbers[i],
                graph.atomic_numbers[j],
                graph.basis_info[i],
                graph.basis_info[j]
            )
            edge_latents.append(z_ij)
        
        edge_attr = torch.stack(edge_latents)
        
        # Step 2: Encode field as L=1 spherical tensor
        field_spherical = self.field_encoder(field_t)  # (3,) → o3.Irreps("1x1o")
        
        # Step 3: E3NN message passing (geometry-aware)
        node_features = self.conv(
            graph.node_features, 
            graph.edge_index,
            edge_attr,
            graph.edge_vec  # Unit vectors for equivariance
        )
        
        # Step 4: Combine with field via equivariant tensor product
        # L=0 density × L=1 field → L=1 response (dipole-like)
        node_with_field = self.field_coupling(node_features, field_spherical)
        
        return node_with_field, edge_attr


class CartesianToSphericalField(nn.Module):
    """Convert E(t) = (Ex, Ey, Ez) to spherical harmonics Y^1_m"""
    def forward(self, field_cartesian):
        # Standard transformation (see e3nn docs)
        Ex, Ey, Ez = field_cartesian
        Y1_m1 = (Ex - 1j * Ey) / np.sqrt(2)
        Y1_0 = Ez
        Y1_p1 = -(Ex + 1j * Ey) / np.sqrt(2)
        return torch.stack([Y1_m1, Y1_0, Y1_p1])
```

**Verification Step:**
```
TEST: Field Equivariance

Protocol:
1. Set E = (1, 0, 0) a.u. (x-polarized)
2. Compute dipole response μ(t) for H2 aligned along x
3. Rotate system by 90° → H2 along y, E along y
4. Verify μ_y(rotated) = μ_x(original) within tolerance

PASS CRITERION: ||μ_rotated - D(R)·μ_original|| < 1e-4

This confirms selection rules are preserved by the model.
```

---

#### Month 5: Multi-Molecule Training + LiH Full Integration

**Technical Objectives:**
- Joint training on H2+, H2, LiH
- Implement variance curriculum (Stage 1.5 from document)
- Validate cross-molecule feature sharing

**Implementation Sprint:**

```python
# Variance-based element weighting
class VarianceAwareLoss(nn.Module):
    """
    L = Σ_ij w_ij |ρ_ij - ρ̂_ij|²
    where w_ij ∝ 1/Var(ρ_ij) across training set
    
    Prevents model from ignoring low-variance (but physically important) elements.
    """
    def __init__(self, variance_floor=1e-10):
        self.variance_floor = variance_floor
        self.element_variances = None  # Computed from training data
        
    def compute_weights(self, dataset):
        """Pre-compute variance for each matrix element position"""
        # For variable-sized matrices, compute per-block variances
        variances = defaultdict(list)
        for traj in dataset:
            for block_key, block in traj.blocks.items():
                variances[block_key].append(block.var(dim=0))  # Temporal variance
        
        self.element_weights = {}
        for key, var_list in variances.items():
            mean_var = torch.stack(var_list).mean(dim=0)
            self.element_weights[key] = 1.0 / (mean_var + self.variance_floor)
            
    def forward(self, rho_pred, rho_true, block_keys):
        loss = 0
        for key in block_keys:
            w = self.element_weights[key]
            diff = (rho_pred[key] - rho_true[key]).abs() ** 2
            loss += (w * diff).sum()
        return loss


# Curriculum training schedule
class VarianceCurriculum:
    """
    Stage 1: Train on high-variance elements only (top 50% by variance)
    Stage 1.5: Gradually include medium-variance elements
    Stage 2: All elements with full weighting
    """
    def __init__(self, total_epochs=100):
        self.stage_boundaries = [0.3, 0.6, 1.0]  # fraction of training
        
    def get_active_mask(self, epoch, total_epochs, element_variances):
        progress = epoch / total_epochs
        
        if progress < 0.3:
            # Top 50% by variance
            threshold = np.percentile(element_variances, 50)
            return element_variances > threshold
        elif progress < 0.6:
            # Top 75% by variance  
            threshold = np.percentile(element_variances, 25)
            return element_variances > threshold
        else:
            # All elements
            return np.ones_like(element_variances, dtype=bool)
```

**Verification Step:**
```
TEST: Variance Curriculum Prevents Mean Collapse

Protocol:
1. Train model WITHOUT variance curriculum on oscillatory H2+ data
2. Train model WITH variance curriculum on same data
3. Compare off-diagonal element predictions after 1000 rollout steps

PASS CRITERION (with curriculum):
- Off-diagonal elements maintain oscillation amplitude > 50% of ground truth
- Phase correlation with ground truth > 0.8

FAIL CRITERION (without curriculum):
- Off-diagonal elements collapse to mean (oscillation < 10% of ground truth)
```

---

#### Month 6: Derivative Loss + Full Physics Constraint Suite

**Technical Objectives:**
- Implement derivative loss L_grad = ||dρ/dt_pred - dρ/dt_true||²
- Complete physics projection layer (Hermitian + trace + positive)
- Benchmark against baselines on Molecule Ladder rungs 1-3

**Implementation Sprint:**

```python
class PhysicsAwareLoss(nn.Module):
    """
    Complete loss function:
    L = L_recon + λ_grad·L_grad + λ_herm·L_herm + λ_trace·L_trace + λ_idem·L_idem
    """
    def __init__(self, dt, n_electrons, overlap_matrix):
        self.dt = dt
        self.n_elec = n_electrons
        self.S = overlap_matrix
        
        # Loss weights (tuned empirically)
        self.lambda_grad = 10.0   # Derivative matching is critical
        self.lambda_herm = 1.0
        self.lambda_trace = 5.0
        self.lambda_idem = 0.5   # Soft constraint, not hard
        
    def forward(self, rho_pred, rho_true, rho_prev_true):
        # Reconstruction loss
        L_recon = F.mse_loss(rho_pred.real, rho_true.real) + \
                  F.mse_loss(rho_pred.imag, rho_true.imag)
        
        # Derivative loss: compare predicted rate of change
        drho_pred = (rho_pred - rho_prev_true) / self.dt  # Model's implied derivative
        drho_true = (rho_true - rho_prev_true) / self.dt   # Actual derivative
        L_grad = F.mse_loss(drho_pred, drho_true)
        
        # Hermiticity: ρ should equal ρ†
        L_herm = (rho_pred - rho_pred.conj().transpose(-1, -2)).abs().mean()
        
        # Trace: Tr(ρS) = N_electrons
        trace_pred = torch.einsum('...ij,...ji->...', rho_pred, self.S).real
        L_trace = (trace_pred - self.n_elec).abs().mean()
        
        # Idempotency (soft): ρSρ ≈ ρ for pure states
        rhoSrho = rho_pred @ self.S @ rho_pred
        L_idem = (rhoSrho - rho_pred).abs().mean()
        
        total = L_recon + self.lambda_grad * L_grad + \
                self.lambda_herm * L_herm + self.lambda_trace * L_trace + \
                self.lambda_idem * L_idem
                
        return total, {
            'recon': L_recon.item(),
            'grad': L_grad.item(),
            'herm': L_herm.item(),
            'trace': L_trace.item(),
            'idem': L_idem.item()
        }


class PhysicsProjection(nn.Module):
    """Hard projection to enforce constraints post-prediction"""
    def forward(self, rho_raw, S, n_elec, purify=True):
        # 1. Hermitianize
        rho = 0.5 * (rho_raw + rho_raw.conj().transpose(-1, -2))
        
        # 2. Scale to correct trace
        current_trace = torch.einsum('ij,ji->', rho, S).real
        rho = rho * (n_elec / current_trace)
        
        # 3. Optional: McWeeney purification for idempotency
        if purify:
            for _ in range(3):  # 3 iterations usually sufficient
                rhoS = rho @ S
                rho = 3 * rhoS @ rho - 2 * rhoS @ rho @ S @ rho
                
        return rho
```

**Verification Step:**
```
TEST: Derivative Loss Maintains Phase Coherence

Protocol:
1. Train identical models with/without L_grad on H2O
2. Evaluate phase of off-diagonal element ρ_01(t) over 2000 steps
3. Compute phase correlation: C = |⟨exp(i·φ_pred)·exp(-i·φ_true)⟩|

PASS CRITERION (with L_grad): Phase correlation C > 0.9 at t=2000
FAIL CRITERION (without L_grad): Phase correlation C < 0.5 (phase drift)
```

---

### PHASE 3: GENERALIZATION (Months 7-9)

#### Month 7: H2O/NH3 Ladder + Zero-Shot Framework

**Technical Objectives:**
- Scale to 3-4 atom molecules
- Establish zero-shot evaluation protocol
- Identify failure modes for unseen molecules

**Implementation Sprint:**

```python
class ZeroShotEvaluator:
    """
    Evaluate model on molecules not seen during training.
    
    Protocol:
    1. Train on {H2+, H2, LiH}
    2. Evaluate on {H2O, NH3} without any fine-tuning
    3. Measure MSE, physics violations, rollout horizon
    """
    def __init__(self, model, physics_checker):
        self.model = model
        self.physics_checker = physics_checker
        
    def evaluate_zero_shot(self, test_molecule_dataset):
        results = {
            'mse_1step': [],
            'mse_100step': [],
            'mse_1000step': [],
            'physics_violations': [],
            'valid_horizon': []  # Steps until 10% error exceeded
        }
        
        for trajectory in test_molecule_dataset:
            # Single-step accuracy
            rho_pred_1 = self.model.predict_step(trajectory.rho[0], 
                                                  trajectory.geometry,
                                                  trajectory.field[0])
            results['mse_1step'].append(mse(rho_pred_1, trajectory.rho[1]))
            
            # Rollout accuracy
            rho_rollout = self.model.rollout(trajectory.rho[0],
                                             trajectory.geometry, 
                                             trajectory.field,
                                             n_steps=1000)
            
            results['mse_100step'].append(mse(rho_rollout[100], trajectory.rho[100]))
            results['mse_1000step'].append(mse(rho_rollout[1000], trajectory.rho[1000]))
            
            # Find valid horizon
            errors = [mse(rho_rollout[t], trajectory.rho[t]) for t in range(len(rho_rollout))]
            horizon = next((t for t, e in enumerate(errors) if e > 0.1), len(errors))
            results['valid_horizon'].append(horizon)
            
        return {k: np.mean(v) for k, v in results.items()}
```

**Zero-Shot Performance Targets:**

| Metric | Target (Zero-Shot) | Acceptable |
|--------|-------------------|------------|
| 1-step MSE | < 0.01 | < 0.05 |
| 100-step MSE | < 0.05 | < 0.15 |
| Valid horizon | > 500 steps | > 200 steps |
| Trace error | < 0.01 | < 0.05 |

---

#### Month 8: Uncertainty Quantification + Adaptive Re-Anchoring

**Technical Objectives:**
- Implement ensemble-based uncertainty estimation
- Design adaptive re-anchoring trigger logic
- Validate uncertainty correlates with actual error

**Implementation Sprint: Anchor Trigger Logic**

```python
class AdaptiveReAnchorController:
    """
    Decides when to hand control back to NWChem for re-anchoring.
    
    Trigger conditions (ANY triggers re-anchor):
    1. Uncertainty exceeds threshold
    2. Physics violations exceed threshold  
    3. Maximum ML steps reached
    4. User-defined observable anomaly
    """
    def __init__(self, 
                 uncertainty_threshold=0.05,
                 trace_threshold=0.02,
                 hermiticity_threshold=0.01,
                 max_ml_steps=500,
                 n_ensemble=5):
        
        self.uncertainty_threshold = uncertainty_threshold
        self.trace_threshold = trace_threshold
        self.hermiticity_threshold = hermiticity_threshold
        self.max_ml_steps = max_ml_steps
        self.n_ensemble = n_ensemble
        
        self.ml_step_counter = 0
        
    def should_reanchor(self, ensemble_predictions, S, n_elec):
        """
        Returns: (should_anchor: bool, reason: str, diagnostics: dict)
        """
        self.ml_step_counter += 1
        
        # Stack ensemble predictions: (n_ensemble, nbf, nbf)
        rho_stack = torch.stack(ensemble_predictions)
        rho_mean = rho_stack.mean(dim=0)
        
        # Criterion 1: Ensemble disagreement (epistemic uncertainty)
        uncertainty = rho_stack.std(dim=0).mean().item()
        if uncertainty > self.uncertainty_threshold:
            return True, "HIGH_UNCERTAINTY", {
                'uncertainty': uncertainty,
                'threshold': self.uncertainty_threshold
            }
        
        # Criterion 2: Trace violation
        trace_error = abs(torch.einsum('ij,ji->', rho_mean, S).real.item() - n_elec)
        if trace_error > self.trace_threshold:
            return True, "TRACE_VIOLATION", {
                'trace_error': trace_error,
                'threshold': self.trace_threshold
            }
        
        # Criterion 3: Hermiticity violation
        herm_error = (rho_mean - rho_mean.conj().T).abs().max().item()
        if herm_error > self.hermiticity_threshold:
            return True, "HERMITICITY_VIOLATION", {
                'herm_error': herm_error,
                'threshold': self.hermiticity_threshold
            }
        
        # Criterion 4: Max steps
        if self.ml_step_counter >= self.max_ml_steps:
            return True, "MAX_STEPS_REACHED", {
                'steps': self.ml_step_counter,
                'max': self.max_ml_steps
            }
        
        # All checks passed
        return False, "CONTINUE", {
            'uncertainty': uncertainty,
            'trace_error': trace_error,
            'herm_error': herm_error,
            'steps': self.ml_step_counter
        }
    
    def reset(self):
        """Call after DFT re-anchoring completes"""
        self.ml_step_counter = 0
```

**Verification Step:**
```
TEST: Uncertainty Calibration

Protocol:
1. Run ensemble (N=5) on 100 test trajectories
2. Bin predictions by uncertainty level (low/medium/high)
3. Compute actual MSE within each bin

PASS CRITERION: Monotonic relationship
- Low uncertainty bin: MSE < 0.02
- Medium uncertainty bin: MSE ∈ [0.02, 0.08]
- High uncertainty bin: MSE > 0.08

This confirms uncertainty estimates are meaningful for triggering.
```

---

#### Month 9: Meta-Learning for Few-Shot Adaptation

**Technical Objectives:**
- Implement MAML-style adaptation for new molecules
- Demonstrate 10-trajectory fine-tuning is sufficient
- Prepare model for production deployment

**Implementation Sprint:**

```python
class MAMLAdapter:
    """
    Model-Agnostic Meta-Learning for rapid adaptation to new molecules.
    
    Inner loop: Fine-tune on K trajectories from new molecule
    Outer loop: Optimize for fast adaptation across molecule distribution
    """
    def __init__(self, model, inner_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
    def adapt(self, support_trajectories, query_trajectory=None):
        """
        Few-shot adaptation to new molecule.
        
        support_trajectories: K trajectories from target molecule (K ~ 5-10)
        query_trajectory: Optional held-out trajectory for validation
        """
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.model)
        
        # Inner loop optimization
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for step in range(self.inner_steps):
            total_loss = 0
            for traj in support_trajectories:
                rho_pred = adapted_model.rollout(traj.rho[0], 
                                                  traj.geometry,
                                                  traj.field,
                                                  n_steps=100)
                loss = F.mse_loss(rho_pred, traj.rho[:100])
                total_loss += loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Evaluate on query if provided
        if query_trajectory is not None:
            with torch.no_grad():
                rho_pred = adapted_model.rollout(query_trajectory.rho[0],
                                                  query_trajectory.geometry,
                                                  query_trajectory.field,
                                                  n_steps=1000)
                query_loss = F.mse_loss(rho_pred, query_trajectory.rho[:1000])
            return adapted_model, query_loss.item()
        
        return adapted_model, None
```

**Few-Shot Performance Targets:**

| Shots (K) | Target MSE@1000 | Improvement over Zero-Shot |
|-----------|-----------------|---------------------------|
| 0 (zero-shot) | < 0.10 | Baseline |
| 5 | < 0.05 | 2× |
| 10 | < 0.02 | 5× |
| 20 | < 0.01 | 10× |

---

### PHASE 4: PRODUCTION (Months 10-12)

#### Month 10: NWChem Integration + Warm-Start Protocol

**Technical Objectives:**
- Build bidirectional NWChem ↔ ML interface
- Implement warm-start SCF from ML predictions
- Validate end-to-end acceleration

**Implementation Sprint:**

```python
class NWChemMLHybridDriver:
    """
    Orchestrates ML prediction and DFT re-anchoring.
    
    Communication with NWChem via restart files and Python interface.
    """
    def __init__(self, model, nwchem_interface, anchor_controller):
        self.model = model
        self.nwchem = nwchem_interface
        self.controller = anchor_controller
        
    def run_trajectory(self, geometry, field_protocol, total_steps, dt):
        """
        Run full RT-TDDFT trajectory with ML acceleration.
        """
        results = {
            'density_series': [],
            'ml_steps': 0,
            'dft_steps': 0,
            'anchor_events': []
        }
        
        # Initial DFT calculation
        rho = self.nwchem.compute_ground_state(geometry)
        S = self.nwchem.get_overlap_matrix()
        n_elec = self.nwchem.get_n_electrons()
        
        results['density_series'].append(rho)
        
        for step in range(1, total_steps):
            field_t = field_protocol(step * dt)
            
            # ML prediction (ensemble for uncertainty)
            ensemble_preds = self.model.predict_ensemble(
                rho, geometry, field_t, n_ensemble=5
            )
            
            # Check anchor trigger
            should_anchor, reason, diagnostics = self.controller.should_reanchor(
                ensemble_preds, S, n_elec
            )
            
            if should_anchor:
                # Re-anchor with DFT
                results['anchor_events'].append({
                    'step': step,
                    'reason': reason,
                    'diagnostics': diagnostics
                })
                
                # Warm-start: use ML mean as initial guess
                rho_ml_mean = torch.stack(ensemble_preds).mean(dim=0)
                rho = self.nwchem.propagate_step(
                    rho_current=rho,
                    field=field_t,
                    initial_guess=rho_ml_mean  # Warm start!
                )
                results['dft_steps'] += 1
                self.controller.reset()
            else:
                # Use ML prediction
                rho = torch.stack(ensemble_preds).mean(dim=0)
                rho = physics_project(rho, S, n_elec)  # Enforce constraints
                results['ml_steps'] += 1
            
            results['density_series'].append(rho)
        
        return results


class WarmStartBenchmark:
    """Measure SCF speedup from warm-starting with ML prediction"""
    
    def benchmark(self, geometries, n_trials=50):
        results = {'cold_start': [], 'warm_start': []}
        
        for geo in geometries:
            for trial in range(n_trials):
                # Cold start: atomic guess
                t0 = time.time()
                rho_cold = self.nwchem.compute_ground_state(geo, guess='atomic')
                results['cold_start'].append(time.time() - t0)
                
                # Warm start: ML prediction
                rho_ml = self.model.predict_ground_state(geo)
                t0 = time.time()
                rho_warm = self.nwchem.compute_ground_state(geo, guess=rho_ml)
                results['warm_start'].append(time.time() - t0)
        
        speedup = np.mean(results['cold_start']) / np.mean(results['warm_start'])
        return speedup, results
```

**Verification Step:**
```
TEST: End-to-End Acceleration

Protocol:
1. Run pure DFT trajectory: 5000 steps on H2O (baseline time T_dft)
2. Run ML-accelerated trajectory with adaptive re-anchoring
3. Compare total wall time and trajectory accuracy

PASS CRITERIA:
- Wall time speedup > 10× (target: 50×)
- Max trajectory MSE < 0.05
- Number of re-anchors < 20 (i.e., > 250 ML steps per anchor)
- All observables (dipole moment) within 5% of pure DFT
```

---

#### Month 11: Benzene Stress Test + Performance Optimization

**Technical Objectives:**
- Validate on benzene (C₆H₆, 42 electrons, 66 basis functions)
- Optimize inference speed (CUDA kernels, batching)
- Profile and address bottlenecks

**Performance Targets:**

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Inference time (H2O) | < 1 ms/step | < 0.5 ms/step |
| Inference time (C6H6) | < 10 ms/step | < 5 ms/step |
| Memory (C6H6) | < 4 GB | < 2 GB |
| DFT comparison (C6H6) | ~1 s/step | - |

**Optimization Strategies:**
1. **Fused CUDA kernels** for physics projection layer
2. **TensorRT** compilation of encoder/decoder
3. **Mixed precision** (FP16 for non-critical operations)
4. **Batched ensemble** inference (process all 5 ensemble members simultaneously)

---

#### Month 12: Documentation, Benchmarking, Release

**Technical Objectives:**
- Comprehensive benchmark suite against baselines
- Documentation for external users
- Open-source release preparation

**Deliverables:**
1. **arXiv preprint** with benchmark results
2. **GitHub repository** with trained models + inference code
3. **Docker container** for easy deployment
4. **NWChem plugin** for seamless integration

---

## 4. Hardware & Compute Budget

### 4.1 GPU Requirements (Training)

| Phase | Primary Compute | H100 GPU-Hours | Notes |
|-------|----------------|----------------|-------|
| Phase 1 | E3NN prototyping, H2+ training | 500 | Small molecules, fast iteration |
| Phase 2 | Multi-molecule training, hyperparameter search | 2,000 | LiH scale-up |
| Phase 3 | Full molecule ladder, ensemble training | 5,000 | H2O, NH3, meta-learning |
| Phase 4 | Benzene, optimization | 2,000 | Large-scale validation |
| **Total** | | **9,500** | ~400 GPU-days |

### 4.2 CPU Requirements (DFT Data Generation)

| Molecule Tier | CPU-Hours | Allocation |
|--------------|-----------|------------|
| H2+, H2 | 200 | Phase 1 |
| LiH | 500 | Phase 1-2 |
| H2O, NH3 | 2,500 | Phase 2-3 |
| CH4, C2H4 | 4,000 | Phase 3 |
| C6H6 | 5,000 | Phase 4 |
| **Total** | **12,200** | ~500 CPU-days |

### 4.3 Budget Summary

| Resource | Quantity | Estimated Cost |
|----------|----------|---------------|
| H100 GPU-hours | 9,500 | $28,500 (@ $3/hr cloud) |
| CPU-hours (HPC) | 12,200 | $1,220 (@ $0.10/hr) |
| Storage (trajectories) | 5 TB | $500/year |
| **Total** | | **~$30,000** |

---

## 5. Success Metrics: Three Production KPIs

### KPI 1: Zero-Shot Generalization (Scientific Validity)

**Definition:** Model trained on Molecule Ladder rungs 1-5 achieves acceptable accuracy on unseen molecule (rung 6-7) without fine-tuning.

**Threshold:**
```
PASS: Zero-shot MSE@1000 < 0.10 on C2H4 (not in training set)
      AND valid horizon > 300 steps
      
FAIL: Zero-shot MSE@1000 > 0.20 OR valid horizon < 100 steps
```

**Measurement:** Month 9 evaluation

---

### KPI 2: End-to-End Speedup (Practical Value)

**Definition:** Total wall-clock time reduction for complete RT-TDDFT trajectory including re-anchoring overhead.

**Threshold:**
```
PASS: ≥ 20× speedup on H2O (5000 steps)
      while maintaining trajectory MSE < 0.05

STRETCH: ≥ 50× speedup with MSE < 0.02

FAIL: < 10× speedup OR MSE > 0.10
```

**Measurement:** Month 10-11 benchmarks

---

### KPI 3: Physics Compliance (Trustworthiness)

**Definition:** Model predictions satisfy quantum mechanical constraints without requiring post-hoc correction.

**Threshold:**
```
PASS: After 1000-step rollout (no re-anchoring):
      - Trace error < 1% of N_electrons
      - Hermiticity violation < 0.01
      - No negative eigenvalues with magnitude > 0.001
      
FAIL: Any violation exceeds threshold by 2×
```

**Measurement:** Continuous monitoring throughout Phases 2-4

---

## 6. Risk Registry & Contingency Plans

| Risk ID | Risk | Probability | Impact | Mitigation | Contingency |
|---------|------|-------------|--------|------------|-------------|
| R1 | Equivariance breaking during training | Medium | High | CI/CD equivariance tests | Fall back to invariant features |
| R2 | DFT data generation bottleneck | High | Medium | Prioritize small molecules | Use synthetic tight-binding data |
| R3 | Benzene scale-up fails | Medium | High | Progressive nbf ladder | Cap at CH4/C2H4 for release |
| R4 | Uncertainty uncalibrated | Medium | Medium | Holdout validation | Switch to temperature scaling |
| R5 | NWChem interface issues | Low | High | Early prototyping (M10) | Target PySCF instead |

---

## 7. Monthly Milestones Summary

| Month | Milestone | Verification | Go/No-Go |
|-------|-----------|--------------|----------|
| M1 | E3NN encoder passes equivariance test | SO(3) rotation test | ✓ if ||error|| < 1e-5 |
| M2 | H2+ generalizes across 5 bond lengths | MSE < 0.1 on held-out | ✓ if stable 1000 steps |
| M3 | Variable basis works for LiH | Reconstruction MSE < 0.05 | ✓ if no NaN gradients |
| M4 | Equivariant field coupling validated | Selection rule test | ✓ if dipole response correct |
| M5 | Multi-molecule training converges | Joint loss decreasing | ✓ if no mode collapse |
| M6 | Derivative loss maintains phase | Phase correlation > 0.9 | ✓ for all molecules |
| M7 | Zero-shot on H2O acceptable | MSE < 0.15 | ✓ (adjust ladder if not) |
| M8 | Uncertainty correlates with error | Monotonic binning | ✓ for production viability |
| M9 | 10-shot adaptation works | 5× improvement | ✓ for practical deployment |
| M10 | NWChem integration complete | End-to-end runs | ✓ for production path |
| M11 | Benzene validates at scale | MSE < 0.10, speedup > 10× | ✓ for paper submission |
| M12 | Documentation complete | External user can run | ✓ for release |

---

## Appendix A: Equivariant Field Encoding Implementation

```python
import torch
from e3nn import o3

class EquivariantFieldEncoder(torch.nn.Module):
    """
    Encode external electric field E(t) as L=1 spherical tensor,
    then couple with geometry features via equivariant tensor product.
    
    Physics: Light-matter coupling is μ·E where μ (dipole) is L=1.
    This encoding preserves selection rules for electronic transitions.
    """
    def __init__(self, geometry_irreps="32x0e + 16x1o + 8x2e"):
        super().__init__()
        
        # Field is a polar vector: L=1, odd parity
        self.field_irreps = o3.Irreps("1x1o")
        self.geometry_irreps = o3.Irreps(geometry_irreps)
        
        # Output irreps from tensor product: 0e⊗1o→1o, 1o⊗1o→0e+1e+2e, 2e⊗1o→1o+2o+3o
        self.output_irreps = o3.Irreps("16x0e + 16x1o + 16x1e + 8x2e + 8x2o")
        
        # Learnable tensor product for geometry-field coupling
        self.tp = o3.FullyConnectedTensorProduct(
            self.geometry_irreps,
            self.field_irreps,
            self.output_irreps,
            shared_weights=False
        )
        
        # Additional MLP on invariant (L=0) output for nonlinear processing
        self.invariant_mlp = torch.nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32)
        )
        
    def forward(self, geometry_features, field_cartesian):
        """
        Args:
            geometry_features: (batch, geometry_irreps.dim) from E3NN encoder
            field_cartesian: (batch, 3) electric field in Cartesian coords
            
        Returns:
            coupled_features: (batch, output_irreps.dim) geometry-field coupling
        """
        # Convert field to spherical harmonics (Y_1^m)
        # E3NN expects (batch, 3) for "1x1o" irrep
        field_spherical = field_cartesian  # Already in correct format for e3nn
        
        # Equivariant tensor product
        coupled = self.tp(geometry_features, field_spherical)
        
        # Extract and enhance invariant component
        invariant_slice = self.output_irreps.slices()[0]  # First 16x0e
        invariants = coupled[:, invariant_slice]
        enhanced_invariants = self.invariant_mlp(invariants)
        
        # Recombine (replace original invariants with enhanced)
        coupled[:, invariant_slice] = enhanced_invariants
        
        return coupled
```

---

## Appendix B: Complete Training Configuration

```yaml
# config/phase2_training.yaml
model:
  geometry_encoder:
    type: "e3nn_nequip"
    num_layers: 4
    irreps_hidden: "64x0e + 32x1o + 16x2e"
    radial_basis: "bessel"
    num_basis: 8
    cutoff: 5.0
    
  density_encoder:
    type: "universal_block"
    latent_dim: 64
    max_angular_momentum: 2
    attention_heads: 4
    
  dynamics:
    type: "geometry_conditioned_mamba"
    d_model: 256
    d_state: 16
    n_layers: 6
    conditioning: "film"  # or "cross_attention"
    
  decoder:
    type: "attention_projection"
    physics_projection: true
    
training:
  optimizer:
    type: "adamw"
    lr: 3e-4
    weight_decay: 0.01
    
  scheduler:
    type: "cosine_annealing"
    T_max: 100
    eta_min: 1e-6
    
  loss:
    reconstruction_weight: 1.0
    derivative_weight: 10.0
    hermiticity_weight: 1.0
    trace_weight: 5.0
    idempotency_weight: 0.5
    
  curriculum:
    variance_stages: [0.3, 0.6, 1.0]  # fraction of training
    variance_thresholds: [50, 25, 0]   # percentile cutoffs
    
  scheduled_sampling:
    warmup_epochs: 20
    final_teacher_force_prob: 0.1
    
  batch_size: 32
  sequence_length: 256
  gradient_clip: 1.0
  epochs: 100
  
data:
  molecules: ["H2+", "H2", "LiH"]
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  augmentation:
    geometry_noise: 0.05  # Å
    field_noise: 0.001    # a.u.
```

---

*Document Version: 1.0*  
*Generated: Research Roadmap for RT-TDDFT Universal Accelerator*  
*Program Duration: 12 Months*
