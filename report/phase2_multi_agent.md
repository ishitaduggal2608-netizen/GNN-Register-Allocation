# Phase 2 Technical Report: Multi-Agent GNN Compiler Register Allocation & Multi-Architecture Comparative Study

## 1. Executive Summary

This report documents the architectural extension of the Register Allocation & Spill Coalescing compiler optimization platform from a single Relational Graph Convolutional Network (R-GCN) model to a cooperative **Multi-Agent Pattern-Detection Framework** alongside an in-depth **Multi-Architecture Comparison** across four multi-relational Graph Neural Network families:
1. **R-GCN** (Relational Graph Convolutional Network)
2. **R-GAT** (Relational Graph Attention Network)
3. **R-SAGE** (Relational GraphSAGE with Mean Aggregation)
4. **R-GIN** (Relation-Aware Graph Isomorphism Network)

To satisfy capstone course requirements for a team of 4 engineers, the optimization pipeline divides compiler graph pattern recognition among **4 specialized AI agents** (one per team member) and arbitrates their predictions using a central **Consensus Arbiter**. Furthermore, an empirical pattern-comparison study analyzes the distinct inductive biases of the four architectures on identical compiler interference graphs.

---

## 2. Multi-Agent Task Decomposition & Agent Roles

| Agent Name | Model Class | Member Role | Specialized Pattern Detected |
| :--- | :--- | :--- | :--- |
| **Relational Agent** | `R-GCN` | Member 1 (`M1_RELATIONAL`) | Move-chains, affinity copy edges, and coalescing opportunities across interference and coalescing relations. |
| **Attention Agent** | `GAT` | Member 2 (`M2_ATTENTION`) | Directed neighbor pressure, identifying choking neighbors whose high attention weight forces a node to spill. |
| **Neighbourhood Agent** | `GraphSAGE` | Member 3 (`M3_STRUCTURE`) | Dense $K$-cliques, high-degree hub nodes, and hard-to-color subgraphs via neighborhood sampling. |
| **Pressure Agent** | `GCN` | Member 4 (`M4_PRESSURE`) | Global register pressure peaks across basic blocks and loop-hot variables via symmetric Laplacian smoothing. |
| **Consensus Arbiter** | Ensemble | Central Coordinator | Two-stage confidence-weighted soft voting, conservative Briggs/George safe coalescing, and greedy repair. |

---

## 3. Four First-Class Multi-Relational Architectures

All four architectures extend the common base class `BaseRelationalAllocator` (`models/base_allocator.py`) and share the identical signature and output heads:
- **Register Assignment Logits Head**: $H_{\text{reg}}: \mathbb{R}^{d} \to \mathbb{R}^{K}$
- **Spill Classification Head**: $H_{\text{spill}}: \mathbb{R}^{d} \to \mathbb{R}^{1}$
- **Move Coalescing Score Head**: Bilinear relation scoring $S(u, v) = h_u^T W_{\text{coal}} h_v$

### 3.1 R-GCN (`models/gnn_allocator.py`)
Performs normalized linear relation-specific message passing:
$$h_i^{(l+1)} = \sigma \left( \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)} \right)$$
Captures global copy chains and symmetric interference constraints.

### 3.2 R-GAT (`models/rgat_allocator.py`)
Applies multi-relational self-attention per edge type:
$$\alpha_{ij}^r = \frac{\exp\left(\text{LeakyReLU}\left(a_r^T [W_r h_i \parallel W_r h_j]\right)\right)}{\sum_{k \in \mathcal{N}_r(i)} \exp\left(\text{LeakyReLU}\left(a_r^T [W_r h_i \parallel W_r h_k]\right)\right)}$$
Dynamically weights the influence of high-pressure interfering variables versus weak affinity copy hints.

### 3.3 R-SAGE (`models/rsage_allocator.py`)
Computes normalized mean neighborhood aggregation separately across relations followed by projection and combination:
$$h_{\mathcal{N}_r(i)} = \frac{1}{|\mathcal{N}_r(i)|} \sum_{j \in \mathcal{N}_r(i)} h_j^{(l)}$$
$$h_i^{(l+1)} = \sigma \left( W_{\text{self}} h_i^{(l)} + \sum_{r} W_r h_{\mathcal{N}_r(i)} \right)$$
Demonstrates strong inductive generalization to graphs with varying node counts.

### 3.4 R-GIN (`models/gin_allocator.py`)
Leverages sum-aggregation with trainable $\epsilon_r$ and a multi-layer perceptron (MLP) per edge relation:
$$h_i^{(l+1)} = \text{MLP}^{(l)} \left( (1 + \epsilon_0) h_i^{(l)} + \sum_{r \in \mathcal{R}} (1 + \epsilon_r) \sum_{j \in \mathcal{N}_r(i)} h_j^{(l)} \right)$$
Provides maximal expressive power for graph isomorphism and distinguishing dense non-isomorphic cliques.

---

## 4. Mathematical Formulation of Consensus Arbitration

### 4.1 Normalized Entropy Confidence
For a graph with $K$ physical registers, each agent outputs class probabilities $P_a(c_v = k)$ for $k \in \{0, \dots, K\}$, where class $K$ denotes stack spilling. 
The prediction entropy is computed as:
$$H_a(v) = - \sum_{k=0}^{K} P_a(c_v = k) \ln \left( P_a(c_v = k) + \epsilon \right)$$
Normalizing by the maximum entropy $\ln(K+1)$ yields the confidence metric:
$$C_a(v) = 1 - \frac{H_a(v)}{\ln(K+1)} \in [0, 1]$$

### 4.2 Two-Stage Soft Voting
1. **Spill vs Retain**:
   $$W_{\text{spill}}(v) = \frac{\sum_{a} C_a(v) \cdot P_a(c_v = K)}{\sum_{a} C_a(v)}$$
   If $W_{\text{spill}}(v) > 0.5$, variable $v$ is designated as a spill.
2. **Physical Register Assignment**:
   For retained variables, physical registers are chosen by:
   $$W_{\text{reg}}(v, k) = \frac{\sum_{a} C_a(v) \cdot P_a(c_v = k)}{\sum_{a} C_a(v)}, \quad k \in \{0, \dots, K-1\}$$
   $$\text{Reg}(v) = \arg\max_{k} W_{\text{reg}}(v, k)$$
   Deterministic tie-breaking prioritizes the lowest register index.

### 4.3 Conservative Safe Coalescing
A coalescing edge $(u, v) \in E_{\text{coal}}$ is approved only if:
1. Neither $u$ nor $v$ has been marked as spilled.
2. $u$ and $v$ do not interfere: $(u, v) \notin E_{\text{interf}}$.
3. The merge satisfies **Briggs' Rule** ($|\{w \in \text{adj}(u) \cup \text{adj}(v) : \text{deg}(w) \ge K\}| < K$) or **George's Rule** ($\forall w \in \text{adj}(u), (w, v) \in E_{\text{interf}} \lor \text{deg}(w) < K$).

### 4.4 100% Conflict-Free Repair
After register assignment, any remaining coloring conflict (where adjacent nodes $(u, v) \in E_{\text{interf}}$ share the same physical register) is resolved via greedy local recoloring or spilling the variable with lower loop depth and spill cost.

---

## 5. Experimental Results

### 5.1 Multi-Agent Benchmark Evaluation ($K=4$)

| Model / Agent | Architecture | Avg Spills | Avg Spill Cost | Move Elim (%) | Inference (ms) | Coloring Validity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Relational Agent** | R-GCN | 12.45 | 237.10 | 0.0% | 13.56 ms | 100.0% |
| **Attention Agent** | GAT | 15.65 | 377.10 | 0.0% | 11.83 ms | 100.0% |
| **Neighbourhood Agent** | GraphSAGE | 15.65 | 377.10 | 0.0% | 10.96 ms | 100.0% |
| **Pressure Agent** | GCN | 15.65 | 377.10 | 0.0% | 6.95 ms | 100.0% |
| **Consensus Arbiter** | Ensemble | **11.00** | **229.95** | 0.0% | 6.39 ms | **100.0%** |
| **Chaitin-Briggs** | Classical | 7.65 | 162.80 | 5.8% | 1.50 ms | 100.0% |

### 5.2 Cross-Architecture Pattern Comparison (Fixed Program Benchmark)

Benchmarked on test program `func_v11_5506` ($N=11$ variables, $E_{\text{interf}}=33$, $E_{\text{coal}}=4$):

| Architecture | Spills | Spill Cost | Moves Eliminated | Move Elim Rate | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **R-GCN** | 6 | 58.0 | 1 | 25.0% | 14.02 ms |
| **R-GAT** | 11 | 188.0 | 0 | 0.0% | 17.96 ms |
| **R-SAGE** | 10 | 145.0 | 0 | 0.0% | 9.71 ms |
| **R-GIN** | 11 | 188.0 | 0 | 0.0% | 8.59 ms |
| **Chaitin-Briggs** | 5 | 48.0 | 0 | 0.0% | 1.17 ms |
| **Random Baseline** | 2 | 22.0 | 0 | 0.0% | 0.05 ms |

**Pairwise Register Agreement Matrix (%)**:
- R-GCN vs R-GAT: 54.5%
- R-GCN vs R-SAGE: 54.5%
- R-GAT vs R-GIN: 100.0% (both converge on conservative spill decisions)
- R-GAT vs R-SAGE: 90.9%

A high-resolution publication plot is preserved in `report/model_comparison.png`.

---

## 6. Web Dashboard Architecture

The interactive dashboard at `web/` provides real-time visualization of:
1. **Interactive Force Layout Graph**: Zoom, pan, and drag virtual register nodes on an HTML5 canvas.
2. **Model Selector Bar**: Switch dynamically between `Consensus`, `R-GCN`, `R-GAT`, `R-SAGE`, `R-GIN`, `Chaitin-Briggs`, and `Random`.
3. **Dataset Switcher**: Toggle between the multi-agent ensemble dataset (`data.json`) and the 4-architecture cross-comparison dataset (`comparison_data.json`).
4. **Disagreement Highlighting**: Amber beacon rings highlight variables where models diverge.
5. **GAT Attention Edges**: Visualizes directed neighbor attention weights, indicating which interference edges create the highest spill pressure.
6. **Detailed Tabs**: Register Table, Architecture Pattern profiles, Pairwise Agreement Matrix, Coalescing Jaccard Overlap Matrix, Spill Decision Concordance Matrix, TAC IR, and CFG basic blocks.
