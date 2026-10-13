# Baseline Results & Audit Report (Step 0)

Date: 2026-10-07
Environment: Python 3.14.4, PyTorch 2.12.1, NetworkX 3.6.1, scikit-learn 1.9.0, pytest 9.1.1 (Windows 11 CPU)
Repository: `Graph-Neural-Network-Registrer-Allocation-to-predict-regisster-spill-coalescing-`

---

## 1. Test Suite Baseline

- `python -m pytest`: 15 passed in 19.49s
  - `tests/test_chaitin.py`: 2 passed
  - `tests/test_compiler.py`: 4 passed
  - `tests/test_end_to_end.py`: 2 passed
  - `tests/test_gnn.py`: 3 passed
  - `tests/test_repair.py`: 4 passed
- `python -m unittest discover -s tests`: 15 passed in 3.83s

---

## 2. CLI Modes Baseline Runs

### Mode: `train`
Command: `python main.py --mode train --samples 60 --epochs 20 --registers 4 --seed 42`
- Dataset: 60 programs (Train: 42, Val: 9, Test: 9)
- Training Loss: 1.6510 -> 0.2261
- Validation Accuracy: 55.8% (Violations: 29.41%)
- Held-out Test Accuracy: 56.2% (Color Violations: 46.67%)
- Saved checkpoint: `gnn_allocator.pt`

### Mode: `evaluate`
Command: `python main.py --mode evaluate --registers 4 --seed 42`
Test size: 30 generated programs (K=4)

| Strategy | Avg Spills | Avg Spill Cost | Move Elim % |
| :--- | :--- | :--- | :--- |
| **GNN (R-GCN)** | 9.27 | 210.63 | 2.4% |
| **Chaitin-Briggs** | 7.43 | 160.50 | 4.3% |
| **Random** | 9.00 | 238.03 | 1.9% |

*Note: GNN move elimination (2.4%) occurred only incidentally via register collisions since an empty coalesced list `[]` was supplied to metrics evaluation.*

### Mode: `run-compiler`
Command: `python main.py --mode run-compiler --registers 4 --seed 42`
Sample program: 29 instructions, 12 virtual registers, 48 interference edges, 6 move coalescing edges.
- GNN Spills: 7
- Chaitin-Briggs Spills: 4
- Post-Hoc Conflicts Repaired: 2

### Mode: `export-web-data`
Command: `python main.py --mode export-web-data --registers 4 --seed 42`
- Output: `web/data.json` successfully generated.

### Mode: `export-json`
Command: `python main.py --mode export-json --samples 20 --registers 4 --seed 42`
- Output: 20 graph files generated in `data/raw_graphs/`.

### Mode: `model-comparison`
Command: `python main.py --mode model-comparison --samples 30 --epochs 5 --registers 4 --seed 42`
Seeds: [0, 1, 2], Epochs: 5

| Architecture | Val Accuracy (%) | Avg Spills | Inference Time (ms) |
| :--- | :--- | :--- | :--- |
| **GCN** | 59.6% +/- 0.0% | 12.1 +/- 0.1 | 2.42 ms |
| **GraphSAGE** | 59.6% +/- 0.0% | 12.2 +/- 0.0 | 2.30 ms |
| **GAT** | 59.6% +/- 0.0% | 12.2 +/- 0.0 | 3.38 ms |
| **R-GCN** | 62.2% +/- 0.9% | 8.7 +/- 0.5 | 3.88 ms |

### Mode: `ablation`
Command: `python main.py --mode ablation --samples 30 --epochs 5 --registers 4 --seed 42`

| Feature Subset | Accuracy (%) | Violations (%) |
| :--- | :--- | :--- |
| **Full Features** | 61.5% | 66.67% |
| **No spill_cost** | 53.8% | 60.71% |
| **No loop_depth** | 61.5% | 80.00% |
| **No degree** | 59.6% | 44.44% |
| **No move_degree** | 55.8% | 85.71% |
| **No live_range_length** | 59.6% | 41.67% |
| **No use_count** | 65.4% | 54.55% |

---

## 3. Audit of the 6 Identified Gaps

### a) Checkpoint Architecture Serialization
- **Confirmed**: In `models/trainer.py`, `save_checkpoint` saves `{"model_state_dict", "in_node_features", "hidden_dim", "num_registers", "normalization_stats"}` without any `"architecture"` or `"model_type"` identifier. `load_checkpoint` unconditionally instantiates `RelationalGNNRegisterAllocator`.
- **Impact**: Checkpoints for GCN, GraphSAGE, or GAT cannot be saved or reloaded.

### b) Discarded Learned Coalescing
- **Confirmed**: In `evaluation/evaluator.py`, `benchmark_batch` calls:
  `gnn_metrics = CompilerAllocationMetrics.evaluate("GNN-Guided", ig, gnn_assignment, [], self.num_registers)` passing an empty list `[]` for coalesced pairs.
- Furthermore, `models/gnn_allocator.py` defines `CoalesceClassifier` as an unused standalone class that is never imported or called in training or benchmarking.

### c) Ignored Coalescing Adjacency in Baselines
- **Confirmed**: In `models/gnn_allocator.py`, `GCNSpillPredictor`, `SAGESpillPredictor`, and `GATSpillPredictor` accept `coal_adj: torch.Tensor = None` in their `forward()` signatures, but never use it. Only `RelationalGNNRegisterAllocator` actually passes `coal_adj` to `RelationalGConv`.

### d) GNN Spills Exceeding Chaitin-Briggs
- **Confirmed**: In the baseline evaluation, GNN averages 9.27 spills (spill cost 210.63) compared to Chaitin-Briggs at 7.43 spills (spill cost 160.50). The README also documents GNN at 9.00 spills vs CB at 8.44 spills.

### e) Repository Hygiene & Paths
- **Confirmed**: 22 compiled `.pyc` files in `compiler/__pycache__`, `dataset/__pycache__`, `evaluation/__pycache__`, `models/__pycache__`, and `tests/__pycache__` are tracked in git.
- **Confirmed**: No `.gitignore` file exists in repo root.
- **Confirmed**: `README.md` line 67 hardcodes the absolute local path `c:\ALl work\GNN\`.

### f) Nature of Ground-Truth Training Labels
- **Confirmed**: In `dataset/generator.py` line 124, `ground_truth = allocator.allocate(ig)` uses `ChaitinBriggsAllocator(num_registers=num_registers)`.
- The dataset is supervised by canonicalized Chaitin-Briggs outputs; hence the models are learning to imitate the Chaitin-Briggs heuristic rather than an optimal or oracle solver.
