<div align="center">

# 🧠 Multi-Agent GNN Register Allocation

### *Predicting Register Spill & Coalescing with Graph Neural Networks*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-0095D5?style=for-the-badge)](https://networkx.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-6366F1?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

<br/>

> **A research and educational compiler + ML project** that uses four specialist Graph Neural Network agents and a confidence-weighted Consensus Arbiter to solve the classic NP-hard register allocation problem — complete with a live interactive browser dashboard.

<br/>

[🚀 Quick Start](#-quick-start) · [📊 Dashboard](#-launch-the-dashboard) · [🧬 Architecture](#-architecture) · [👥 Team](#-team) · [📖 Docs](#-cli-reference)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Team](#-team)
- [Architecture](#-architecture)
- [Models, Agents & Baselines](#-models-agents--baselines)
- [Repository Layout](#-repository-layout)
- [Quick Start](#-quick-start)
- [Run Tests](#-run-tests)
- [CLI Reference](#-cli-reference)
- [Launch the Dashboard](#-launch-the-dashboard)
- [Understanding Dashboard Results](#-understanding-dashboard-results)
- [Pipeline Deep Dive](#-pipeline-deep-dive)
- [Results & Metrics](#-results--metrics)
- [Limitations](#-limitations)

---

## 🔬 Overview

**Register Allocation** is one of the most critical and computationally hard phases in a compiler backend. It maps unlimited virtual registers (program values) to a small, fixed set of physical CPU registers. When registers are exhausted, values must be **spilled** to memory — which is expensive at runtime.

This project tackles register allocation with a **multi-agent GNN ensemble**:

| Challenge | Our Approach |
|-----------|-------------|
| NP-hard coloring | GNN-guided soft decisions |
| Diverse graph patterns | 4 specialized agents, each tuned to a different graph signal |
| Single-model blind spots | Confidence-weighted Consensus Arbiter combines all agents |
| Move instruction overhead | Coalescing edge analysis to eliminate redundant `MOVE`s |
| Classical baseline comparison | Chaitin-Briggs heuristic + Random allocator included |

The full pipeline covers:
- **Three-Address Code (TAC)** IR generation
- **Control Flow Graph (CFG)** construction & liveness analysis
- **Multi-Relational Interference Graph** with interference + coalescing edges
- **Four relational GNN architectures**: R-GCN, R-GAT, R-SAGE, R-GIN
- **Four specialist pattern agents**: RelationalAgent, AttentionAgent, NeighbourhoodAgent, PressureAgent
- **Confidence-Weighted Consensus Arbiter** (ensemble decision-maker)
- **Conflict repair**, benchmark metrics, and model comparison tooling
- **Interactive browser dashboard** with force-layout graph visualization

---

## 👥 Team

This is a **four-member group project** for our compiler design and machine learning course.

<table>
<tr>
  <th align="center">Member</th>
  <th align="center">Agent Ownership</th>
  <th align="center">Architecture</th>
  <th align="center">Key Responsibilities</th>
</tr>
<tr>
  <td align="center">
    <b>Divyanjali Tyagi</b><br/>
    <sub>Team Member 1</sub>
  </td>
  <td align="center">🔗 RelationalAgent (R-GCN)</td>
  <td align="center"><code>agents/rgcn_agent.py</code></td>
  <td>
    R-GCN multi-relational message passing, move-chain detection, affinity-edge feature engineering, coalescing pattern analysis
  </td>
</tr>
<tr>
  <td align="center">
    <b>Ishita Duggal</b><br/>
    <sub>Team Member 2</sub>
  </td>
  <td align="center">👁️ AttentionAgent (GAT)</td>
  <td align="center"><code>agents/gat_agent.py</code></td>
  <td>
    GAT attention-weight analysis, interference pressure scoring, top-attended neighbor diagnostics, attention-matrix visualization
  </td>
</tr>
<tr>
  <td align="center">
    <b>Prem Chand</b><br/>
    <sub>Team Member 3</sub>
  </td>
  <td align="center">🕸️ NeighbourhoodAgent (GraphSAGE)</td>
  <td align="center"><code>agents/sage_agent.py</code></td>
  <td>
    GraphSAGE neighborhood sampling, hub-node detection, dense-subgraph / k-core analysis, forcing clique identification
  </td>
</tr>
<tr>
  <td align="center">
    <b>Nitin</b><br/>
    <sub>Team Member 4</sub>
  </td>
  <td align="center">📈 PressureAgent (GCN)</td>
  <td align="center"><code>agents/gcn_agent.py</code></td>
  <td>
    GCN global interference pressure, loop-hot variable analysis, pressure-peak detection, Consensus Arbiter integration & pipeline coordination
  </td>
</tr>
</table>

> The team shares responsibility for testing, evaluation, documentation, the dashboard, and integrating all agents with the compiler pipeline.

---

## 🧬 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPILER PIPELINE                                │
│                                                                         │
│   TAC IR Generator → CFG Builder → Liveness Analyzer → IG Constructor  │
│        ir.py              cfg.py       liveness.py   interference_graph │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    Multi-Relational Interference Graph
                   (Nodes: Virtual Regs | Edges: Interference + Coalescing)
                                 │
              ┌──────────────────┴──────────────────┐
              │         SPECIALIST AGENTS           │
              │                                     │
    ┌─────────┴────────┐            ┌───────────────┴──────────────┐
    │  Relational (R-GCN)│         │  Attention (GAT)              │
    │  Move-chain focus │           │  Pressure attention focus     │
    └─────────┬────────┘            └───────────────┬──────────────┘
    ┌─────────┴────────┐            ┌───────────────┴──────────────┐
    │  Neighbourhood    │           │  Pressure (GCN)               │
    │  (GraphSAGE)      │           │  Global register pressure     │
    └─────────┬────────┘            └───────────────┬──────────────┘
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                    ╔════════════╧════════════╗
                    ║   CONSENSUS ARBITER     ║
                    ║  Confidence-weighted    ║
                    ║  soft-voting ensemble   ║
                    ╚════════════╤════════════╝
                                 │
              ┌──────────────────┴──────────────────┐
              │     Conflict Repair + Evaluation     │
              │   repair.py, evaluator.py, metrics   │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │      Interactive Web Dashboard       │
              │   web/index.html + web/app.js        │
              └─────────────────────────────────────┘
```

---

## 🤖 Models, Agents & Baselines

There are **two related-but-distinct** GNN workflows in this project:

### Relational GNN Architectures *(training & comparison)*

| Architecture | File | Key Idea |
|---|---|---|
| **R-GCN** | `models/gnn_allocator.py` | Relation-specific weight matrices for interference + coalescing message passing |
| **R-GAT** | `models/rgat_allocator.py` | Learned attention weights per relation type |
| **R-SAGE** | `models/rsage_allocator.py` | Mean neighborhood aggregation per relation |
| **R-GIN** | `models/gin_allocator.py` | Sum aggregation + relation-specific MLP updates for maximum expressive power |

### Specialist Pattern Agents *(multi-agent ensemble)*

| Agent | File | Pattern Focus |
|---|---|---|
| **RelationalAgent** (R-GCN) | `agents/rgcn_agent.py` | Move chains, coalescing affinity edges, relational edge ratios |
| **AttentionAgent** (GAT) | `agents/gat_agent.py` | Inward attention pressure, top interfering neighbors forcing spills |
| **NeighbourhoodAgent** (GraphSAGE) | `agents/sage_agent.py` | Hub nodes, dense subgraphs, k-cores, forcing cliques |
| **PressureAgent** (GCN) | `agents/gcn_agent.py` | Global pressure peaks, loop-hot variables, choking-point detection |

### Ensemble & Baselines

| Component | Type | Description |
|---|---|---|
| **Consensus Arbiter** | `agents/consensus.py` | Two-stage confidence-weighted soft-voting across all 4 agents |
| **Chaitin-Briggs** | `compiler/chaitin_briggs.py` | Classical Kempe-chain graph-coloring with optimistic spilling |
| **Random** | `models/trainer.py` | Uniform random register assignment — lower-bound baseline |

---

## 📁 Repository Layout

```
Graph Neural Network/
│
├── main.py                    # 🚀 CLI entry point — train, evaluate, export, compare
│
├── compiler/                  # 📦 Compiler pipeline
│   ├── ir.py                  #   Three-address code IR & TAC generator
│   ├── cfg.py                 #   Control flow graph builder
│   ├── liveness.py            #   Liveness analysis (live-in / live-out)
│   ├── interference_graph.py  #   Interference + coalescing graph builder
│   ├── chaitin_briggs.py      #   Classical heuristic allocator baseline
│   └── export.py              #   JSON export for dashboard
│
├── agents/                    # 🤖 Specialist pattern agents
│   ├── base.py                #   Abstract agent base class
│   ├── rgcn_agent.py          #   RelationalAgent  (Divyanjali Tyagi)
│   ├── gat_agent.py           #   AttentionAgent   (Ishita Duggal)
│   ├── sage_agent.py          #   NeighbourhoodAgent (Prem Chand)
│   ├── gcn_agent.py           #   PressureAgent    (Nitin)
│   ├── consensus.py           #   Confidence-weighted Consensus Arbiter
│   └── registry.py            #   Agent factory & checkpoint loader
│
├── models/                    # 🧠 Relational GNN architectures
│   ├── gnn_allocator.py       #   R-GCN architecture
│   ├── rgat_allocator.py      #   R-GAT architecture
│   ├── rsage_allocator.py     #   R-SAGE architecture
│   ├── gin_allocator.py       #   R-GIN architecture
│   ├── base_allocator.py      #   Shared base model
│   ├── repair.py              #   Conflict repair heuristics
│   └── trainer.py             #   Training loop, random baseline
│
├── dataset/                   # 📊 Synthetic program generation
│   ├── generator.py           #   TAC program generator
│   └── dataset.py             #   Graph dataset builder
│
├── evaluation/                # 📈 Benchmarking & metrics
│   ├── evaluator.py           #   Multi-model evaluator
│   ├── metrics.py             #   Spill cost, move elimination, validity metrics
│   └── model_comparator.py    #   Cross-architecture comparison & report
│
├── tests/                     # ✅ Pytest test suite
│   ├── test_agents.py
│   ├── test_consensus.py
│   ├── test_relational_models.py
│   ├── test_chaitin.py
│   ├── test_compiler.py
│   ├── test_end_to_end.py
│   ├── test_gnn.py
│   └── test_repair.py
│
├── web/                       # 🌐 Interactive browser dashboard
│   ├── index.html             #   Dashboard UI
│   ├── app.js                 #   Force-layout graph engine + tab rendering
│   ├── style.css              #   Dark glassmorphism UI styles
│   ├── data.json              #   Multi-agent dataset (main run)
│   ├── comparison_data.json   #   Architecture comparison dataset
│   └── embedded_data.js       #   Auto-generated for offline/file:// rendering
│
├── checkpoints/               # 💾 Saved agent model weights
├── results/                   # 📋 Generated evaluation results
├── report/                    # 📑 Comparison plots & markdown reports
├── docs/                      # 📚 Supplementary documentation
├── gnn-regalloc/              # 🔗 Separate Java + PyG ML sub-workflow
├── gnn_allocator.pt           # 🏆 Trained model checkpoint
└── requirements.txt           # 📦 Python dependencies
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/nitinrohilla-05/Graph-Neural-Network-Registrer-Allocation-to-predict-regisster-spill-coalescing-.git
cd Graph-Neural-Network-Registrer-Allocation-to-predict-regisster-spill-coalescing-
```

Create and activate a virtual environment:

```bash
# Create venv
python -m venv .venv

# Activate — Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **PyTorch Note:** If the default `torch` wheel does not match your GPU/OS, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to get the correct CUDA-enabled wheel before installing the rest.

### 3. Verify Installation

```bash
python -m pytest tests/ -v --tb=short
```

### 4. Run the Full Pipeline (One Command)

```bash
# Train agents → export dashboard data → launch browser dashboard
python main.py --mode train-agents --quick --checkpoint-dir checkpoints
python main.py --mode export-web-data --registers 4
python -m http.server 8000 --directory web
# → Open http://localhost:8000
```

---

## ✅ Run Tests

Run the full test suite:

```bash
python -m pytest
```

Run specific modules:

```bash
python -m pytest tests/test_agents.py -v          # Agent tests
python -m pytest tests/test_consensus.py -v       # Consensus arbiter tests
python -m pytest tests/test_relational_models.py -v  # GNN architecture tests
python -m pytest tests/test_compiler.py -v        # Compiler pipeline tests
python -m pytest tests/test_end_to_end.py -v      # Full end-to-end tests
```

Run with concise output:

```bash
python -m pytest --tb=short -q
```

---

## 📖 CLI Reference

Use `python main.py --help` to see all flags.

### Train a Relational Architecture

```bash
# Train R-GCN (default)
python main.py --mode train --model rgcn --epochs 20 --samples 60 --registers 4

# Train other architectures
python main.py --mode train --model rgat --epochs 20
python main.py --mode train --model rsage --epochs 20
python main.py --mode train --model gin --epochs 20

# Custom output checkpoint path
python main.py --mode train --model rgcn --output checkpoints/rgcn_v2.pt
```

### Train & Evaluate Specialist Agents

```bash
# Quick CPU smoke run (reduced samples/epochs)
python main.py --mode train-agents --quick --checkpoint-dir checkpoints
python main.py --mode evaluate-agents --quick

# Full training run
python main.py --mode train-agents --samples 60 --epochs 20 --registers 4 --checkpoint-dir checkpoints

# Generate agent diagnostics report
python main.py --mode agents-report --quick
# → Writes results/agents_results.json & results/agents_results.md
```

### Compare All Models

```bash
# Quick comparison (single program)
python main.py --mode compare-models --registers 4

# Aggregate over multiple generated programs
python main.py --mode compare-models --samples 20 --registers 4
# → Writes web/comparison_data.json & report/model_comparison.png
```

### Export Dashboard Data

```bash
# Export multi-agent web visualizer dataset
python main.py --mode export-web-data --registers 4
# → Writes web/data.json & web/embedded_data.js

# Export raw synthetic graph samples
python main.py --mode export-json --samples 60 --registers 4
```

### All Available Modes

| Mode | Description |
|------|-------------|
| `train` | Train one relational GNN architecture (R-GCN / R-GAT / R-SAGE / R-GIN) |
| `evaluate` | Benchmark a model checkpoint against allocator baselines |
| `run-compiler` | Run the compiler allocation demo |
| `export-web-data` | Export multi-agent data for the interactive dashboard |
| `export-json` | Export raw synthetic graph samples as JSON |
| `model-comparison` | Multi-seed architecture comparison study |
| `ablation` | Feature ablation evaluation |
| `train-agents` | Train all four specialist agent checkpoints |
| `evaluate-agents` | Evaluate agents and the Consensus Arbiter |
| `agents-report` | Generate full agent result files |
| `compare-models` | Compare relational architectures + baselines |

### Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `rgcn` | GNN architecture: `rgcn`, `rgat`, `rsage`, `gin` |
| `--registers` | `4` | Number of physical registers K |
| `--samples` | `60` | Number of synthetic programs |
| `--epochs` | `20` | Training epochs |
| `--hidden-dim` | `64` | GNN hidden dimension |
| `--seed` | `42` | Random seed for reproducibility |
| `--checkpoint-dir` | `checkpoints` | Directory to load/save agent checkpoints |
| `--model-path` | `gnn_allocator.pt` | Path to model checkpoint |
| `--quick` | — | Use reduced samples/epochs for fast testing |

> ⚠️ Keep `--registers`, `--seed`, and `--checkpoint-dir` consistent when comparing results across runs.

---

## 🌐 Launch the Dashboard

The dashboard is a fully static web app — no backend server required for the embedded data.

### Option A: Local HTTP Server (Recommended for Fresh Data)

```bash
python -m http.server 8000 --directory web
```

Open **[http://localhost:8000](http://localhost:8000)** in any modern browser.

### Option B: Direct File Access (Offline Mode)

Simply open `web/index.html` directly in Chrome or Firefox. The `embedded_data.js` file provides instant offline rendering without a server.

### Dashboard Data Files

| File | Contents |
|------|----------|
| `web/data.json` | Multi-agent assignments, spill probs, confidence, disagreements |
| `web/comparison_data.json` | Cross-architecture comparison graph and agreement matrices |
| `web/embedded_data.js` | Auto-generated embedded datasets for offline/file:// access |

Refresh dashboard data by running:

```bash
python main.py --mode export-web-data
python main.py --mode compare-models
```

Then reload the page.

---

## 📊 Understanding Dashboard Results

### Panels & Tabs

| Panel | What It Shows |
|-------|--------------|
| **Multi-Relational Interference Graph** | Force-layout canvas. Nodes = virtual registers. Red edges = interference. Green dashed = coalescing/move edges. Hover nodes for full diagnostics. Drag to reposition. Scroll to zoom. |
| **Register Table** | Per-variable assignments, loop depth, spill cost, interference degree, spill probability, and confidence — for the active allocator |
| **Agent Patterns** | Specialist agent diagnostics: move chains (RelationalAgent), top-pressure nodes (AttentionAgent), hub nodes & k-cores (NeighbourhoodAgent), loop-hot variables (PressureAgent) |
| **Agreement Matrix** | Pairwise register assignment agreement, spill decision concordance, and coalescing Jaccard overlap across all model pairs |
| **TAC IR** | Full three-address code instruction listing for the compiled program |
| **CFG** | Basic blocks with loop depths and successor edges |

### Node Color Legend

| Color | Meaning |
|-------|---------|
| 🔵 Blue | Assigned to **R0** |
| 🟢 Green | Assigned to **R1** |
| 🟡 Amber | Assigned to **R2** |
| 🩷 Pink | Assigned to **R3** |
| 🔴 Red | **SPILLED** to stack memory |
| 🟠 Orange ring | **Disagreement** — models disagree on this variable |

### Switching Datasets

Use the **"Load Comparison Run"** button in the top bar to toggle between:
- **Multi-agent dataset** — `data.json` — shows Consensus, R-GCN, GAT, GraphSAGE, GCN, Chaitin-Briggs
- **Architecture comparison** — `comparison_data.json` — shows R-GCN, R-GAT, R-SAGE, R-GIN, Chaitin-Briggs, Random

> **Important:** Agreement measures whether two allocators made the *same* decisions, not whether those decisions are optimal. A low spill count alone does not prove validity — check conflict validity, spill cost, move elimination, and evaluation conditions together.

---

## 🔬 Pipeline Deep Dive

### Step 1 — TAC Program Generation

`dataset/generator.py` synthesizes random three-address code programs with configurable loop nesting, variable counts, and `MOVE` instruction density.

### Step 2 — CFG Construction & Liveness

`compiler/cfg.py` partitions instructions into basic blocks and links them with control-flow edges. `compiler/liveness.py` performs iterative backward dataflow to compute **live-in / live-out** sets per block.

### Step 3 — Interference Graph

`compiler/interference_graph.py` constructs a multi-relational graph:
- **Interference edges** between any two variables simultaneously live
- **Coalescing edges** between the source and destination of `MOVE` instructions

### Step 4 — GNN Inference

Each relational architecture processes the graph with relation-specific message passing (R-GCN / R-GAT / R-SAGE / R-GIN) to produce per-variable register assignment logits and spill probability estimates.

### Step 5 — Consensus Arbiter

`agents/consensus.py` gathers each specialist agent's output probability distribution and performs **two-stage confidence-weighted soft-voting** to produce a final, robust allocation decision with agreement percentages.

### Step 6 — Repair & Export

`models/repair.py` checks for constraint violations (two interfering variables assigned the same register) and applies lightweight heuristic fixes. Results are exported to JSON for evaluation and dashboard display.

---

## 📈 Results & Metrics

After running `python main.py --mode agents-report`, check:

| Output | Location |
|--------|----------|
| Agent results JSON | `results/agents_results.json` |
| Agent report markdown | `results/agents_results.md` |
| Model comparison plot | `report/model_comparison.png` |
| Phase-2 agent report | `report/phase2_multi_agent.md` |
| Baseline notes | `docs/baseline_results.md` |

Key metrics computed:

| Metric | Description |
|--------|-------------|
| **Spill Count** | Number of variables spilled to stack memory |
| **Spill Cost** | Weighted spill cost (loop depth × base cost) |
| **Moves Eliminated** | Number of `MOVE` instructions successfully coalesced |
| **Move Elimination Rate** | Percentage of possible moves eliminated |
| **Pairwise Agreement** | Exact register match % between any two model pair |
| **Spill Concordance** | Agreement % specifically on spill vs. non-spill decisions |
| **Coalescing Jaccard** | Overlap of eliminated MOVE sets between model pairs |
| **Unanimous %** | Proportion of variables where all models agree |
| **Inference Latency** | Per-model inference time in milliseconds |

---

## ⚠️ Limitations

- **Synthetic programs only** — the TAC generator produces artificial code; results may not generalize to real-world compiler workloads.
- **Results are program-dependent** — spill counts and agreement metrics vary with the generated program, training seed, checkpoint, hyperparameters, and register count.
- **Chaitin-Briggs & Random are baselines** — they are not neural architectures and serve only as comparison anchors.
- **Dashboard data is a snapshot** — the JSON files represent one exported run and only update when regenerated.
- **`gnn-regalloc/` is a separate workflow** — the nested Java + PyTorch-Geometric project under `gnn-regalloc/` is independent and uses its own `requirements.txt`.
- **No production backend** — this is a research/educational prototype and not a drop-in replacement for production compiler register allocators (LLVM, GCC, etc.).

---

## 📚 Further Reading

| Resource | Description |
|----------|-------------|
| [`docs/baseline_results.md`](docs/baseline_results.md) | Baseline allocator evaluation notes |
| [`report/phase2_multi_agent.md`](report/phase2_multi_agent.md) | Multi-agent design and results report |
| [`gnn-regalloc/README.md`](gnn-regalloc/README.md) | Separate Java + PyG ML sub-workflow documentation |

### Key References

- Chaitin, G.J. et al. — *"Register allocation via coloring"* (1981)
- Schlichtkrull et al. — *"Modeling Relational Data with Graph Convolutional Networks"* (R-GCN, 2018)
- Veličković et al. — *"Graph Attention Networks"* (GAT, 2018)
- Hamilton et al. — *"Inductive Representation Learning on Large Graphs"* (GraphSAGE, 2017)
- Xu et al. — *"How Powerful are Graph Neural Networks?"* (GIN, 2019)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests before pushing: `python -m pytest`
4. Open a pull request with a clear description of your changes

---

## 📄 License

No license file is currently included in this repository. Contact the project
authors before redistributing or reusing the code.

---

## Live website input (TAC → allocation)

The dashboard can now run a user-supplied Three-Address Code (TAC) program instead of only showing the pre-exported graph. Use Python 3.12 in the project root:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 web_server.py
```

Open `http://127.0.0.1:8000` and use the **Run Your Own TAC** panel. This server serves the dashboard and handles `POST /api/allocate` requests, so do not launch the page with `python -m http.server` when using live allocation.

Supported TAC examples:

```text
a = 10
b = 20
c = a + b
d = c * 2
e = a + d
return e
```

The parser also supports variable copies (`x = y`), subtraction, labels (`L1:`), `goto L1`, conditional branches (`if x == y goto L1` and `if x != 0 goto L1`), `return`, and simple `LOAD`, `STORE`, and `CALL` forms. It is a TAC parser, not a C compiler; arbitrary C source code is not supported by this input box.

The response updates the dashboard's TAC, CFG, interference graph, allocation table, and summary metrics. Compatible saved agent checkpoints are loaded from `checkpoints/`; `gnn_allocator.pt` is used as an R-GCN fallback. If no compatible trained checkpoint exists for an agent, the UI warns that its prediction comes from a newly initialized model. Chaitin–Briggs remains the classical baseline.
