# Project Guide

This guide describes the repository's register-allocation workflows, how to run
them, and how the dashboard relates to the models. Run commands from the
repository root unless a command explicitly changes directories.

## 1. What this project does

Register allocation maps compiler virtual registers (values that exist in the
intermediate representation) onto a limited set of physical registers. Values
that are simultaneously live interfere and cannot use the same physical
register. When the available registers are insufficient, an allocator can
spill values to memory. A move can sometimes be coalesced by assigning its
source and destination to the same register.

The main Python workflow generates synthetic three-address code (TAC), derives
control-flow and liveness information, builds an interference graph, and asks
neural models or a classical allocator for register assignments. Conflict
repair is available to turn infeasible neural predictions into valid
allocations. Evaluation code reports allocation and comparison metrics.

This is a research/learning prototype, not a production compiler backend. The
main data generator creates synthetic programs, and benchmark results depend
on the generated program, checkpoint, configuration, and repair settings.

## 2. Models, agents, and baselines

The repository exposes several related but distinct concepts:

| Name | What it is | Role |
| --- | --- | --- |
| **R-GCN** | Relational graph convolution architecture | Relational model and multi-agent specialist |
| **R-GAT** | Relational graph attention architecture | Selectable relational architecture |
| **R-SAGE** | Relational GraphSAGE architecture | Selectable relational architecture |
| **R-GIN** | Relational graph-isomorphism architecture | Selectable relational architecture |
| **GAT** | Graph attention architecture | Specialist agent focused on neighborhood/attention patterns |
| **GraphSAGE** | Neighborhood aggregation architecture | Specialist agent focused on local neighborhoods |
| **GCN** | Graph convolution architecture | Specialist agent focused on graph-wide pressure |
| **Consensus Arbiter** | Confidence-weighted ensemble | Combines the four specialist agents' predictions; it is not a separate GNN |
| **Chaitin-Briggs** | Classical graph-coloring heuristic | Non-AI baseline |
| **Random** | Randomized allocation baseline | Non-AI baseline |

There are **four selectable relational model architectures** (R-GCN, R-GAT,
R-SAGE, R-GIN) and **four specialist agents** (R-GCN, GAT, GraphSAGE, GCN).
Those lists belong to different workflows; their names do not mean there are
eight independent model families. The actual selectable dashboard buttons
depend on the dataset currently loaded.

## 3. How the multi-agent pipeline works

At a high level, the agent workflow is:

1. Generate a TAC program and build its control-flow graph.
2. Compute liveness and construct an interference graph, including
   move/coalescing relationships.
3. Let the R-GCN, GAT, GraphSAGE, and GCN specialist agents predict decisions.
4. Combine those predictions through the Consensus Arbiter.
5. Repair conflicting assignments where configured, and compare with
   Chaitin-Briggs.
6. Export graph, assignment, and diagnostic data for the web dashboard.

The Consensus Arbiter uses agents' confidence when aggregating their
probability predictions. It is the multi-agent ensemble output, not an
independently trained architecture. Chaitin-Briggs uses graph-coloring
heuristics; it does not use a neural network.

## 4. Repository layout

| Directory/file | Purpose |
| --- | --- |
| `main.py` | Entry point for training, evaluation, generation, and export commands |
| `compiler/` | TAC representation, CFG construction, liveness, interference graphs, and classical allocation |
| `dataset/` | Synthetic program generation and graph datasets |
| `agents/` | Agent implementations, checkpoint registry, and consensus logic |
| `models/` | Relational architectures, training helpers, and allocation repair |
| `evaluation/` | Benchmarking, metrics, ablations, and cross-model comparison |
| `tests/` | Python unit and integration tests |
| `web/` | Static interactive dashboard and JSON datasets used by the dashboard |
| `checkpoints/` | Optional local trained agent checkpoints |
| `report/`, `docs/`, `results/` | Reports, baseline documentation, and generated outputs |
| `gnn-regalloc/` | A separate Java compiler / Python ML workflow with its own README and dependencies |

The root `requirements.txt` is for the main Python project. The nested
`gnn-regalloc/requirements.txt` belongs to that separate workflow; install it
only when working on the nested project.

## 5. Setup

Use a supported Python installation and run commands from the repository root.
Creating a virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in a POSIX shell:

```bash
source .venv/bin/activate
```

Install the root project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The declared dependencies include PyTorch, NetworkX, Matplotlib,
scikit-learn, NumPy, and pytest. PyTorch installation can vary by operating
system and hardware accelerator; if the default package does not match your
machine, use the install command recommended by the official PyTorch
installation selector.

## 6. Tests

Run the Python test suite from the repository root:

```bash
python -m pytest
```

To run a focused test module:

```bash
python -m pytest tests/test_agents.py
```

Tests verify project behavior; successful tests do not by themselves establish
that a model will improve allocation quality on real-world compiler workloads.

## 7. Common commands

The CLI entry point is `main.py`. Its `--mode` selects an operation.

### Train a relational architecture

```bash
python main.py --mode train --model rgcn --epochs 20 --samples 60 --registers 4
```

Other supported `--model` choices are `rgat`, `rsage`, and `gin`. The model
checkpoint is written to `gnn_allocator.pt` by default; override it with
`--output`.

### Train the specialist agents

Quick CPU smoke run:

```bash
python main.py --mode train-agents --quick --checkpoint-dir checkpoints
```

Custom run:

```bash
python main.py --mode train-agents --samples 60 --epochs 20 --registers 4 --checkpoint-dir checkpoints
```

### Evaluate and report

```bash
python main.py --mode evaluate-agents --quick
python main.py --mode agents-report --quick
```

The report workflow writes results under `results/`, including
`agents_results.json` and `agents_results.md`.

### Compare model architectures

```bash
python main.py --mode compare-models --registers 4
```

Use multiple generated programs for aggregation:

```bash
python main.py --mode compare-models --samples 20 --registers 4
```

The comparison workflow writes `web/comparison_data.json` and a plot under
`report/`. It may also support a single-program run via `--single-program`.

### Export graph data

Export the multi-agent dashboard dataset:

```bash
python main.py --mode export-web-data --registers 4
```

Export raw synthetic graph files:

```bash
python main.py --mode export-json --samples 60 --registers 4
```

### Other CLI modes

| Mode | Purpose |
| --- | --- |
| `train` | Train one relational neural architecture |
| `evaluate` | Benchmark a model checkpoint against allocator baselines |
| `run-compiler` | Run the generated compiler allocation demonstration |
| `export-web-data` | Export graph and multi-agent data for the dashboard |
| `export-json` | Export generated graph samples |
| `model-comparison` | Run multi-seed model architecture comparison |
| `ablation` | Run feature ablation evaluation |
| `train-agents` | Train the specialist agent checkpoints |
| `evaluate-agents` | Evaluate specialist agents and consensus |
| `agents-report` | Generate agent result files |
| `compare-models` | Compare relational architectures and baselines |

Use `python main.py --help` for the CLI options available in the current
checkout.

## 8. Open the dashboard

The dashboard is a static web application in `web/`. Start a local HTTP server
from the repository root:

```bash
python -m http.server 8000 --directory web
```

Then open [http://localhost:8000](http://localhost:8000). The checked-in JSON
files provide example dashboard data. To display freshly generated data, run
the relevant export command before refreshing the page:

- `web/data.json`: multi-agent program, assignments, and diagnostic data.
- `web/comparison_data.json`: cross-architecture comparison data.
- `web/embedded_data.js`: generated embedded dashboard dataset used for
  file/offline rendering; the web exporter refreshes it alongside the
  multi-agent data.

Use **Load Comparison Run** to switch datasets. The available model buttons,
graph coloring, active allocation summary, and comparison panels depend on
the selected dataset and its exported model assignments.

## 9. Configuration and reproducibility

Important CLI options include:

| Option | Default | Meaning |
| --- | --- | --- |
| `--registers` | `4` | Number of physical registers, K |
| `--samples` | `60` | Number of generated training/comparison samples (mode-dependent) |
| `--epochs` | `20` | Training epoch count |
| `--hidden-dim` | `64` | Hidden feature width for model training |
| `--seed` | `42` | Random seed |
| `--model` | `rgcn` | Relational architecture for `train` |
| `--agent` | `all` | Agent architecture selection for supported agent workflows |
| `--checkpoint-dir` | `checkpoints` | Agent checkpoint directory |
| `--model-path` | `gnn_allocator.pt` | Checkpoint path for supported evaluation modes |
| `--web-output` | `web/data.json` | Output file for multi-agent dashboard export |

When comparing results, keep the dataset, seed, register count, checkpoint, and
repair behavior consistent. Do not compare isolated numbers from different
programs as though they came from the same benchmark.

## 10. Interpreting the dashboard and metrics

- **Graph nodes** represent virtual registers; edges distinguish interference
  from move/coalescing relationships.
- **Register colors** show the selected allocator's assignments. Spill
  assignments indicate values allocated to stack storage.
- **Agent Patterns** presents agent diagnostics and the currently selected
  allocator summary when those values exist in the dataset.
- **Agreement Matrix** compares allocation decisions. Agreement can differ by
  metric: exact register agreement is stricter than spill/non-spill agreement.
- **TAC IR** shows the generated three-address code and a per-variable
  allocation summary.
- **CFG** displays basic blocks and control-flow successors when the dataset
  includes them.

Agreement means two models made the same decision on the compared variables;
it does not mean their decisions are optimal. Likewise, low spill count alone
does not prove allocation validity or runtime improvement. Check conflict
validity, spill cost, move elimination, and the evaluation setup together.

## 11. Additional Java / ML workflow

The `gnn-regalloc/` directory is an additional, separately documented workflow
with Java compiler-side components and a Python/PyTorch-geometric ML side.
Consult [`gnn-regalloc/README.md`](gnn-regalloc/README.md) and its
`requirements.txt` for its own setup and reproduction steps. Its dependencies
and experiments are separate from the root-level CLI described above.

## 12. Limitations

- The main generator uses synthetic TAC programs; generated results may not
  generalize to real compiler workloads.
- Model quality depends on training data, checkpoint availability, hyperparameters,
  register count, and inference-time repair.
- The Chaitin-Briggs and Random baselines are comparison strategies, not neural
  architectures.
- Dashboard datasets are exported snapshots; they only change after new data
  is exported or the user selects another dataset.
- The nested Java/Python project is a separate workflow and should not be
  confused with the root-level Python application.

For detailed benchmark notes and methodology, see
[`report/`](report/), [`docs/baseline_results.md`](docs/baseline_results.md),
and [`gnn-regalloc/report/`](gnn-regalloc/report/).
