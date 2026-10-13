# GNN-based register allocation

This project generates compiler interference graphs in Java, trains PyG models
to predict spills, repairs infeasible predictions, and benchmarks the result
against a conservative Chaitin–Briggs allocator.

## Layout

- `compiler_side/`: CFG/liveness/interference extraction, labels, JSON export,
  synthetic generation, and classical benchmark.
- `data/generated/{train,val,test}/`: 500 graph-disjoint JSON samples.
- `ml_side/`: PyG dataset, spill and register models, evaluation, repair, and
  coalescing head.
- `report/`: Phase 1 derivation and Phase 9 results.
- `results/`: metrics, plots, checkpoints, and Java benchmark timing CSV.

## Reproduce

```powershell
python -m pip install -r requirements.txt

# Compile Java sources and generate the 500 graph corpus.
$sources = Get-ChildItem compiler_side\src\main\java -Recurse -Filter *.java | ForEach-Object FullName
javac -d compiler_side\target\classes $sources
java -cp compiler_side\target\classes generator.DatasetGenerator data\generated 500 42

# Train the spill baseline and run the three-seed comparison.
python -m ml_side.train --data-root data/generated --epochs 100
python -m ml_side.evaluate --data-root data/generated --epochs 30 --batch-size 128

# Benchmark the selected checkpoint against the Java allocator.
java -cp compiler_side\target\classes generator.ClassicalAllocatorBenchmark results\classical_benchmark.csv 500 42 425 500
python -m ml_side.benchmark --checkpoint results/phase6_best_model.pt --classical-csv results/classical_benchmark.csv
```

The existing Phase 6 results use a one-epoch CPU smoke budget; see
[`report/phase9_results.md`](report/phase9_results.md) for the recorded
limitations and longer-run command.
