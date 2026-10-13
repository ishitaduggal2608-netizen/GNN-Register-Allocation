"""Phase 7 benchmark: classical labels/timing versus GNN spill prediction plus repair."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import torch
from torch_geometric.loader import DataLoader

try:
    from .dataset import FeatureNormalizer, RegAllocDataset
    from .models import build_spill_predictor
    from .repair import colouring_conflicts, repair_colouring
except ImportError:
    from dataset import FeatureNormalizer, RegAllocDataset
    from models import build_spill_predictor
    from repair import colouring_conflicts, repair_colouring


def _read_classical_times(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        return {row["graph_id"]: row for row in csv.DictReader(csv_file)}


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_spill_predictor(checkpoint["architecture"], checkpoint["in_channels"],
                                  checkpoint["hidden_channels"], checkpoint["num_layers"],
                                  checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalizer = FeatureNormalizer(checkpoint["normalizer"]["mean"], checkpoint["normalizer"]["std"])
    return model, normalizer, tuple(checkpoint.get("feature_indices") or range(checkpoint["in_channels"]))


@torch.no_grad()
def run_benchmark(test_dir: str | Path, checkpoint_path: str | Path,
                  classical_csv: str | Path | None = None, output_dir: str | Path = "results") -> dict:
    """Benchmarks individual and batched GNN inference; repair guarantees valid output colours."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, feature_indices = _load_model(Path(checkpoint_path), device)
    dataset = RegAllocDataset(test_dir, normalizer, feature_indices)
    classical_times = _read_classical_times(classical_csv)
    rows, gnn_single_seconds, gnn_pipeline_seconds = [], [], []

    for data in dataset:
        started = time.perf_counter()
        probabilities = torch.sigmoid(model(data.x.to(device), data.edge_index.to(device))).cpu()
        gnn_seconds = time.perf_counter() - started
        repaired = repair_colouring(probabilities, data.edge_index, int(data.num_registers))
        pipeline_seconds = time.perf_counter() - started
        classical_colours = data.y_register.clone()
        classical_colours[data.y_spill.bool()] = -1
        graph_id = data.graph_id
        baseline = classical_times.get(graph_id, {})
        rows.append({
            "graph_id": graph_id,
            "classical_spills": int(data.y_spill.sum().item()),
            "classical_valid": colouring_conflicts(classical_colours, data.edge_index) == 0,
            "classical_seconds": float(baseline["classical_allocator_seconds"])
            if baseline else None,
            "gnn_repaired_spills": repaired.total_spills,
            "gnn_pre_repair_valid": repaired.pre_repair_valid,
            "gnn_pre_repair_conflicts": repaired.pre_repair_conflicts,
            "gnn_post_repair_valid": repaired.post_repair_valid,
            "gnn_single_graph_seconds": gnn_seconds,
            "gnn_repair_pipeline_seconds": pipeline_seconds,
        })
        gnn_single_seconds.append(gnn_seconds)
        gnn_pipeline_seconds.append(pipeline_seconds)

    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    started = time.perf_counter()
    _ = model(batch.x.to(device), batch.edge_index.to(device))
    batch_seconds = time.perf_counter() - started
    report = {
        "graphs": rows,
        "summary": {
            "num_graphs": len(rows),
            "classical_mean_spills": sum(row["classical_spills"] for row in rows) / len(rows),
            "gnn_repaired_mean_spills": sum(row["gnn_repaired_spills"] for row in rows) / len(rows),
            "classical_valid_rate": sum(row["classical_valid"] for row in rows) / len(rows),
            "gnn_pre_repair_valid_rate": sum(row["gnn_pre_repair_valid"] for row in rows) / len(rows),
            "gnn_post_repair_valid_rate": sum(row["gnn_post_repair_valid"] for row in rows) / len(rows),
            "gnn_mean_single_graph_seconds": sum(gnn_single_seconds) / len(gnn_single_seconds),
            "gnn_mean_repair_pipeline_seconds": sum(gnn_pipeline_seconds) / len(gnn_pipeline_seconds),
            "gnn_batched_seconds_per_graph": batch_seconds / len(rows),
        },
    }
    timed_classical = [row["classical_seconds"] for row in rows if row["classical_seconds"] is not None]
    if timed_classical:
        report["summary"]["classical_mean_allocator_seconds"] = sum(timed_classical) / len(timed_classical)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase7_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(report, output_dir)
    return report


def _plot(report: dict, output_dir: Path) -> None:
    rows, summary = report["graphs"], report["summary"]
    pyplot.figure(figsize=(5, 5))
    pyplot.scatter([row["classical_spills"] for row in rows],
                   [row["gnn_repaired_spills"] for row in rows], alpha=0.7)
    maximum = max([1] + [max(row["classical_spills"], row["gnn_repaired_spills"]) for row in rows])
    pyplot.plot([0, maximum], [0, maximum], "k--", linewidth=1)
    pyplot.xlabel("Classical allocator spills")
    pyplot.ylabel("GNN + repair spills")
    pyplot.tight_layout()
    pyplot.savefig(output_dir / "phase7_spill_scatter.png", dpi=160)
    pyplot.close()

    labels = ["GNN inference", "GNN + repair", "GNN batched"]
    values = [summary["gnn_mean_single_graph_seconds"], summary["gnn_mean_repair_pipeline_seconds"],
              summary["gnn_batched_seconds_per_graph"]]
    if "classical_mean_allocator_seconds" in summary:
        labels.insert(0, "Classical")
        values.insert(0, summary["classical_mean_allocator_seconds"])
    pyplot.figure(figsize=(6, 4))
    pyplot.bar(labels, values)
    pyplot.ylabel("Mean seconds per graph")
    pyplot.tight_layout()
    pyplot.savefig(output_dir / "phase7_inference_time.png", dpi=160)
    pyplot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/generated/test"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/phase6_best_model.pt"))
    parser.add_argument("--classical-csv", type=Path, default=Path("results/classical_benchmark.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    csv_path = arguments.classical_csv if arguments.classical_csv.exists() else None
    print(run_benchmark(arguments.test_dir, arguments.checkpoint, csv_path, arguments.output_dir)["summary"])


if __name__ == "__main__":
    main()
