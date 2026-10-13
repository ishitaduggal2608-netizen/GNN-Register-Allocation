"""Three-seed architecture comparisons and feature ablations for spill prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import torch

try:
    from .dataset import RegAllocDataset
    from .train import run_training
except ImportError:
    from dataset import RegAllocDataset
    from train import run_training


FEATURE_NAMES = ("degree", "live_range_length", "loop_depth", "use_def_count", "move_related")
SEEDS = (0, 1, 2)


def _summary(scores: list[float]) -> dict[str, float | list[float]]:
    return {"scores": scores, "mean_f1": mean(scores), "std_f1": pstdev(scores)}


def _run_seeds(data_root: Path, architecture: str, feature_indices: tuple[int, ...], epochs: int,
               batch_size: int, hidden_channels: int, num_layers: int, dropout: float,
               learning_rate: float):
    normalizer = RegAllocDataset(data_root / "train", feature_indices=feature_indices).fit_normalizer()
    scores, runs = [], []
    best = None
    for seed in SEEDS:
        model, normalizer, metrics = run_training(
            data_root / "train", data_root / "val", data_root / "test", architecture=architecture,
            hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, seed=seed,
            feature_indices=feature_indices, verbose=False,
            normalizer=normalizer,
        )
        f1 = metrics["test_f1"]
        scores.append(f1)
        runs.append({"seed": seed, **metrics})
        if best is None or f1 > best[0]:
            best = (f1, seed, model, normalizer)
    return _summary(scores) | {"runs": runs}, best


def run_model_comparison(data_root: str | Path, output_dir: str | Path = "results",
                         epochs: int = 30, batch_size: int = 64, hidden_channels: int = 64,
                         num_layers: int = 3, dropout: float = 0.3, learning_rate: float = 1e-3,
                         include_ablation: bool = True,
                         architectures: tuple[str, ...] = ("gcn", "sage", "gat"),
                         append: bool = False) -> dict:
    """Train GCN/SAGE/GAT on exactly the same split, feature set, seeds, and budget."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_features = tuple(range(len(FEATURE_NAMES)))
    result_path = output_dir / "phase6_results.json"
    if append and result_path.exists():
        results = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        results = {"config": {"seeds": list(SEEDS), "epochs": epochs, "batch_size": batch_size,
                              "hidden_channels": hidden_channels, "num_layers": num_layers,
                              "dropout": dropout, "learning_rate": learning_rate,
                              "features": list(FEATURE_NAMES)}, "models": {}, "ablations": {}}

    best_overall = None
    for architecture in architectures:
        summary, best_run = _run_seeds(data_root, architecture, full_features, epochs, batch_size,
                                       hidden_channels, num_layers, dropout, learning_rate)
        results["models"][architecture] = summary
        if best_overall is None or summary["mean_f1"] > best_overall[0]:
            best_overall = (summary["mean_f1"], architecture, best_run)

    overall_best_architecture = max(results["models"], key=lambda name: results["models"][name]["mean_f1"])
    # Save a checkpoint only if this invocation trained the currently best architecture.
    if best_overall is not None and best_overall[1] == overall_best_architecture:
        _, best_architecture, (_, best_seed, best_model, best_normalizer) = best_overall
        torch.save({
            "model_state": best_model.state_dict(), "architecture": best_architecture,
            "in_channels": len(full_features), "hidden_channels": hidden_channels,
            "num_layers": num_layers, "dropout": dropout, "feature_indices": full_features,
            "normalizer": best_normalizer.state_dict(), "selection_seed": best_seed,
        }, output_dir / "phase6_best_model.pt")
    results["best_architecture"] = overall_best_architecture

    if include_ablation:
        subsets = {
            "all_features": full_features,
            "no_loop_depth": (0, 1, 3, 4),
            "no_move_related": (0, 1, 2, 3),
            "no_live_range_length": (0, 2, 3, 4),
        }
        for name, indices in subsets.items():
            summary, _ = _run_seeds(data_root, best_architecture, indices, epochs, batch_size,
                                    hidden_channels, num_layers, dropout, learning_rate)
            results["ablations"][name] = summary | {"feature_names": [FEATURE_NAMES[i] for i in indices]}

    _write_outputs(results, output_dir)
    return results


def _write_outputs(results: dict, output_dir: Path) -> None:
    (output_dir / "phase6_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Phase 6 model comparison", "", "| Model | Test F1 (mean ± std, 3 seeds) |", "|---|---:|"]
    for model, summary in results["models"].items():
        lines.append(f"| {model.upper()} | {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f} |")
    if results["ablations"]:
        lines += ["", "## Ablation of " + results["best_architecture"].upper(), "",
                  "| Feature subset | Test F1 (mean ± std, 3 seeds) |", "|---|---:|"]
        for name, summary in results["ablations"].items():
            lines.append(f"| {name} | {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f} |")
    (output_dir / "phase6_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = [name.upper() for name in results["models"]]
    means = [results["models"][name.lower()]["mean_f1"] for name in names]
    stds = [results["models"][name.lower()]["std_f1"] for name in names]
    pyplot.figure(figsize=(6, 4))
    pyplot.bar(names, means, yerr=stds, capsize=5)
    pyplot.ylabel("Test spill F1")
    pyplot.ylim(0, 1)
    pyplot.tight_layout()
    pyplot.savefig(output_dir / "phase6_model_comparison.png", dpi=160)
    pyplot.close()


def run_feature_ablation(data_root: str | Path, output_dir: str | Path = "results",
                         selected: tuple[str, ...] | None = None) -> dict:
    """Adds the four requested ablations to an existing architecture-comparison result."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    result_path = output_dir / "phase6_results.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    config = results["config"]
    best_architecture = results["best_architecture"]
    subsets = {
        "all_features": (0, 1, 2, 3, 4),
        "no_loop_depth": (0, 1, 3, 4),
        "no_move_related": (0, 1, 2, 3),
        "no_live_range_length": (0, 2, 3, 4),
    }
    for name, indices in subsets.items():
        if selected is not None and name not in selected:
            continue
        summary = results["models"][best_architecture] if name == "all_features" else _run_seeds(
            data_root, best_architecture, indices, config["epochs"], config["batch_size"],
            config["hidden_channels"], config["num_layers"], config["dropout"],
            config["learning_rate"])[0]
        results["ablations"][name] = summary | {"feature_names": [FEATURE_NAMES[i] for i in indices]}
    _write_outputs(results, output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--architecture", choices=("all", "gcn", "sage", "gat"), default="all")
    parser.add_argument("--ablation", choices=("all_features", "no_loop_depth", "no_move_related",
                                                 "no_live_range_length"))
    arguments = parser.parse_args()
    if arguments.ablation_only:
        selected = (arguments.ablation,) if arguments.ablation else None
        results = run_feature_ablation(arguments.data_root, arguments.output_dir, selected)
    else:
        architectures = ("gcn", "sage", "gat") if arguments.architecture == "all" else (arguments.architecture,)
        results = run_model_comparison(arguments.data_root, arguments.output_dir, arguments.epochs,
                                       arguments.batch_size, include_ablation=not arguments.skip_ablation,
                                       architectures=architectures, append=arguments.architecture != "all")
    for name, summary in results["models"].items():
        print(f"{name.upper()}: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")


if __name__ == "__main__":
    main()
