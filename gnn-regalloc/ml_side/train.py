"""Train a spill classifier on graph-disjoint JSON dataset splits."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch_geometric.loader import DataLoader

try:  # Supports both ``python -m ml_side.train`` and ``python ml_side/train.py``.
    from .dataset import MAX_REGISTERS, FeatureNormalizer, RegAllocDataset
    from .models import GCNRegisterPredictor, build_spill_predictor
except ImportError:
    from dataset import MAX_REGISTERS, FeatureNormalizer, RegAllocDataset
    from models import GCNRegisterPredictor, build_spill_predictor


def compute_pos_weight(dataset: RegAllocDataset, device: torch.device) -> torch.Tensor:
    positives = sum(float(dataset[index].y_spill.sum()) for index in range(len(dataset)))
    total = sum(int(dataset[index].num_nodes) for index in range(len(dataset)))
    negatives = total - positives
    # A no-spill training split has no meaningful positive weighting; avoid divide-by-zero.
    weight = 1.0 if positives == 0 else negatives / positives
    return torch.tensor(weight, dtype=torch.float, device=device)


def train_epoch(model, loader, optimizer, pos_weight: torch.Tensor, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch.y_spill, pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def mask_invalid_register_logits(logits: torch.Tensor, register_budget: torch.Tensor,
                                 max_registers: int = MAX_REGISTERS) -> torch.Tensor:
    """Disallow allocation to registers not present in a graph; retain the spill class."""
    register_ids = torch.arange(max_registers, device=logits.device).unsqueeze(0)
    invalid_register = register_ids >= register_budget.unsqueeze(1)
    masked = logits.clone()
    masked[:, :max_registers] = masked[:, :max_registers].masked_fill(invalid_register, -torch.inf)
    return masked


def train_register_epoch(model, loader, optimizer, device: torch.device) -> float:
    """Phase 5 multi-class allocation training with a fixed spill class."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = mask_invalid_register_logits(model(batch.x, batch.edge_index), batch.register_budget)
        loss = torch.nn.functional.cross_entropy(logits, batch.y_allocation)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_register_predictor(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        prediction = mask_invalid_register_logits(model(batch.x, batch.edge_index), batch.register_budget).argmax(dim=-1)
        correct += int((prediction == batch.y_allocation).sum())
        total += batch.y_allocation.numel()
    return {"allocation_accuracy": 0.0 if total == 0 else correct / total}


def run_register_assignment_training(train_dir: str | Path, val_dir: str | Path, test_dir: str | Path,
                                     epochs: int = 30, batch_size: int = 16, learning_rate: float = 1e-3,
                                     seed: int = 42):
    """Train the Phase 5 GCN multi-class register/spill baseline."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_train = RegAllocDataset(train_dir)
    normalizer = raw_train.fit_normalizer()
    train_dataset = RegAllocDataset(train_dir, normalizer)
    val_dataset = RegAllocDataset(val_dir, normalizer)
    test_dataset = RegAllocDataset(test_dir, normalizer)
    model = GCNRegisterPredictor(5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    best_state, best_validation = None, -1.0
    for _ in range(epochs):
        train_register_epoch(model, train_loader, optimizer, device)
        validation = evaluate_register_predictor(model, val_loader, device)["allocation_accuracy"]
        if validation > best_validation:
            best_validation, best_state = validation, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, normalizer, {"best_val_allocation_accuracy": best_validation,
                               **evaluate_register_predictor(model, test_loader, device)}


@torch.no_grad()
def evaluate(model, loader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    all_predictions, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        probabilities = torch.sigmoid(model(batch.x, batch.edge_index))
        all_predictions.append((probabilities > threshold).float().cpu())
        all_labels.append(batch.y_spill.cpu())
    predictions = torch.cat(all_predictions).numpy()
    labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def run_training(train_dir: str | Path, val_dir: str | Path, test_dir: str | Path,
                 architecture: str = "gcn", hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.3, learning_rate: float = 1e-3, epochs: int = 100,
                 batch_size: int = 16, seed: int = 42, checkpoint: str | Path | None = None,
                 feature_indices: tuple[int, ...] | None = None, verbose: bool = True,
                 normalizer: FeatureNormalizer | None = None):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_train = RegAllocDataset(train_dir, feature_indices=feature_indices)
    normalizer = normalizer or raw_train.fit_normalizer()  # Fit once: validation/test never influence it.
    train_dataset = RegAllocDataset(train_dir, normalizer, feature_indices)
    val_dataset = RegAllocDataset(val_dir, normalizer, feature_indices)
    test_dataset = RegAllocDataset(test_dir, normalizer, feature_indices)
    if not len(train_dataset) or not len(val_dataset) or not len(test_dataset):
        raise ValueError("Each graph-disjoint split must contain at least one JSON graph")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = build_spill_predictor(architecture, in_channels=train_dataset[0].num_node_features,
                                  hidden_channels=hidden_channels,
                                  num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = compute_pos_weight(train_dataset, device)

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        validation = evaluate(model, val_loader, device)
        if validation["f1"] > best_f1:
            best_f1 = validation["f1"]
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"epoch={epoch:03d} loss={loss:.4f} val_f1={validation['f1']:.4f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    metrics = {"best_val_f1": best_f1, **{f"test_{key}": value for key, value in test_metrics.items()}}
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "architecture": architecture,
            "in_channels": train_dataset[0].num_node_features,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_indices": feature_indices,
            "normalizer": normalizer.state_dict(),
            "metrics": metrics,
        }, checkpoint_path)
    return model, normalizer, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--architecture", choices=("gcn", "sage", "gat"), default="gcn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/gcn_spill.pt"))
    arguments = parser.parse_args()
    _, _, metrics = run_training(
        arguments.data_root / "train", arguments.data_root / "val", arguments.data_root / "test",
        architecture=arguments.architecture, epochs=arguments.epochs,
        batch_size=arguments.batch_size, checkpoint=arguments.checkpoint,
    )
    print(metrics)


if __name__ == "__main__":
    main()
