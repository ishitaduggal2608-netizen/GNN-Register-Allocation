"""
Training Pipeline Module for GNN Register Allocator.
Manages epoch optimization, metric computation, class imbalance weighting, and model checkpoint saving/loading
across all relational architectures (R-GCN, R-GAT, R-SAGE, R-GIN).
"""

from typing import Dict, Any, List, Optional, Union
import os
import pickle
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import InterferenceGraphDataset, GraphDataSample
from models.base_allocator import GraphColoringLoss
from models.gnn_allocator import RelationalGNNRegisterAllocator


def create_model_by_type(
    model_type: str = "rgcn",
    in_channels: int = 6,
    hidden_dim: int = 64,
    num_registers: int = 4,
    dropout: float = 0.1
) -> nn.Module:
    """Factory creating relational allocator models by architecture key."""
    m = model_type.lower()
    if m in ("rgat",):
        from models.rgat_allocator import RelationalGATRegisterAllocator
        model = RelationalGATRegisterAllocator(
            in_node_features=in_channels, hidden_dim=hidden_dim, num_registers=num_registers, dropout=dropout
        )
        model.architecture = "rgat"
    elif m in ("rsage",):
        from models.rsage_allocator import RelationalSAGERegisterAllocator
        model = RelationalSAGERegisterAllocator(
            in_node_features=in_channels, hidden_dim=hidden_dim, num_registers=num_registers, dropout=dropout
        )
        model.architecture = "rsage"
    elif m in ("gin", "rgin"):
        from models.gin_allocator import RelationalGINRegisterAllocator
        model = RelationalGINRegisterAllocator(
            in_node_features=in_channels, hidden_dim=hidden_dim, num_registers=num_registers, dropout=dropout
        )
        model.architecture = "gin"
    elif m in ("gat",):
        from models.gnn_allocator import GATSpillPredictor
        model = GATSpillPredictor(
            in_channels=in_channels, hidden_channels=hidden_dim, num_registers=num_registers, dropout=dropout
        )
        model.architecture = "gat"
    elif m in ("sage", "graphsage"):
        from models.gnn_allocator import SAGESpillPredictor
        model = SAGESpillPredictor(
            in_channels=in_channels, hidden_channels=hidden_dim, num_registers=num_registers, dropout=dropout
        )
        model.architecture = "sage"
    elif m in ("gcn",):
        from models.gnn_allocator import GCNSpillPredictor
        model = GCNSpillPredictor(
            in_channels=in_channels, hidden_channels=hidden_dim, num_registers=num_registers, dropout=dropout
        )
        model.architecture = "gcn"
    else:
        model = RelationalGNNRegisterAllocator(
            in_node_features=in_channels, hidden_dim=hidden_dim, num_registers=num_registers, dropout=dropout
        )
        model.architecture = "rgcn"

    return model


class GNNTrainer:
    """Trainer class for training and validating GNN Register Allocator models."""
    def __init__(
        self,
        model: Union[nn.Module, str] = "rgcn",
        dataset: Any = None,
        val_dataset: Optional[Any] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        use_pos_weight: bool = True,
        model_type: Optional[str] = None
    ):
        num_regs = getattr(dataset, "num_registers", 4) if dataset is not None else 4
        in_dim = getattr(dataset, "feature_dim", 6) if dataset is not None else 6

        # Support instantiating by model_type parameter or string model argument
        if isinstance(model, str):
            selected_type = model
            self.model = create_model_by_type(
                selected_type, in_channels=in_dim, hidden_dim=64, num_registers=num_regs
            ).to(device)
        elif model_type is not None and not isinstance(model, nn.Module):
            self.model = create_model_by_type(
                model_type, in_channels=in_dim, hidden_dim=64, num_registers=num_regs
            ).to(device)
        else:
            self.model = model.to(device)
            if model_type is not None:
                self.model.architecture = model_type

        self.dataset: Any = dataset
        self.val_dataset: Optional[Any] = val_dataset
        self.device: torch.device = torch.device(device)
        self.normalization_stats = getattr(dataset, "normalization_stats", None)
        if self.normalization_stats is not None:
            setattr(self.model, "normalization_stats", self.normalization_stats)

        pos_weight = None
        if use_pos_weight and hasattr(dataset, "get_pos_weight"):
            pos_weight = dataset.get_pos_weight().to(self.device)

        self.criterion = GraphColoringLoss(conflict_weight=0.5, coalesce_weight=0.3, pos_weight=pos_weight)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "ce_loss": [],
            "conflict_loss": [],
            "val_accuracy": [],
            "val_color_violations": []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Trains for one epoch across all dataset samples."""
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_conflict = 0.0

        for sample in self.dataset:
            x = sample.node_features.to(self.device)
            interf_adj = sample.interf_adj.to(self.device)
            coal_adj = sample.coal_adj.to(self.device)
            target_colors = sample.target_colors.to(self.device)
            coalesce_labels = sample.coalesce_labels.to(self.device)

            self.optimizer.zero_grad()
            color_logits, coalesce_logits = self.model(x, interf_adj, coal_adj)

            loss, loss_dict = self.criterion(
                color_logits, coalesce_logits, target_colors, interf_adj, coal_adj, coalesce_labels
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss_dict["total_loss"]
            total_ce += loss_dict["ce_loss"]
            total_conflict += loss_dict["conflict_loss"]

        self.scheduler.step()
        n_samples = max(len(self.dataset), 1)

        metrics = {
            "train_loss": total_loss / n_samples,
            "ce_loss": total_ce / n_samples,
            "conflict_loss": total_conflict / n_samples
        }

        self.history["train_loss"].append(metrics["train_loss"])
        self.history["ce_loss"].append(metrics["ce_loss"])
        self.history["conflict_loss"].append(metrics["conflict_loss"])

        return metrics

    def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
        """Evaluates model performance: accuracy and color conflict rate."""
        self.model.eval()
        target_ds = eval_dataset if eval_dataset is not None else self.val_dataset
        if target_ds is None or len(target_ds) == 0:
            return {"accuracy": 0.0, "violation_rate_pct": 0.0}

        correct = 0
        total = 0
        total_conflicts = 0
        total_graphs_with_conflicts = 0

        with torch.no_grad():
            for sample in target_ds:
                x = sample.node_features.to(self.device)
                interf_adj = sample.interf_adj.to(self.device)
                coal_adj = sample.coal_adj.to(self.device)
                target_colors = sample.target_colors.to(self.device)

                color_logits, _ = self.model(x, interf_adj, coal_adj)
                pred_colors = torch.argmax(color_logits, dim=-1)

                correct += (pred_colors == target_colors).sum().item()
                total += target_colors.size(0)

                # Count invalid coloring conflicts along interference edges
                num_registers = getattr(self.model, "num_registers", 4)
                has_graph_conflict = False
                N = target_colors.size(0)
                for i in range(N):
                    for j in range(i + 1, N):
                        if interf_adj[i, j] > 0:
                            ci, cj = pred_colors[i].item(), pred_colors[j].item()
                            if ci < num_registers and cj < num_registers and ci == cj:
                                total_conflicts += 1
                                has_graph_conflict = True

                if has_graph_conflict:
                    total_graphs_with_conflicts += 1

        accuracy = correct / max(total, 1)
        violation_rate = (total_graphs_with_conflicts / len(target_ds)) * 100.0

        return {
            "accuracy": accuracy,
            "violation_rate_pct": violation_rate,
            "total_conflicts": total_conflicts
        }

    def train(self, num_epochs: int = 20, verbose: bool = True) -> Dict[str, List[float]]:
        """Full training loop over num_epochs."""
        if verbose:
            arch = getattr(self.model, "architecture", "GNN").upper()
            print(f"Starting {arch} Training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()

            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_color_violations"].append(val_metrics["violation_rate_pct"])

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
                print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - "
                      f"Loss: {train_metrics['train_loss']:.4f} (CE: {train_metrics['ce_loss']:.4f}, Conflict: {train_metrics['conflict_loss']:.4f}) | "
                      f"Val Acc: {val_metrics['accuracy']*100:.1f}% | Color Violations: {val_metrics['violation_rate_pct']:.2f}%")

        return self.history

    def save_checkpoint(self, filepath: str = "gnn_allocator.pt"):
        """Saves model state dict with self-describing architecture metadata."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        arch = getattr(self.model, "architecture", "rgcn")
        torch.save({
            "architecture": arch,
            "model_state_dict": self.model.state_dict(),
            "in_node_features": getattr(self.model, "in_channels", getattr(self.model, "in_node_features", 6)),
            "in_channels": getattr(self.model, "in_channels", getattr(self.model, "in_node_features", 6)),
            "hidden_dim": getattr(self.model, "hidden_channels", getattr(self.model, "hidden_dim", 64)),
            "num_registers": getattr(self.model, "num_registers", 4),
            "normalization_stats": self.normalization_stats
        }, filepath)
        print(f"Saved GNN Model Checkpoint to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str = "gnn_allocator.pt", device: str = "cpu") -> nn.Module:
        """Loads GNN model from checkpoint, restoring matching architecture."""
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        except TypeError:
            warnings.warn(
                "This PyTorch version does not support weights_only=True; falling back to legacy torch.load. "
                "Only load checkpoints from trusted sources.",
                RuntimeWarning
            )
            checkpoint = torch.load(filepath, map_location=device)
        except pickle.UnpicklingError:
            warnings.warn(
                "weights_only=True could not load this legacy checkpoint; falling back to unsafe legacy loading. "
                "Only load checkpoints from trusted sources.",
                RuntimeWarning
            )
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        arch = str(checkpoint.get("architecture", "rgcn")).lower()
        num_regs = checkpoint.get("num_registers", 4)
        in_dim = checkpoint.get("in_channels", checkpoint.get("in_node_features", 6))
        hidden_dim = checkpoint.get("hidden_dim", 64)

        if arch in ("rgat",):
            from models.rgat_allocator import RelationalGATRegisterAllocator
            model = RelationalGATRegisterAllocator(in_node_features=in_dim, hidden_dim=hidden_dim, num_registers=num_regs)
        elif arch in ("rsage",):
            from models.rsage_allocator import RelationalSAGERegisterAllocator
            model = RelationalSAGERegisterAllocator(in_node_features=in_dim, hidden_dim=hidden_dim, num_registers=num_regs)
        elif arch in ("gin", "rgin"):
            from models.gin_allocator import RelationalGINRegisterAllocator
            model = RelationalGINRegisterAllocator(in_node_features=in_dim, hidden_dim=hidden_dim, num_registers=num_regs)
        elif arch in ("gcn",):
            from models.gnn_allocator import GCNSpillPredictor
            model = GCNSpillPredictor(in_channels=in_dim, hidden_channels=hidden_dim, num_registers=num_regs)
        elif arch in ("sage", "graphsage"):
            from models.gnn_allocator import SAGESpillPredictor
            model = SAGESpillPredictor(in_channels=in_dim, hidden_channels=hidden_dim, num_registers=num_regs)
        elif arch in ("gat",):
            from models.gnn_allocator import GATSpillPredictor
            model = GATSpillPredictor(in_channels=in_dim, hidden_channels=hidden_dim, num_registers=num_regs)
        else:
            model = RelationalGNNRegisterAllocator(
                in_node_features=in_dim,
                hidden_dim=hidden_dim,
                num_registers=num_regs
            )

        model.architecture = arch
        model.load_state_dict(checkpoint["model_state_dict"])
        if "normalization_stats" in checkpoint:
            setattr(model, "normalization_stats", checkpoint["normalization_stats"])
        model.to(device)
        model.eval()
        return model
