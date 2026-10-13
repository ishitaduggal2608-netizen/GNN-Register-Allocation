"""Comparable message-passing baselines for node-level spill prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class _SpillPredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least two")
        self.convs = torch.nn.ModuleList(self._make_convs(in_channels, hidden_channels, num_layers))
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def _make_convs(self, in_channels: int, hidden_channels: int, num_layers: int):
        raise NotImplementedError

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for convolution in self.convs:
            x = convolution(x, edge_index)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, edge_index)).squeeze(-1)


class GCNSpillPredictor(_SpillPredictor):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__(in_channels, hidden_channels, num_layers, dropout)

    def _make_convs(self, in_channels: int, hidden_channels: int, num_layers: int):
        return _stack_convolutions(GCNConv, in_channels, hidden_channels, num_layers)


class SAGESpillPredictor(_SpillPredictor):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__(in_channels, hidden_channels, num_layers, dropout)

    def _make_convs(self, in_channels: int, hidden_channels: int, num_layers: int):
        return _stack_convolutions(SAGEConv, in_channels, hidden_channels, num_layers)


class GATSpillPredictor(_SpillPredictor):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__(in_channels, hidden_channels, num_layers, dropout)

    def _make_convs(self, in_channels: int, hidden_channels: int, num_layers: int):
        channels = [in_channels] + [hidden_channels] * num_layers
        return [GATConv(channels[index], channels[index + 1], heads=4, concat=False)
                for index in range(num_layers)]


class GCNRegisterPredictor(GCNSpillPredictor):
    """Phase 5 baseline: register classes 0..15 plus fixed class 16 for SPILL."""

    def __init__(self, in_channels: int, max_registers: int = 16, hidden_channels: int = 64,
                 num_layers: int = 3, dropout: float = 0.3):
        self.max_registers = max_registers
        super().__init__(in_channels, hidden_channels, num_layers, dropout)
        self.classifier = torch.nn.Linear(hidden_channels, max_registers + 1)


def _stack_convolutions(convolution_type, in_channels: int, hidden_channels: int, num_layers: int):
    channels = [in_channels] + [hidden_channels] * num_layers
    return [convolution_type(channels[index], channels[index + 1]) for index in range(num_layers)]


def build_spill_predictor(architecture: str, in_channels: int, hidden_channels: int = 64,
                          num_layers: int = 3, dropout: float = 0.3) -> _SpillPredictor:
    models = {"gcn": GCNSpillPredictor, "sage": SAGESpillPredictor, "gat": GATSpillPredictor}
    try:
        model_type = models[architecture.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown architecture {architecture!r}; choose gcn, sage, or gat") from error
    return model_type(in_channels, hidden_channels, num_layers, dropout)


class CoalesceClassifier(torch.nn.Module):
    """Phase 8 move-pair head over frozen penultimate GNN embeddings."""

    def __init__(self, node_embed_dim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(node_embed_dim * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, move_edge_index: torch.Tensor) -> torch.Tensor:
        source, target = move_edge_index
        pair_features = torch.cat((node_embeddings[source], node_embeddings[target]), dim=-1)
        return self.mlp(pair_features).squeeze(-1)
