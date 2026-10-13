"""PyG datasets and train-split-only normalization for register-allocation graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch_geometric.data import Data, Dataset

MAX_REGISTERS = 16

@dataclass(frozen=True)
class FeatureNormalizer:
    """Per-feature standardizer fitted exclusively on training graph nodes."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, feature_tensors: Sequence[torch.Tensor]) -> "FeatureNormalizer":
        if not feature_tensors:
            raise ValueError("Cannot fit a normalizer on an empty training split")
        features = torch.cat(feature_tensors, dim=0).float()
        mean = features.mean(dim=0)
        # Avoid NaNs for a one-node training fixture and division by zero for constants.
        std = features.std(dim=0, unbiased=False).clamp_min(1e-8)
        return cls(mean=mean, std=std)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}


class RegAllocDataset(Dataset):
    """Loads one exported allocation graph per JSON file into a PyG ``Data`` object."""

    def __init__(self, json_dir: str | Path, normalizer: FeatureNormalizer | None = None,
                 feature_indices: Sequence[int] | None = None):
        self.json_dir = Path(json_dir)
        if not self.json_dir.is_dir():
            raise FileNotFoundError(f"JSON graph directory does not exist: {self.json_dir}")
        self.files = sorted(self.json_dir.glob("*.json"))
        self.normalizer = normalizer
        self.feature_indices = tuple(feature_indices) if feature_indices is not None else None
        super().__init__(root=None)

    def len(self) -> int:
        return len(self.files)

    def raw_node_features(self, idx: int) -> torch.Tensor:
        graph = self._read_graph(idx)
        return self._node_features(graph, self.feature_indices)

    def fit_normalizer(self) -> FeatureNormalizer:
        return FeatureNormalizer.fit([self.raw_node_features(index) for index in range(self.len())])

    def get(self, idx: int) -> Data:
        graph = self._read_graph(idx)
        nodes = sorted(graph["nodes"], key=lambda node: node["id"])
        node_features = self._node_features(graph, self.feature_indices)
        if self.normalizer is not None:
            node_features = self.normalizer.transform(node_features)

        node_positions = {node["id"]: position for position, node in enumerate(nodes)}
        directed_edges = [
            [node_positions[source], node_positions[target]]
            for source, target in graph["interference_edges"]
        ]
        if directed_edges:
            edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        y_spill = torch.tensor(
            [1.0 if node["label_spill"] else 0.0 for node in nodes], dtype=torch.float
        )
        y_register = torch.tensor(
            [node["label_register"] if node["label_register"] is not None else graph["num_registers"]
             for node in nodes],
            dtype=torch.long,
        )
        # For one model across k={4,8,16}, use one fixed spill class (16).
        # ``y_register`` above remains the JSON-faithful target for analysis.
        y_allocation = torch.tensor(
            [node["label_register"] if node["label_register"] is not None else MAX_REGISTERS
             for node in nodes], dtype=torch.long,
        )
        register_budget = torch.full((len(nodes),), int(graph["num_registers"]), dtype=torch.long)
        move_pairs = [
            [node_positions[source], node_positions[target]]
            for move in graph["move_edges"]
            for source, target in [move["pair"]]
        ]
        if move_pairs:
            move_edge_index = torch.tensor(move_pairs, dtype=torch.long).t().contiguous()
            y_coalesce = torch.tensor(
                [1.0 if move["coalesce_safe"] else 0.0 for move in graph["move_edges"]],
                dtype=torch.float,
            )
        else:
            move_edge_index = torch.empty((2, 0), dtype=torch.long)
            y_coalesce = torch.empty((0,), dtype=torch.float)
        data = Data(x=node_features, edge_index=edge_index, y_spill=y_spill, y_register=y_register)
        data.y_allocation = y_allocation
        data.register_budget = register_budget
        data.move_edge_index = move_edge_index
        data.y_coalesce = y_coalesce
        data.num_registers = int(graph["num_registers"])
        data.graph_id = graph["graph_id"]
        return data

    def _read_graph(self, idx: int) -> dict:
        with self.files[idx].open(encoding="utf-8") as graph_file:
            return json.load(graph_file)

    @staticmethod
    def _node_features(graph: dict, feature_indices: Sequence[int] | None = None) -> torch.Tensor:
        nodes = sorted(graph["nodes"], key=lambda node: node["id"])
        features = torch.tensor(
            [
                [
                    node["degree"],
                    node["live_range_length"],
                    node["loop_depth"],
                    node["use_def_count"],
                    float(node["is_move_related"]),
                ]
                for node in nodes
            ],
            dtype=torch.float,
        )
        return features if feature_indices is None else features[:, list(feature_indices)]
