"""
Pressure Pattern Detection Agent (Agent 4).
Wraps the Graph Convolutional Network (GCN) architecture with symmetric Laplacian normalization.
Specializes in detecting global register pressure around dense interference clusters
and loop-hot live ranges.
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import torch
import torch.nn as nn

from models.gnn_allocator import GCNSpillPredictor
from compiler.interference_graph import InterferenceGraph
from .base import PatternAgent, AgentPrediction


class PressureAgent(PatternAgent):
    """
    Pressure Agent (GCN):
    Detects global register pressure around dense clusters and loop-hot variables.
    Analyzes multi-hop graph diffusion to spot regions where chromatic demand exceeds K.
    """
    OWNER: str = "M4_PRESSURE"

    def __init__(
        self,
        num_registers: int = 4,
        in_channels: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__(
            name="PressureAgent",
            architecture="gcn",
            specialty="global register pressure around dense clusters and loop-hot nodes",
            num_registers=num_registers,
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = GCNSpillPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_registers=num_registers,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """GCN forward pass."""
        return self.model(x, interf_adj, coal_adj)

    def explain(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None,
        ig: Optional[InterferenceGraph] = None
    ) -> Dict[str, Any]:
        """
        Explains pressure patterns:
        - Loop-hot nodes (variables residing inside loops)
        - Local chromatic density index (degree / K)
        - Global pressure peak estimation
        - High spill-cost nodes subject to heavy interference
        """
        N = x.size(0)
        var_names = [f"v{i}" for i in range(N)]
        if ig is not None:
            var_names = [v.name for v in ig.variables]

        degrees = torch.sum(interf_adj > 0.5, dim=-1)  # [N]
        pressure_indices = (degrees / float(self.num_registers)).tolist()

        # Check loop depth (feature column 1) and spill cost (column 0) if available
        loop_depths = [0] * N
        spill_costs = [1.0] * N
        if ig is not None:
            features = ig.get_node_features()
            for i, name in enumerate(var_names):
                if name in features:
                    loop_depths[i] = features[name].loop_depth
                    spill_costs[i] = features[name].spill_cost

        loop_hot_nodes = [
            {"var_name": var_names[i], "loop_depth": loop_depths[i], "degree": int(degrees[i].item())}
            for i in range(N) if loop_depths[i] > 0
        ]
        loop_hot_nodes.sort(key=lambda d: (d["loop_depth"], d["degree"]), reverse=True)

        highest_pressure_nodes = []
        for i in torch.argsort(degrees, descending=True)[:5].tolist():
            highest_pressure_nodes.append({
                "var_name": var_names[i],
                "pressure_ratio": round(pressure_indices[i], 2),
                "loop_depth": loop_depths[i],
                "spill_cost": round(spill_costs[i], 2),
                "is_choking": pressure_indices[i] >= 1.0
            })

        max_deg = int(torch.max(degrees).item()) if N > 0 else 0
        global_pressure_peak = round(max_deg / float(self.num_registers), 2)

        return {
            "agent": self.name,
            "architecture": self.architecture,
            "specialty": self.specialty,
            "owner": self.OWNER,
            "global_pressure_peak": global_pressure_peak,
            "loop_hot_node_count": len(loop_hot_nodes),
            "loop_hot_nodes": loop_hot_nodes[:5],
            "highest_pressure_nodes": highest_pressure_nodes
        }
