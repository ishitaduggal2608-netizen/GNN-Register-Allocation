"""
Neighbourhood Pattern Detection Agent (Agent 3).
Wraps the GraphSAGE architecture with mean aggregation.
Specializes in detecting local neighbourhood structures, hub nodes,
and hard-to-color dense subgraphs (e.g. (K+1)-cliques that force spills).
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import torch
import torch.nn as nn

from models.gnn_allocator import SAGESpillPredictor
from compiler.interference_graph import InterferenceGraph
from .base import PatternAgent, AgentPrediction


class NeighbourhoodAgent(PatternAgent):
    """
    Neighbourhood Agent (GraphSAGE):
    Detects local neighbourhood topology, hub nodes, and hard-to-color regions.
    Excels in inductive generalisation to graphs larger than those in training.
    """
    OWNER: str = "M3_STRUCTURE"

    def __init__(
        self,
        num_registers: int = 4,
        in_channels: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__(
            name="NeighbourhoodAgent",
            architecture="sage",
            specialty="local neighbourhood structure and hub nodes",
            num_registers=num_registers,
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = SAGESpillPredictor(
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
        """GraphSAGE forward pass."""
        return self.model(x, interf_adj, coal_adj)

    def explain(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None,
        ig: Optional[InterferenceGraph] = None
    ) -> Dict[str, Any]:
        """
        Explains neighbourhood structural patterns:
        - Hub nodes (high interference degree)
        - Cliques >= K (theoretically forcing spills)
        - Core numbers (k-core decomposition)
        - Average local clustering coefficients
        """
        N = x.size(0)
        var_names = [f"v{i}" for i in range(N)]
        if ig is not None:
            var_names = [v.name for v in ig.variables]

        # Construct NetworkX graph from interference adjacency
        g = nx.Graph()
        for i, name in enumerate(var_names):
            g.add_node(name)

        for i in range(N):
            for j in range(i + 1, N):
                if interf_adj[i, j] > 0.5:
                    g.add_edge(var_names[i], var_names[j])

        # Hub node detection
        degrees = dict(g.degree())
        sorted_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        hub_nodes = [
            {"var_name": node, "degree": deg, "is_critical": deg >= self.num_registers}
            for node, deg in sorted_hubs[:5]
        ]

        # Maximal cliques of size >= K
        cliques = list(nx.find_cliques(g))
        forcing_cliques = [
            sorted(c) for c in cliques if len(c) >= self.num_registers
        ]

        # K-core decomposition
        core_numbers = nx.core_number(g) if len(g) > 0 else {}
        max_core = max(core_numbers.values()) if core_numbers else 0

        # Local clustering coefficient
        clustering = nx.clustering(g) if len(g) > 0 else {}
        avg_clustering = sum(clustering.values()) / max(len(clustering), 1)

        return {
            "agent": self.name,
            "architecture": self.architecture,
            "specialty": self.specialty,
            "owner": self.OWNER,
            "hub_nodes": hub_nodes,
            "forcing_cliques_count": len(forcing_cliques),
            "forcing_cliques": forcing_cliques[:5],
            "max_k_core": max_core,
            "avg_clustering_coefficient": round(avg_clustering, 4)
        }
