"""
Relational Pattern Detection Agent (Agent 1).
Wraps the Relational Graph Convolutional Network (R-GCN) architecture.
Specializes in detecting relational structures: hard interference edges (must differ)
versus affinity move-coalescing edges (prefer same physical register).
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import torch
import torch.nn as nn

from models.gnn_allocator import RelationalGNNRegisterAllocator
from compiler.interference_graph import InterferenceGraph
from .base import PatternAgent, AgentPrediction


class RelationalAgent(PatternAgent):
    """
    Relational Agent (R-GCN):
    Owns move-coalescing decisions and relational pattern detection.
    Consumes both interference and move edges via multi-relational graph convolutions.
    """
    OWNER: str = "M1_RELATIONAL"

    def __init__(
        self,
        num_registers: int = 4,
        in_channels: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(
            name="RelationalAgent",
            architecture="rgcn",
            specialty="move-chains and coalescing opportunities",
            num_registers=num_registers,
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = RelationalGNNRegisterAllocator(
            in_node_features=in_channels,
            hidden_dim=hidden_dim,
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
        """
        Executes multi-relational forward pass using both interference and coalescing adjacencies.
        """
        if coal_adj is None:
            coal_adj = torch.zeros_like(interf_adj)
        return self.model(x, interf_adj, coal_adj)

    def explain(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None,
        ig: Optional[InterferenceGraph] = None
    ) -> Dict[str, Any]:
        """
        Explains relational patterns:
        - Move chains (connected components in the move-coalescing graph)
        - Learned coalescing scores for each candidate move pair
        - Edge density ratio (interference vs coalescing)
        - Potential coalescing candidates ranking
        """
        N = x.size(0)
        var_names = [f"v{i}" for i in range(N)]
        if ig is not None:
            var_names = [v.name for v in ig.variables]

        if coal_adj is None:
            coal_adj = torch.zeros((N, N), device=x.device)

        # Build NetworkX graph for move edges to identify move chains
        coal_nx = nx.Graph()
        for i, name in enumerate(var_names):
            coal_nx.add_node(name)

        move_edges_idx = []
        for i in range(N):
            for j in range(i + 1, N):
                if coal_adj[i, j] > 0.5:
                    coal_nx.add_edge(var_names[i], var_names[j])
                    move_edges_idx.append((i, j))

        # Detect move chains (components of size >= 2)
        move_chains = [
            sorted(list(c)) for c in nx.connected_components(coal_nx) if len(c) >= 2
        ]

        # Calculate learned coalescing scores
        with torch.no_grad():
            _, coalesce_logits = self.model(x, interf_adj, coal_adj)
            coalesce_probs = torch.sigmoid(coalesce_logits)

        move_candidates = []
        for i, j in move_edges_idx:
            score = float(coalesce_probs[i, j].item())
            interferes = bool(interf_adj[i, j] > 0.5)
            move_candidates.append({
                "var_u": var_names[i],
                "var_v": var_names[j],
                "coalesce_score": round(score, 4),
                "interferes": interferes,
                "safe_heuristic": not interferes and score > 0.5
            })
        move_candidates.sort(key=lambda d: d["coalesce_score"], reverse=True)

        num_interf = int(torch.sum(interf_adj > 0.5).item()) // 2
        num_coal = len(move_edges_idx)

        return {
            "agent": self.name,
            "architecture": self.architecture,
            "specialty": self.specialty,
            "owner": self.OWNER,
            "num_interference_edges": num_interf,
            "num_coalesce_edges": num_coal,
            "relational_edge_ratio": round(num_coal / max(num_interf, 1), 3),
            "move_chains_detected": move_chains,
            "move_chain_count": len(move_chains),
            "top_coalescing_candidates": move_candidates[:5]
        }
