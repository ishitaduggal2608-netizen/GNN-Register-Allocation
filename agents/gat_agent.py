"""
Attention Pattern Detection Agent (Agent 2).
Wraps the Graph Attention Network (GAT) architecture.
Specializes in detecting which interfering neighbours exert the highest register pressure,
pushing specific nodes toward stack spilling.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn

from models.gnn_allocator import GATSpillPredictor
from compiler.interference_graph import InterferenceGraph
from .base import PatternAgent, AgentPrediction


class AttentionAgent(PatternAgent):
    """
    Attention Agent (GAT):
    Detects which interfering neighbours push a node to spill.
    Exposes last-layer attention weights through `last_attention_weights`
    and produces neighbour-pressure explanations.
    """
    OWNER: str = "M2_ATTENTION"

    def __init__(
        self,
        num_registers: int = 4,
        in_channels: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__(
            name="AttentionAgent",
            architecture="gat",
            specialty="which interfering neighbours push a node to spill",
            num_registers=num_registers,
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = GATSpillPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_registers=num_registers,
            num_layers=num_layers,
            dropout=dropout
        )

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        """Exposes the last layer attention weights [N, N]."""
        return getattr(self.model, "last_attention_weights", None)

    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """GAT forward pass."""
        return self.model(x, interf_adj, coal_adj)

    def explain(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None,
        ig: Optional[InterferenceGraph] = None
    ) -> Dict[str, Any]:
        """
        Explains attention patterns:
        - Top-3 attended interfering neighbours per node
        - Nodes experiencing the highest inward attention pressure
        - Sparsity and peak attention scores
        """
        N = x.size(0)
        var_names = [f"v{i}" for i in range(N)]
        if ig is not None:
            var_names = [v.name for v in ig.variables]

        # Trigger forward pass if attention weights not yet computed
        if self.last_attention_weights is None or self.last_attention_weights.size(0) != N:
            with torch.no_grad():
                self.model(x, interf_adj, coal_adj)

        attn = self.last_attention_weights
        if attn is None:
            attn = torch.zeros((N, N), device=x.device)

        node_attention_map: Dict[str, List[Dict[str, Any]]] = {}
        inward_pressure = torch.sum(attn * interf_adj, dim=0)  # [N] column sum = incoming attention

        for i in range(N):
            name_i = var_names[i]
            nbr_indices = torch.nonzero(interf_adj[i] > 0.5, as_tuple=False).squeeze(-1).tolist()
            if isinstance(nbr_indices, int):
                nbr_indices = [nbr_indices]

            scored_nbrs = []
            for j in nbr_indices:
                if j < N:
                    score = float(attn[i, j].item())
                    scored_nbrs.append({
                        "neighbor": var_names[j],
                        "attention_weight": round(score, 4)
                    })
            scored_nbrs.sort(key=lambda item: item["attention_weight"], reverse=True)
            node_attention_map[name_i] = scored_nbrs[:3]

        top_pressure_nodes = []
        for i in torch.argsort(inward_pressure, descending=True)[:5].tolist():
            top_pressure_nodes.append({
                "var_name": var_names[i],
                "inward_pressure_score": round(float(inward_pressure[i].item()), 4),
                "interference_degree": int(torch.sum(interf_adj[i] > 0.5).item())
            })

        return {
            "agent": self.name,
            "architecture": self.architecture,
            "specialty": self.specialty,
            "owner": self.OWNER,
            "top_attended_neighbors": node_attention_map,
            "highest_pressure_nodes": top_pressure_nodes,
            "attention_matrix_shape": list(attn.shape)
        }
