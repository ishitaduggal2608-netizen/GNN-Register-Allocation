"""
Relation-Aware Graph Isomorphism Network (R-GIN) Allocator for Compiler Register Allocation.
Utilizes sum-aggregation with relation-aware MLP updates to provide maximally discriminative
structural representation of graph topology, interference cliques, and move affinity.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_allocator import BaseRelationalAllocator


class RelationalGINConv(nn.Module):
    """
    Multi-Relational Graph Isomorphism Network Layer:
    Performs sum-aggregation separately for interference and coalescing edge types,
    followed by a non-linear Multi-Layer Perceptron (MLP) update:
    h^{(l+1)} = MLP( (1 + eps_interf) * h + sum_{v in N_interf} h_v + gamma_coal * sum_{u in N_coal} h_u )
    """
    def __init__(self, in_features: int, out_features: int, learn_eps: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
            self.coal_weight = nn.Parameter(torch.ones(1) * 0.5)
        else:
            self.register_buffer("eps", torch.zeros(1))
            self.register_buffer("coal_weight", torch.ones(1) * 0.5)

        # 2-layer MLP mapping sum-aggregated features to output representation
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Sum-aggregation over interference graph
        sum_interf = torch.matmul(interf_adj, x)

        # Sum-aggregation over coalescing graph
        if coal_adj is not None and torch.sum(coal_adj) > 0:
            sum_coal = torch.matmul(coal_adj, x)
        else:
            sum_coal = torch.zeros_like(x)

        # Relation-weighted sum combination
        aggregated = (1.0 + self.eps) * x + sum_interf + self.coal_weight * sum_coal

        return self.mlp(aggregated)


class RelationalGINRegisterAllocator(BaseRelationalAllocator):
    """
    Relation-Aware Graph Isomorphism Network (R-GIN) Register Allocator.
    Combines sum-aggregation MLP layers with bilinear move-coalescing scoring
    and conflict-aware register allocation.
    """
    def __init__(
        self,
        in_node_features: int = 6,
        hidden_dim: int = 64,
        num_registers: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        in_channels: Optional[int] = None
    ):
        super().__init__(
            in_node_features=in_node_features,
            hidden_dim=hidden_dim,
            num_registers=num_registers,
            dropout=dropout,
            in_channels=in_channels
        )
        self.num_layers = num_layers

        self.convs = nn.ModuleList([
            RelationalGINConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if coal_adj is None:
            coal_adj = torch.zeros_like(interf_adj)

        h = self.encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = conv(h, interf_adj, coal_adj)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + h_in  # Residual skip connection

        color_logits = self.color_head(h)
        coalesce_logits = self.compute_coalesce_logits(h)

        return color_logits, coalesce_logits

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if coal_adj is None:
            coal_adj = torch.zeros_like(interf_adj)

        h = self.encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = conv(h, interf_adj, coal_adj)
            h = bn(h)
            h = F.relu(h)
            h = h + h_in
        return h


# Convenience aliases
GINRegisterAllocator = RelationalGINRegisterAllocator
RGINRegisterAllocator = RelationalGINRegisterAllocator
