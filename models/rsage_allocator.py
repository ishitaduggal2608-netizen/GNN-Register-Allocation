"""
Relational GraphSAGE (R-SAGE) Allocator for Compiler Register Allocation.
Computes neighborhood mean/pool aggregation separately over interference and
move-coalescing relations before combining with target node representations.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_allocator import BaseRelationalAllocator


class RelationalSAGEConv(nn.Module):
    """
    Multi-Relational GraphSAGE Layer:
    Performs normalized mean aggregation across:
    1. Interference neighborhood (competition for physical registers)
    2. Coalescing neighborhood (affinity to share same physical register)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_interf = nn.Linear(in_features, out_features, bias=False)
        self.w_coal = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_interf.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_coal.weight, a=math.sqrt(5))

    def _mean_aggregate(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Row-normalized mean neighbor aggregation: D^{-1} A X."""
        deg = torch.sum(adj, dim=-1, keepdim=True)
        # Avoid division by zero for isolated nodes
        norm_adj = torch.where(deg > 0, adj / deg.clamp(min=1.0), torch.zeros_like(adj))
        return torch.matmul(norm_adj, x)

    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h_self = self.w_self(x)

        # 1. Interference mean aggregation
        mean_interf = self._mean_aggregate(interf_adj, x)
        h_interf = self.w_interf(mean_interf)

        # 2. Coalescing mean aggregation
        if coal_adj is not None and torch.sum(coal_adj) > 0:
            mean_coal = self._mean_aggregate(coal_adj, x)
            h_coal = self.w_coal(mean_coal)
        else:
            h_coal = torch.zeros_like(h_self)

        return h_self + h_interf + h_coal + self.bias


class RelationalSAGERegisterAllocator(BaseRelationalAllocator):
    """
    Relational GraphSAGE (R-SAGE) Register Allocator.
    Combines relational inductive neighborhood aggregation with bilinear move scoring
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
            RelationalSAGEConv(hidden_dim, hidden_dim)
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


# Convenience alias
RSAGERegisterAllocator = RelationalSAGERegisterAllocator
