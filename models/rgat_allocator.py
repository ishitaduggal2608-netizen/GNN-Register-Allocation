"""
Relational Graph Attention Network (R-GAT) Allocator for Compiler Register Allocation.
Computes relation-specific attention-weighted aggregation over both interference
and move-coalescing edges.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_allocator import BaseRelationalAllocator


class RelationalGATConv(nn.Module):
    """
    Multi-Relational Graph Attention Layer:
    Computes distinct attention distributions over:
    1. Interference edges (hard register conflicts & neighbor pressure)
    2. Move-coalescing edges (affinity copy edges)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_interf = nn.Linear(in_features, out_features, bias=False)
        self.w_coal = nn.Linear(in_features, out_features, bias=False)

        # Attention vectors [2 * out_features, 1]
        self.a_interf = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.a_coal = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.last_interf_attention: Optional[torch.Tensor] = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_interf.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_coal.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.a_interf)
        nn.init.xavier_uniform_(self.a_coal)

    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N = x.size(0)
        h_self = self.w_self(x)

        # --- 1. Interference Attention ---
        h_interf_proj = self.w_interf(x)  # [N, H]
        h_i = h_interf_proj.unsqueeze(1).repeat(1, N, 1)  # [N, N, H]
        h_j = h_interf_proj.unsqueeze(0).repeat(N, 1, 1)  # [N, N, H]
        pair_interf = torch.cat([h_i, h_j], dim=-1)       # [N, N, 2H]

        interf_logits = F.leaky_relu(
            torch.matmul(pair_interf, self.a_interf).squeeze(-1),
            negative_slope=0.2
        )  # [N, N]

        # Mask non-interference pairs (allow self-loop for stability)
        interf_mask = (interf_adj + torch.eye(N, device=x.device)) == 0
        interf_logits = interf_logits.masked_fill(interf_mask, -1e9)
        interf_attn = F.softmax(interf_logits, dim=-1)
        interf_attn = F.dropout(interf_attn, p=self.dropout, training=self.training)
        self.last_interf_attention = interf_attn.detach()

        h_interf_agg = torch.matmul(interf_attn, h_interf_proj)

        # --- 2. Move-Coalescing Attention ---
        if coal_adj is not None and torch.sum(coal_adj) > 0:
            h_coal_proj = self.w_coal(x)
            h_i_c = h_coal_proj.unsqueeze(1).repeat(1, N, 1)
            h_j_c = h_coal_proj.unsqueeze(0).repeat(N, 1, 1)
            pair_coal = torch.cat([h_i_c, h_j_c], dim=-1)

            coal_logits = F.leaky_relu(
                torch.matmul(pair_coal, self.a_coal).squeeze(-1),
                negative_slope=0.2
            )

            # Mask nodes without coalescing edge
            coal_mask = (coal_adj == 0)
            coal_logits = coal_logits.masked_fill(coal_mask, -1e9)

            # Avoid NaN on rows with zero coalescing edges
            has_coal = (torch.sum(coal_adj, dim=-1) > 0)
            coal_attn = torch.zeros((N, N), device=x.device)
            if has_coal.any():
                coal_attn[has_coal] = F.softmax(coal_logits[has_coal], dim=-1)
                coal_attn = F.dropout(coal_attn, p=self.dropout, training=self.training)

            h_coal_agg = torch.matmul(coal_attn, h_coal_proj)
        else:
            h_coal_agg = torch.zeros_like(h_self)

        return h_self + h_interf_agg + h_coal_agg + self.bias


class RelationalGATRegisterAllocator(BaseRelationalAllocator):
    """
    Relational Graph Attention Network (R-GAT) Register Allocator.
    Combines multi-relational attention aggregation with bilinear coalescing scoring
    and conflict-aware register classification.
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
            RelationalGATConv(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        self.last_attention_weights: Optional[torch.Tensor] = None

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

        self.last_attention_weights = self.convs[-1].last_interf_attention

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
RGATRegisterAllocator = RelationalGATRegisterAllocator
