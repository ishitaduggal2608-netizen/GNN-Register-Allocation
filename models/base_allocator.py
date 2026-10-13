"""
Base Class and Shared Utilities for Multi-Relational Register Allocation Models.
Provides BaseRelationalAllocator with shared node encoders, classification heads,
coalescing bilinear scoring, and GraphColoringLoss.
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRelationalAllocator(nn.Module):
    """
    Abstract Base Class for Multi-Relational Compiler Register Allocation Models.
    Unifies:
    1. Node feature encoder [in_features -> hidden_dim]
    2. Register assignment classifier head [hidden_dim -> num_classes (K registers + 1 spill)]
    3. Bilinear coalescing edge scorer [hidden_dim x hidden_dim -> N x N logits]
    """
    def __init__(
        self,
        in_node_features: int = 6,
        hidden_dim: int = 64,
        num_registers: int = 4,
        dropout: float = 0.1,
        in_channels: Optional[int] = None
    ):
        super().__init__()
        if in_channels is not None:
            in_node_features = in_channels

        self.in_node_features: int = in_node_features
        self.in_channels: int = in_node_features
        self.hidden_dim: int = hidden_dim
        self.hidden_channels: int = hidden_dim
        self.num_registers: int = num_registers
        self.num_classes: int = num_registers + 1  # K physical registers + 1 spill class
        self.dropout_rate: float = dropout

        # Shared 2-layer MLP Encoder with BatchNorm
        self.encoder = nn.Sequential(
            nn.Linear(in_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # Register Color Classifier Head
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )

        # Bilinear Coalescing Edge Scorer Head: score(u, v) = h_u^T * W_edge * h_v
        self.edge_w = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_w)

        self.dropout = nn.Dropout(dropout)

    def compute_coalesce_logits(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes pair-wise move-coalescing affinity logits for all node pairs [N, N].
        h: [N, hidden_dim]
        """
        h_transform = torch.matmul(h, self.edge_w)  # [N, hidden_dim]
        return torch.matmul(h_transform, h.T)       # [N, N]

    def _normalize_symmetric_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetrically normalizes adjacency matrix: D^{-1/2} A D^{-1/2}."""
        deg = torch.sum(adj, dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        deg_mat = torch.diag_embed(deg_inv_sqrt)
        return torch.matmul(torch.matmul(deg_mat, adj), deg_mat)

    def _normalize_mean_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Row-normalizes adjacency matrix: D^{-1} A."""
        deg = torch.sum(adj, dim=-1, keepdim=True).clamp(min=1.0)
        return adj / deg

    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Must be implemented by subclasses.
        Returns:
            color_logits: [N, K+1]
            coalesce_logits: [N, N]
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns node embeddings [N, hidden_dim] prior to the classification head.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_node_embeddings()")


class GraphColoringLoss(nn.Module):
    """
    Differentiable Graph Coloring Loss Function for Compiler Register Allocation:
    1. Cross-Entropy Loss on Ground Truth Register Assignment / Spill
    2. Soft Interference Color Conflict Loss: penalizes adjacent nodes sharing register color
    3. Move-Coalescing BCE Loss: encourages move-connected nodes to share register assignment
    """
    def __init__(
        self,
        conflict_weight: float = 0.5,
        coalesce_weight: float = 0.3,
        pos_weight: Optional[torch.Tensor] = None
    ):
        super(GraphColoringLoss, self).__init__()
        self.conflict_weight: float = conflict_weight
        self.coalesce_weight: float = coalesce_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        color_logits: torch.Tensor,       # [N, K+1]
        coalesce_logits: torch.Tensor,    # [N, N]
        target_colors: torch.Tensor,      # [N]
        interf_adj: torch.Tensor,         # [N, N]
        coal_adj: torch.Tensor,           # [N, N]
        coalesce_labels: torch.Tensor     # [N, N]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self.ce_loss(color_logits, target_colors)

        probs = F.softmax(color_logits, dim=-1)  # [N, K+1]
        reg_probs = probs[:, :-1]                 # [N, K] - Physical registers only

        prob_overlap = torch.matmul(reg_probs, reg_probs.T)  # [N, N]
        conflict_matrix = prob_overlap * interf_adj          # Penalize only along interference edges

        loss_conflict = torch.sum(conflict_matrix) / torch.clamp(torch.sum(interf_adj), min=1.0)

        move_mask = (coal_adj > 0)
        if torch.sum(move_mask) > 0:
            loss_coalesce = self.bce_loss(coalesce_logits[move_mask], coalesce_labels[move_mask])
        else:
            loss_coalesce = torch.tensor(0.0, device=color_logits.device)

        total_loss = loss_ce + self.conflict_weight * loss_conflict + self.coalesce_weight * loss_coalesce

        return total_loss, {
            "ce_loss": loss_ce.item(),
            "conflict_loss": loss_conflict.item(),
            "coalesce_loss": loss_coalesce.item(),
            "total_loss": total_loss.item()
        }
