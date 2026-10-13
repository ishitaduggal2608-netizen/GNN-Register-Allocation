"""
Graph Neural Network Architectures for Register Allocation, Spill Node Prediction, and Move Coalescing.
Includes GCN, GraphSAGE, GAT, Relational-GCN (R-GCN), and CoalesceClassifier.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



class GCNSpillPredictor(nn.Module):
    """GCN Baseline Model for Spill Node Prediction & Register Coloring."""
    def __init__(self, in_channels: int = 6, hidden_channels: int = 64, num_registers: int = 4, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_registers + 1
        self.num_registers = num_registers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_channels, hidden_channels)) for _ in range(num_layers)
        ])
        for w in self.weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, self.num_classes)
        )
        self.dropout = dropout

    def _norm_adj(self, adj: torch.Tensor) -> torch.Tensor:
        I = torch.eye(adj.size(0), device=adj.device)
        A_tilde = adj + I
        deg = torch.sum(A_tilde, dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        deg_mat = torch.diag_embed(deg_inv_sqrt)
        return torch.matmul(torch.matmul(deg_mat, A_tilde), deg_mat)

    def forward(self, x: torch.Tensor, interf_adj: torch.Tensor, coal_adj: torch.Tensor = None):
        h = F.relu(self.encoder(x))
        norm_adj = self._norm_adj(interf_adj)

        for w in self.weights:
            h_next = torch.matmul(norm_adj, torch.matmul(h, w))
            h = F.relu(h_next)
            h = F.dropout(h, p=self.dropout, training=self.training)

        color_logits = self.classifier(h)
        coalesce_logits = torch.matmul(h, h.T)
        return color_logits, coalesce_logits

    def get_node_embeddings(self, x: torch.Tensor, interf_adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.encoder(x))
        norm_adj = self._norm_adj(interf_adj)
        for w in self.weights:
            h = F.relu(torch.matmul(norm_adj, torch.matmul(h, w)))
        return h


class SAGESpillPredictor(nn.Module):
    """GraphSAGE Baseline Model with Mean Aggregation."""
    def __init__(self, in_channels: int = 6, hidden_channels: int = 64, num_registers: int = 4, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_registers + 1
        self.num_registers = num_registers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.w_self = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_channels, hidden_channels)) for _ in range(num_layers)])
        self.w_neigh = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_channels, hidden_channels)) for _ in range(num_layers)])
        
        for ws, wn in zip(self.w_self, self.w_neigh):
            nn.init.kaiming_uniform_(ws, a=math.sqrt(5))
            nn.init.kaiming_uniform_(wn, a=math.sqrt(5))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, self.num_classes)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, interf_adj: torch.Tensor, coal_adj: torch.Tensor = None):
        h = F.relu(self.encoder(x))
        deg = torch.sum(interf_adj, dim=-1, keepdim=True).clamp(min=1.0)
        norm_mean = interf_adj / deg

        for ws, wn in zip(self.w_self, self.w_neigh):
            h_self = torch.matmul(h, ws)
            h_neigh = torch.matmul(torch.matmul(norm_mean, h), wn)
            h = F.relu(h_self + h_neigh)
            h = F.dropout(h, p=self.dropout, training=self.training)

        color_logits = self.classifier(h)
        coalesce_logits = torch.matmul(h, h.T)
        return color_logits, coalesce_logits


class GATSpillPredictor(nn.Module):
    """Graph Attention Network (GAT) Baseline Model."""
    def __init__(self, in_channels: int = 6, hidden_channels: int = 64, num_registers: int = 4, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_registers + 1
        self.num_registers = num_registers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.w_proj = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_channels, hidden_channels)) for _ in range(num_layers)])
        self.a_attn = nn.ParameterList([nn.Parameter(torch.Tensor(2 * hidden_channels, 1)) for _ in range(num_layers)])

        for wp, aa in zip(self.w_proj, self.a_attn):
            nn.init.kaiming_uniform_(wp, a=math.sqrt(5))
            nn.init.xavier_uniform_(aa)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, self.num_classes)
        )
        self.dropout = dropout
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, interf_adj: torch.Tensor, coal_adj: torch.Tensor = None):
        h = F.relu(self.encoder(x))
        N = h.size(0)

        for layer_idx, (wp, aa) in enumerate(zip(self.w_proj, self.a_attn)):
            h_proj = torch.matmul(h, wp)  # [N, H]
            # Pairwise attention logits
            h_i = h_proj.unsqueeze(1).repeat(1, N, 1)  # [N, N, H]
            h_j = h_proj.unsqueeze(0).repeat(N, 1, 1)  # [N, N, H]
            pair_cat = torch.cat([h_i, h_j], dim=-1)   # [N, N, 2H]
            
            attn_logits = F.leaky_relu(torch.matmul(pair_cat, aa).squeeze(-1), negative_slope=0.2)  # [N, N]
            mask = (interf_adj + torch.eye(N, device=x.device)) == 0
            attn_logits = attn_logits.masked_fill(mask, -1e9)
            attn_weights = F.softmax(attn_logits, dim=-1)

            if layer_idx == len(self.w_proj) - 1:
                self.last_attention_weights = attn_weights.detach()

            h = F.relu(torch.matmul(attn_weights, h_proj))
            h = F.dropout(h, p=self.dropout, training=self.training)

        color_logits = self.classifier(h)
        coalesce_logits = torch.matmul(h, h.T)
        return color_logits, coalesce_logits


class RelationalGConv(nn.Module):
    """Relational Graph Convolutional Layer for multi-relational graphs (Interference & Coalescing)."""
    def __init__(self, in_features: int, out_features: int):
        super(RelationalGConv, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        self.weight_self = nn.Linear(in_features, out_features, bias=False)
        self.weight_interf = nn.Linear(in_features, out_features, bias=False)
        self.weight_coal = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_interf.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_coal.weight, a=math.sqrt(5))

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetrically normalizes adjacency matrix D^{-1/2} A D^{-1/2}."""
        deg = torch.sum(adj, dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        deg_mat = torch.diag_embed(deg_inv_sqrt)
        return torch.matmul(torch.matmul(deg_mat, adj), deg_mat)

    def forward(self, x: torch.Tensor, interf_adj: torch.Tensor, coal_adj: torch.Tensor) -> torch.Tensor:
        norm_interf = self._normalize_adj(interf_adj)
        norm_coal = self._normalize_adj(coal_adj)

        h_self = self.weight_self(x)
        h_interf = torch.matmul(norm_interf, self.weight_interf(x))
        h_coal = torch.matmul(norm_coal, self.weight_coal(x))

        out = h_self + h_interf + h_coal + self.bias
        return out


class RelationalGNNRegisterAllocator(nn.Module):
    """
    Relational Graph Neural Network Architecture for Compiler Register Allocation and Spill Coalescing.
    Outputs:
    1. Register Color Logits [N, K+1]: Class 0..K-1 = Physical Registers, Class K = Spill
    2. Coalescing Edge Logits [N, N]: raw scores for whether a move edge can be safely coalesced.
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
        super(RelationalGNNRegisterAllocator, self).__init__()
        if in_channels is not None:
            in_node_features = in_channels
        self.in_node_features: int = in_node_features
        self.in_channels: int = in_node_features
        self.hidden_dim: int = hidden_dim
        self.hidden_channels: int = hidden_dim
        self.num_registers: int = num_registers
        self.num_classes: int = num_registers + 1  # K physical registers + 1 spill class


        # Node Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # Relational GNN Layers
        self.gconvs = nn.ModuleList([
            RelationalGConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Register Color Classifier Head
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )

        # Coalescing Edge Classifier Head (Bilinear edge scoring)
        self.edge_w = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_w)

    def forward(self, x: torch.Tensor, interf_adj: torch.Tensor, coal_adj: torch.Tensor):
        h = self.encoder(x)

        for gconv, bn in zip(self.gconvs, self.bns):
            h_in = h
            h = gconv(h, interf_adj, coal_adj)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + h_in  # Residual connection

        color_logits = self.color_head(h)  # [N, K+1]

        h_transform = torch.matmul(h, self.edge_w)  # [N, hidden_dim]
        coalesce_logits = torch.matmul(h_transform, h.T)  # [N, N]

        return color_logits, coalesce_logits

    def get_node_embeddings(self, x: torch.Tensor, interf_adj: torch.Tensor, coal_adj: torch.Tensor) -> torch.Tensor:
        """Returns frozen node embeddings [N, hidden_dim] from penultimate GNN layer."""
        h = self.encoder(x)
        for gconv, bn in zip(self.gconvs, self.bns):
            h_in = h
            h = gconv(h, interf_adj, coal_adj)
            h = bn(h)
            h = F.relu(h)
            h = h + h_in
        return h


class CoalesceClassifier(nn.Module):
    """
    Phase 8 Coalescing Edge Classifier.
    Predicts move edge coalescing safety (coalesce_safe: True/False) using node embeddings.
    """
    def __init__(self, node_embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, node_embeddings: torch.Tensor, move_edge_index: torch.Tensor) -> torch.Tensor:
        """
        node_embeddings: [N, D]
        move_edge_index: [2, E]
        Returns: logits [E]
        """
        src, dst = move_edge_index[0], move_edge_index[1]
        pair_features = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
        return self.mlp(pair_features).squeeze(-1)


class GraphColoringLoss(nn.Module):
    """
    Custom Hybrid Loss Function for Compiler Register Allocation:
    1. Cross-Entropy Loss on Ground Truth Register Assignment / Spill with pos_weight support
    2. Soft Interference Color Conflict Loss
    3. Move-Coalescing BCE Loss
    """
    def __init__(self, conflict_weight: float = 0.5, coalesce_weight: float = 0.3, pos_weight: torch.Tensor = None):
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
    ):
        loss_ce = self.ce_loss(color_logits, target_colors)

        probs = F.softmax(color_logits, dim=-1)  # [N, K+1]
        reg_probs = probs[:, :-1]                 # [N, K]

        prob_overlap = torch.matmul(reg_probs, reg_probs.T)  # [N, N]
        conflict_matrix = prob_overlap * interf_adj          # Only where interference edge exists

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
