"""
GNN Model Package for Register Allocation & Spill Coalescing.
Contains Relational GNN Architectures (R-GCN, R-GAT, R-SAGE, R-GIN) and Training Pipeline.
"""

from .base_allocator import BaseRelationalAllocator, GraphColoringLoss
from .gnn_allocator import RelationalGNNRegisterAllocator
from .rgat_allocator import RelationalGATRegisterAllocator, RGATRegisterAllocator
from .rsage_allocator import RelationalSAGERegisterAllocator, RSAGERegisterAllocator
from .gin_allocator import RelationalGINRegisterAllocator, GINRegisterAllocator
from .trainer import GNNTrainer

__all__ = [
    "BaseRelationalAllocator",
    "GraphColoringLoss",
    "RelationalGNNRegisterAllocator",
    "RelationalGATRegisterAllocator",
    "RGATRegisterAllocator",
    "RelationalSAGERegisterAllocator",
    "RSAGERegisterAllocator",
    "RelationalGINRegisterAllocator",
    "GINRegisterAllocator",
    "GNNTrainer",
]
