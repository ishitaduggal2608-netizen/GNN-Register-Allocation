"""
Post-Hoc Greedy Conflict Resolution Module for GNN Register Allocation.
Implements greedy post-processing to eliminate coloring conflicts on raw GNN predictions.
"""

from typing import Tuple, Dict, Set, List
import torch



def repair_conflicts(
    edge_index_or_adj: torch.Tensor,
    predicted_labels: torch.Tensor,
    num_registers: int = 4,
    spill_class: int = 4
) -> Tuple[torch.Tensor, int]:
    """
    Greedy conflict repair using graph coloring.

    Preserves valid model predictions where possible. When a node's
    predicted register conflicts with an already-colored neighbor,
    tries another available register; if none is available, spills it.

    Returns:
        Repaired labels and the number of conflicting nodes encountered.
    """
    if num_registers < 1:
        raise ValueError("num_registers must be at least 1")

    if spill_class < num_registers:
        raise ValueError("spill_class must be outside the physical register range")

    labels = predicted_labels.clone().long()
    N = len(labels)

    adjacency: Dict[int, Set[int]] = {i: set() for i in range(N)}

    if edge_index_or_adj.dim() != 2:
        raise ValueError("Graph input must be a 2D tensor")

    if edge_index_or_adj.size(0) == 2:
        edges = edge_index_or_adj.t().tolist()
        for u, v in edges:
            if u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)
    elif (
        edge_index_or_adj.size(0) == N
        and edge_index_or_adj.size(1) == N
    ):
        edges = edge_index_or_adj.nonzero(as_tuple=False).tolist()
        for u, v in edges:
            if u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)
    else:
        raise ValueError("Expected [2, E] edge index or [N, N] adjacency matrix")

    # Highest-degree nodes first.
    order = sorted(range(N), key=lambda n: -len(adjacency[n]))

    # Uncolored nodes are handled only after their neighbors' decisions
    # are made; no previously assigned color is changed afterward.
    repaired = torch.full_like(labels, spill_class)
    conflicts_found = 0

    for node in order:
        predicted = labels[node].item()
        neighbor_colors = {
            repaired[nbr].item()
            for nbr in adjacency[node]
            if repaired[nbr].item() != spill_class
        }

        # Keep the predicted register if it is valid and available.
        if (
            0 <= predicted < num_registers
            and predicted not in neighbor_colors
        ):
            repaired[node] = predicted
            continue

        if predicted != spill_class:
            conflicts_found += 1

        # Try the lowest available physical register.
        available = [
            r for r in range(num_registers)
            if r not in neighbor_colors
        ]

        if available:
            repaired[node] = available[0]
        else:
            repaired[node] = spill_class

    return repaired, conflicts_found
