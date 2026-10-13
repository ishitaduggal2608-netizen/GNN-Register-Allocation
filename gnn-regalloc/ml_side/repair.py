"""Deterministic feasibility repair for spill predictions and provisional colours."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RepairResult:
    colours: torch.Tensor  # -1 is SPILL, otherwise a zero-based register ID.
    pre_repair_valid: bool
    post_repair_valid: bool
    pre_repair_conflicts: int

    @property
    def total_spills(self) -> int:
        return int((self.colours < 0).sum().item())


def colouring_conflicts(colours: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Counts directed edge conflicts; callers may pass the usual bidirectional PyG edges."""
    source, target = edge_index.cpu()
    conflicts = (colours[source] >= 0) & (colours[source] == colours[target])
    return int(conflicts.sum().item() // 2)


def repair_colouring(spill_probabilities: torch.Tensor, edge_index: torch.Tensor,
                     num_registers: int, threshold: float = 0.5) -> RepairResult:
    """
    Converts a spill-only GNN prediction into a valid allocation.

    The provisional assignment uses ``node_id % k`` so pre-repair validity is a
    meaningful, independently measured quantity.  Repair then greedily retains
    a legal provisional colour where possible, changes colour if needed, and
    spills only when no register remains available.
    """
    if num_registers < 1:
        raise ValueError("num_registers must be positive")
    probabilities = spill_probabilities.detach().cpu()
    num_nodes = probabilities.numel()
    provisional = torch.full((num_nodes,), -1, dtype=torch.long)
    keep = probabilities <= threshold
    provisional[keep] = torch.arange(num_nodes, dtype=torch.long)[keep] % num_registers
    conflicts = colouring_conflicts(provisional, edge_index)

    adjacency = [set() for _ in range(num_nodes)]
    for source, target in edge_index.t().cpu().tolist():
        adjacency[source].add(target)
    order = sorted(range(num_nodes), key=lambda node: (-len(adjacency[node]), node))
    repaired = torch.full((num_nodes,), -1, dtype=torch.long)
    for node in order:
        if not keep[node]:
            continue
        unavailable = {int(repaired[neighbor]) for neighbor in adjacency[node] if repaired[neighbor] >= 0}
        preferred = int(provisional[node])
        if preferred not in unavailable:
            repaired[node] = preferred
            continue
        for colour in range(num_registers):
            if colour not in unavailable:
                repaired[node] = colour
                break

    return RepairResult(repaired, conflicts == 0,
                        colouring_conflicts(repaired, edge_index) == 0, conflicts)


def repair_conflicts(edge_index: torch.Tensor, predicted_labels: torch.Tensor,
                     num_registers: int, spill_class: int) -> tuple[torch.Tensor, int]:
    """Repairs raw multi-class register labels, highest-degree nodes first."""
    labels = predicted_labels.detach().clone().cpu()
    adjacency = [set() for _ in range(labels.numel())]
    for source, target in edge_index.t().cpu().tolist():
        adjacency[source].add(target)
    conflicts_found = 0
    for node in sorted(range(labels.numel()), key=lambda item: (-len(adjacency[item]), item)):
        if labels[node] == spill_class:
            continue
        neighbour_labels = {int(labels[neighbour]) for neighbour in adjacency[node]}
        if int(labels[node]) in neighbour_labels:
            conflicts_found += 1
            available = [register for register in range(num_registers) if register not in neighbour_labels]
            labels[node] = available[0] if available else spill_class
    return labels, conflicts_found
