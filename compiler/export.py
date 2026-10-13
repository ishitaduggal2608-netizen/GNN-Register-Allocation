"""
JSON Export and Import Module for Interference Graphs and Register Allocation Labels.
Implements the exact JSON schema defined in Phase 2.5 of the GNN Register Allocation specification.
"""

import json
import os
from typing import Dict, List, Any, Optional
from .interference_graph import InterferenceGraph
from .chaitin_briggs import AllocationResult, canonicalize_register_assignment, variable_order_from_program


def export_graph_to_json(
    ig: InterferenceGraph,
    gt_res: Optional[AllocationResult] = None,
    graph_id: str = "func_001",
    num_registers: int = 4,
    filepath: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exports an InterferenceGraph and ground truth AllocationResult to JSON format matching Phase 2.5 schema.
    """
    features = ig.get_node_features()
    var_to_idx = {v.name: i for i, v in enumerate(ig.variables)}
    canonical_assignment = {}
    if gt_res is not None:
        canonical_assignment = canonicalize_register_assignment(
            variable_order_from_program(ig.program),
            gt_res.register_assignment
        )

    nodes_list = []
    for i, v in enumerate(ig.variables):
        name = v.name
        feat = features[name]

        # Ground truth label extraction
        label_reg = None
        label_spill = False
        if gt_res is not None:
            if name in canonical_assignment:
                reg_str = canonical_assignment[name]  # e.g., 'R0', 'R1'
                label_reg = int(reg_str.replace("R", ""))
                label_spill = False
            else:
                label_reg = None
                label_spill = True

        nodes_list.append({
            "id": i,
            "name": name,
            "degree": feat.degree,
            "live_range_length": feat.live_range_length,
            "loop_depth": feat.loop_depth,
            "use_def_count": feat.use_count,
            "is_move_related": feat.move_degree > 0,
            "spill_cost": float(feat.spill_cost),
            "label_register": label_reg,
            "label_spill": label_spill
        })

    # Interference Edges [[u_id, v_id], ...]
    interf_edges = []
    for u, v in ig.interference_edges:
        if u in var_to_idx and v in var_to_idx:
            interf_edges.append([var_to_idx[u], var_to_idx[v]])

    # Move Edges [{"pair": [u_id, v_id], "coalesce_safe": bool}]
    move_edges = []
    coalesced_set = set(gt_res.coalesced_pairs) if gt_res else set()
    for u, v in ig.coalescing_edges:
        if u in var_to_idx and v in var_to_idx:
            is_safe = (u, v) in coalesced_set or (v, u) in coalesced_set
            move_edges.append({
                "pair": [var_to_idx[u], var_to_idx[v]],
                "coalesce_safe": is_safe
            })

    payload = {
        "graph_id": graph_id,
        "num_registers": num_registers,
        "nodes": nodes_list,
        "interference_edges": interf_edges,
        "move_edges": move_edges
    }

    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    return payload


def load_graph_from_json(filepath: str) -> Dict[str, Any]:
    """Loads JSON interference graph file into a dictionary payload."""
    with open(filepath, "r") as f:
        return json.load(f)
