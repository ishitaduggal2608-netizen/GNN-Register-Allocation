"""
Consensus Arbiter & Safe Coalescing Decoder.
Implements two-stage confidence-weighted soft voting across pattern detection agents,
conservative Briggs/George coalescing validation, and agreement diagnostics.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any, Optional
import math
import networkx as nx
import torch
import numpy as np

from compiler.interference_graph import InterferenceGraph
from models.repair import repair_conflicts
from .base import AgentPrediction, extract_graph_tensors


@dataclass
class ConsensusResult:
    """Outcome of ConsensusArbiter multi-agent soft voting."""
    register_assignment: Dict[str, str]        # var_name -> 'R0'..'R(K-1)' or 'SPILL'
    spilled_vars: List[str]                    # list of spilled variable names
    coalesced_pairs: List[Tuple[str, str]]     # conservatively accepted move pairs
    eliminated_moves: int                      # moves eliminated
    conflicts_repaired: int                    # conflicts repaired post-hoc
    pairwise_agreement: Dict[str, float]       # e.g. "rgcn-gat": 85.2%
    unanimous_pct: float                       # percentage of nodes with unanimous agreement
    disagreement_nodes: List[Dict[str, Any]]   # nodes where agents disagreed
    consensus_probs: torch.Tensor              # [N, K+1] ensemble probabilities
    coalesce_matrix: torch.Tensor              # [N, N] combined coalescing scores


class ConsensusArbiter:
    """
    Coordinator Arbiter (not a fifth detector):
    Combines predictions from multiple PatternAgent models via two-stage confidence-weighted soft voting:
    Stage 1: Spill vs No-Spill decision.
    Stage 2: Physical register class assignment for surviving nodes.
    Applies conservative Briggs/George safety test to ensure valid coalescing.
    Runs greedy conflict repair to guarantee 0 conflicts.
    """
    def __init__(self, num_registers: int = 4, coalesce_threshold: float = 0.5):
        self.num_registers = num_registers
        self.coalesce_threshold = coalesce_threshold

    def evaluate_conservative_coalesce(
        self,
        u: str,
        v: str,
        interf_graph: nx.Graph,
        K: int
    ) -> bool:
        """
        Conservative Briggs & George Coalescing Safety Test:
        Returns True if merging u and v is guaranteed not to increase the chromatic number beyond K.
        """
        if interf_graph.has_edge(u, v):
            return False

        neighbors_u = set(interf_graph.neighbors(u))
        neighbors_v = set(interf_graph.neighbors(v))
        combined_neighbors = neighbors_u.union(neighbors_v)

        # Briggs criterion: merged node has fewer than K neighbors of degree >= K
        high_deg_count = sum(1 for n in combined_neighbors if interf_graph.degree(n) >= K)
        if high_deg_count < K:
            return True

        # George criterion: every neighbor of u either has degree < K or interferes with v
        george_u = all(interf_graph.degree(n) < K or interf_graph.has_edge(v, n) for n in neighbors_u)
        if george_u:
            return True

        george_v = all(interf_graph.degree(n) < K or interf_graph.has_edge(u, n) for n in neighbors_v)
        return george_v

    def arbitrate(
        self,
        predictions: Dict[str, AgentPrediction],
        ig: InterferenceGraph,
        weights: Optional[Dict[str, float]] = None
    ) -> ConsensusResult:
        """
        Executes two-stage confidence-weighted consensus across all provided agent predictions.
        """
        agent_names = list(predictions.keys())
        if not agent_names:
            raise ValueError("ConsensusArbiter requires at least one agent prediction.")

        N = len(ig.variables)
        K = self.num_registers
        spill_class = K
        var_names = [v.name for v in ig.variables]
        var_to_idx = {v.name: i for i, v in enumerate(ig.variables)}

        # Default equal weighting across agents
        if weights is None:
            weights = {name: 1.0 / len(agent_names) for name in agent_names}
        else:
            w_sum = sum(weights.values()) or 1.0
            weights = {k: v / w_sum for k, v in weights.items()}

        # 1. Soft Confidence-Weighted Aggregation:
        # For each node i, weight w_{a, i} = w_a * conf_{a, i}
        consensus_probs = torch.zeros((N, K + 1), dtype=torch.float32)

        for i in range(N):
            node_agent_weights = []
            for name in agent_names:
                pred = predictions[name]
                conf = float(pred.confidence[i].item()) if pred.confidence is not None else 0.5
                node_agent_weights.append(weights[name] * max(conf, 1e-4))

            total_w = sum(node_agent_weights) or 1.0
            norm_node_w = [w / total_w for w in node_agent_weights]

            for w_a, name in zip(norm_node_w, agent_names):
                pred = predictions[name]
                consensus_probs[i] += w_a * pred.color_probs[i]

        # 2. Two-Stage Decision:
        # Stage 1: Spill vs No-Spill
        # Stage 2: Best physical register for non-spill nodes with deterministic tie-breaking (lowest index R0 < R1...)
        raw_pred_classes = torch.zeros(N, dtype=torch.long)
        for i in range(N):
            p_spill = consensus_probs[i, spill_class].item()
            if p_spill >= 0.5:
                raw_pred_classes[i] = spill_class
            else:
                # Argmax over physical registers 0..K-1
                reg_probs = consensus_probs[i, :K].tolist()
                best_reg = 0
                max_p = -1.0
                for r_idx, p_val in enumerate(reg_probs):
                    if p_val > max_p:
                        max_p = p_val
                        best_reg = r_idx
                raw_pred_classes[i] = best_reg

        # 3. Conflict Resolution Post-Processing
        # Build interference adjacency tensor
        interf_adj = torch.zeros((N, N), dtype=torch.float32)
        for u, v in ig.interference_edges:
            if u in var_to_idx and v in var_to_idx:
                i, j = var_to_idx[u], var_to_idx[v]
                interf_adj[i, j] = 1.0
                interf_adj[j, i] = 1.0

        repaired_classes, conflicts_repaired = repair_conflicts(
            interf_adj, raw_pred_classes, num_registers=K, spill_class=spill_class
        )

        # Build initial assignment dictionary
        assignment: Dict[str, str] = {}
        spilled_vars: List[str] = []
        for name in var_names:
            idx = var_to_idx[name]
            cls_idx = repaired_classes[idx].item()
            if cls_idx < K:
                assignment[name] = f"R{cls_idx}"
            else:
                assignment[name] = "SPILL"
                spilled_vars.append(name)

        # 4. Conservative Coalescing Validation
        # Aggregate coalescing scores
        coalesce_mats = [
            pred.coalesce_score for pred in predictions.values() if pred.coalesce_score is not None
        ]
        if coalesce_mats:
            combined_coal_scores = torch.mean(torch.stack(coalesce_mats), dim=0)
        else:
            combined_coal_scores = torch.zeros((N, N))

        g_interf, _ = ig.to_networkx()
        work_interf = g_interf.copy()
        accepted_coalesced_pairs: List[Tuple[str, str]] = []

        # Sort candidate move edges by combined learned score
        scored_moves = []
        for u, v in ig.coalescing_edges:
            if u in var_to_idx and v in var_to_idx:
                i, j = var_to_idx[u], var_to_idx[v]
                score = float(combined_coal_scores[i, j].item())
                scored_moves.append((score, u, v))
        scored_moves.sort(reverse=True, key=lambda x: x[0])

        alias_map = {name: name for name in var_names}
        def get_alias(name: str) -> str:
            while alias_map[name] != name:
                name = alias_map[name]
            return name

        for score, u, v in scored_moves:
            u_alias = get_alias(u)
            v_alias = get_alias(v)

            if u_alias == v_alias:
                continue

            # (i) passes learned threshold
            if score < self.coalesce_threshold:
                continue

            # (ii) endpoints do not interfere & (iii) conservative Briggs/George check passes
            if self.evaluate_conservative_coalesce(u_alias, v_alias, work_interf, K):
                # Safely merge
                alias_map[u_alias] = v_alias
                accepted_coalesced_pairs.append((u_alias, v_alias))

                # Update working interference graph
                for nbr in list(work_interf.neighbors(u_alias)):
                    if nbr != v_alias and not work_interf.has_edge(v_alias, nbr):
                        work_interf.add_edge(v_alias, nbr)
                work_interf.remove_node(u_alias)

                # Unify physical register assignment if both were assigned registers
                color_u = assignment.get(u_alias, "SPILL")
                color_v = assignment.get(v_alias, "SPILL")
                if color_u != "SPILL" and color_v != "SPILL":
                    assignment[u_alias] = color_v
                elif color_v != "SPILL":
                    assignment[u_alias] = color_v
                elif color_u != "SPILL":
                    assignment[v_alias] = color_u

        # Map aliases back
        for name in var_names:
            alias = get_alias(name)
            if alias in assignment:
                assignment[name] = assignment[alias]

        # Count eliminated MOVE instructions
        from compiler.ir import OpCode
        eliminated_moves = 0
        for inst in ig.program.instructions:
            if inst.op == OpCode.MOVE:
                t = get_alias(inst.target.name) if inst.target else None
                s = get_alias(inst.arg1.name) if inst.arg1 else None
                if t and s:
                    assign_t = assignment.get(t, "SPILL_T")
                    assign_s = assignment.get(s, "SPILL_S")
                    if (t == s or assign_t == assign_s) and not assign_t.startswith("SPILL"):
                        eliminated_moves += 1

        # 5. Agreement & Disagreement Analytics
        # Pairwise agreement %
        pairwise_agreement: Dict[str, float] = {}
        for idx_a, name_a in enumerate(agent_names):
            for idx_b, name_b in enumerate(agent_names):
                if idx_a < idx_b:
                    classes_a = predictions[name_a].pred_classes
                    classes_b = predictions[name_b].pred_classes
                    match_pct = float(torch.sum(classes_a == classes_b).item()) / max(N, 1) * 100.0
                    pair_key = f"{name_a}_vs_{name_b}"
                    pairwise_agreement[pair_key] = round(match_pct, 1)

        # Unanimous agreement %
        unanimous_nodes = 0
        disagreement_nodes: List[Dict[str, Any]] = []

        features = ig.get_node_features()
        for i, name in enumerate(var_names):
            agent_votes = {
                a_name: predictions[a_name].pred_classes[i].item() for a_name in agent_names
            }
            unique_votes = set(agent_votes.values())
            if len(unique_votes) == 1:
                unanimous_nodes += 1
            else:
                vote_summary = {}
                for a_name, cls_id in agent_votes.items():
                    vote_summary[a_name] = f"R{cls_id}" if cls_id < K else "SPILL"

                disagreement_nodes.append({
                    "var_name": name,
                    "degree": features[name].degree if name in features else 0,
                    "loop_depth": features[name].loop_depth if name in features else 0,
                    "spill_cost": features[name].spill_cost if name in features else 1.0,
                    "consensus_decision": assignment.get(name, "SPILL"),
                    "agent_votes": vote_summary
                })

        unanimous_pct = round((unanimous_nodes / max(N, 1)) * 100.0, 1)

        return ConsensusResult(
            register_assignment=assignment,
            spilled_vars=[name for name, reg in assignment.items() if reg == "SPILL"],
            coalesced_pairs=accepted_coalesced_pairs,
            eliminated_moves=eliminated_moves,
            conflicts_repaired=conflicts_repaired,
            pairwise_agreement=pairwise_agreement,
            unanimous_pct=unanimous_pct,
            disagreement_nodes=disagreement_nodes,
            consensus_probs=consensus_probs,
            coalesce_matrix=combined_coal_scores
        )
