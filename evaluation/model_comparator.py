"""
Model Comparator & Cross-Architecture Pattern Analysis.
Evaluates R-GCN, R-GAT, R-SAGE, R-GIN, Chaitin-Briggs, and Random Allocator on the same program.
Computes:
1. Pairwise Register Assignment Agreement (%)
2. Spill Decision Confusion Matrix & Agreement (%)
3. Coalescing Edge Jaccard Similarity
4. Static Matplotlib Heatmap & Performance Figure
5. Web Comparison JSON Export
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np

from compiler.ir import Program, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import ChaitinBriggsAllocator
from evaluation.evaluator import BenchmarkEvaluator


class ModelComparator:
    """
    Compares how different GNN architectures and heuristic baselines allocate registers
    and make spill/coalescing trade-offs on the exact same compiler interference graph.
    """
    def __init__(self, num_registers: int = 4):
        self.num_registers: int = num_registers
        self.evaluator = BenchmarkEvaluator(num_registers=num_registers)
        self.cb_allocator = ChaitinBriggsAllocator(num_registers=num_registers)

    def compare_single_program(
        self,
        models: Dict[str, nn.Module],
        ig: InterferenceGraph,
        prog: Program
    ) -> Dict[str, Any]:
        """
        Executes all models on a single program, recording per-node assignments,
        pairwise agreement, confusion matrices, and coalesced move sets.
        """
        var_names = [v.name for v in ig.variables]
        N = len(var_names)

        assignments: Dict[str, Dict[str, str]] = {}
        spill_sets: Dict[str, set] = {}
        spill_costs: Dict[str, float] = {}
        coalesced_move_sets: Dict[str, set] = {}
        inference_times_ms: Dict[str, float] = {}

        # 1. Run GNN Models
        for name, model in models.items():
            t0 = time.perf_counter()
            asgn, _, _ = self.evaluator.run_gnn_allocation(model, ig, apply_repair=True)
            t_ms = (time.perf_counter() - t0) * 1000.0

            assignments[name] = asgn
            spills = {v for v, r in asgn.items() if r == "SPILL"}
            spill_sets[name] = spills
            spill_costs[name] = sum(ig.spill_costs.get(v, 1.0) for v in spills)
            inference_times_ms[name] = round(t_ms, 2)

            # Move elimination by GNN: MOVE u <- v where u, v share register and do not interfere
            coal_set = set()
            for inst in prog.instructions:
                if inst.op == OpCode.MOVE and inst.target and inst.arg1:
                    t = inst.target.name
                    s = inst.arg1.name
                    if t in asgn and s in asgn:
                        rt, rs = asgn[t], asgn[s]
                        if rt == rs and rt != "SPILL" and (t, s) not in ig.interference_edges and (s, t) not in ig.interference_edges:
                            edge_pair = tuple(sorted([t, s]))
                            coal_set.add(edge_pair)
            coalesced_move_sets[name] = coal_set

        # 2. Run Chaitin-Briggs Baseline
        t0 = time.perf_counter()
        cb_res = self.cb_allocator.allocate(ig)
        t_ms = (time.perf_counter() - t0) * 1000.0

        assignments["chaitin_briggs"] = cb_res.register_assignment
        cb_spills = set(cb_res.spilled_vars)
        spill_sets["chaitin_briggs"] = cb_spills
        spill_costs["chaitin_briggs"] = sum(ig.spill_costs.get(v, 1.0) for v in cb_spills)
        inference_times_ms["chaitin_briggs"] = round(t_ms, 2)
        coalesced_move_sets["chaitin_briggs"] = {tuple(sorted(p)) for p in cb_res.coalesced_pairs}

        # 3. Run Random Baseline
        t0 = time.perf_counter()
        rnd_asgn = self.evaluator.run_random_allocation(ig)
        t_ms = (time.perf_counter() - t0) * 1000.0

        assignments["random"] = rnd_asgn
        rnd_spills = {v for v, r in rnd_asgn.items() if r == "SPILL"}
        spill_sets["random"] = rnd_spills
        spill_costs["random"] = sum(ig.spill_costs.get(v, 1.0) for v in rnd_spills)
        inference_times_ms["random"] = round(t_ms, 2)

        rnd_coal_set = set()
        for inst in prog.instructions:
            if inst.op == OpCode.MOVE and inst.target and inst.arg1:
                t, s = inst.target.name, inst.arg1.name
                if t in rnd_asgn and s in rnd_asgn:
                    if rnd_asgn[t] == rnd_asgn[s] and rnd_asgn[t] != "SPILL" and (t, s) not in ig.interference_edges and (s, t) not in ig.interference_edges:
                        rnd_coal_set.add(tuple(sorted([t, s])))
        coalesced_move_sets["random"] = rnd_coal_set

        model_keys = list(assignments.keys())

        # 4. Pairwise Agreement Matrix (% identical register/spill decision)
        register_agreement_matrix: Dict[str, Dict[str, float]] = {m: {} for m in model_keys}
        spill_agreement_matrix: Dict[str, Dict[str, float]] = {m: {} for m in model_keys}
        spill_confusion_matrices: Dict[str, Dict[str, Dict[str, int]]] = {m: {} for m in model_keys}
        coalescing_jaccard_matrix: Dict[str, Dict[str, float]] = {m: {} for m in model_keys}

        for i, m1 in enumerate(model_keys):
            for j, m2 in enumerate(model_keys):
                # (a) Register coloring agreement
                agree_count = sum(1 for v in var_names if assignments[m1].get(v) == assignments[m2].get(v))
                reg_agree_pct = round((agree_count / max(N, 1)) * 100.0, 1)
                register_agreement_matrix[m1][m2] = reg_agree_pct

                # (b) Spill decision confusion matrix
                both_spill = sum(1 for v in var_names if (v in spill_sets[m1] and v in spill_sets[m2]))
                m1_only = sum(1 for v in var_names if (v in spill_sets[m1] and v not in spill_sets[m2]))
                m2_only = sum(1 for v in var_names if (v not in spill_sets[m1] and v in spill_sets[m2]))
                neither = sum(1 for v in var_names if (v not in spill_sets[m1] and v not in spill_sets[m2]))

                conf_mat = {
                    "both_spill": both_spill,
                    "m1_spill_only": m1_only,
                    "m2_spill_only": m2_only,
                    "neither_spill": neither
                }
                spill_confusion_matrices[m1][m2] = conf_mat
                spill_agree_pct = round(((both_spill + neither) / max(N, 1)) * 100.0, 1)
                spill_agreement_matrix[m1][m2] = spill_agree_pct

                # (c) Coalescing Jaccard overlap
                s1, s2 = coalesced_move_sets[m1], coalesced_move_sets[m2]
                if len(s1) == 0 and len(s2) == 0:
                    jaccard = 1.0
                elif len(s1.union(s2)) == 0:
                    jaccard = 0.0
                else:
                    jaccard = round(len(s1.intersection(s2)) / len(s1.union(s2)), 3)
                coalescing_jaccard_matrix[m1][m2] = jaccard

        # 5. Identify Disagreement Nodes
        disagreement_nodes = []
        for v in var_names:
            votes = {m: assignments[m].get(v, "SPILL") for m in model_keys}
            unique_votes = set(votes.values())
            if len(unique_votes) > 1:
                disagreement_nodes.append({
                    "var_name": v,
                    "loop_depth": ig.get_node_features()[v].loop_depth if v in ig.get_node_features() else 0,
                    "spill_cost": ig.spill_costs.get(v, 1.0),
                    "degree": ig.get_degree(v) if hasattr(ig, "get_degree") else ig._degree.get(v, 0),
                    "votes": votes
                })

        # Summary model metrics
        total_moves = sum(1 for inst in prog.instructions if inst.op == OpCode.MOVE)
        model_stats = {}
        for m in model_keys:
            elim_count = len(coalesced_move_sets[m])
            model_stats[m] = {
                "spills": len(spill_sets[m]),
                "spill_cost": round(spill_costs[m], 1),
                "moves_eliminated": elim_count,
                "move_elim_rate_pct": round((elim_count / max(total_moves, 1)) * 100.0, 1),
                "inference_time_ms": inference_times_ms[m]
            }

        return {
            "program_name": prog.name,
            "num_registers": self.num_registers,
            "num_variables": N,
            "total_move_instructions": total_moves,
            "model_keys": model_keys,
            "assignments": assignments,
            "model_stats": model_stats,
            "register_agreement_matrix": register_agreement_matrix,
            "spill_agreement_matrix": spill_agreement_matrix,
            "spill_confusion_matrices": spill_confusion_matrices,
            "coalescing_jaccard_matrix": coalescing_jaccard_matrix,
            "disagreement_nodes": disagreement_nodes
        }

    def compare_aggregate_programs(
        self,
        models: Dict[str, nn.Module],
        test_samples: List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, Any]]
    ) -> Dict[str, Any]:
        """
        Runs comparison across multiple programs, returning mean metrics and agreement statistics.
        """
        all_single_results = []
        for prog, cfg, liveness, ig, _ in test_samples:
            res = self.compare_single_program(models, ig, prog)
            all_single_results.append(res)

        first = all_single_results[0]
        model_keys = first["model_keys"]

        mean_reg_agreement: Dict[str, Dict[str, float]] = {m: {} for m in model_keys}
        mean_spill_agreement: Dict[str, Dict[str, float]] = {m: {} for m in model_keys}
        mean_jaccard: Dict[str, Dict[str, float]] = {m: {} for m in model_keys}

        for m1 in model_keys:
            for m2 in model_keys:
                mean_reg_agreement[m1][m2] = round(
                    float(np.mean([r["register_agreement_matrix"][m1][m2] for r in all_single_results])), 1
                )
                mean_spill_agreement[m1][m2] = round(
                    float(np.mean([r["spill_agreement_matrix"][m1][m2] for r in all_single_results])), 1
                )
                mean_jaccard[m1][m2] = round(
                    float(np.mean([r["coalescing_jaccard_matrix"][m1][m2] for r in all_single_results])), 3
                )

        aggregate_stats = {}
        for m in model_keys:
            aggregate_stats[m] = {
                "avg_spills": round(float(np.mean([r["model_stats"][m]["spills"] for r in all_single_results])), 2),
                "avg_spill_cost": round(float(np.mean([r["model_stats"][m]["spill_cost"] for r in all_single_results])), 2),
                "avg_move_elim_rate_pct": round(float(np.mean([r["model_stats"][m]["move_elim_rate_pct"] for r in all_single_results])), 1),
                "avg_time_ms": round(float(np.mean([r["model_stats"][m]["inference_time_ms"] for r in all_single_results])), 2),
            }

        return {
            "num_programs": len(test_samples),
            "num_registers": self.num_registers,
            "model_keys": model_keys,
            "mean_register_agreement": mean_reg_agreement,
            "mean_spill_agreement": mean_spill_agreement,
            "mean_coalescing_jaccard": mean_jaccard,
            "aggregate_stats": aggregate_stats,
            "sample_single_result": first
        }

    def generate_comparison_plot(
        self,
        comparison_result: Dict[str, Any],
        output_path: str = "report/model_comparison.png"
    ) -> str:
        """
        Generates publication-quality figure:
        - Heatmap of Pairwise Register Assignment Agreement
        - Bar Chart of Spills vs Moves Eliminated
        - Bar Chart of Total Spill Cost
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        model_keys = comparison_result["model_keys"]
        labels = [m.replace("_", "-").upper() for m in model_keys]
        n_models = len(model_keys)

        # Build agreement matrix array
        matrix = np.zeros((n_models, n_models))
        matrix_dict = comparison_result.get("mean_register_agreement", comparison_result.get("register_agreement_matrix"))
        for i, m1 in enumerate(model_keys):
            for j, m2 in enumerate(model_keys):
                matrix[i, j] = matrix_dict[m1][m2]

        stats_dict = comparison_result.get("aggregate_stats", comparison_result.get("model_stats"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # --- 1. Agreement Heatmap ---
        im = ax1.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        ax1.set_xticks(range(n_models))
        ax1.set_yticks(range(n_models))
        ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax1.set_yticklabels(labels, fontsize=9, fontweight="bold")
        ax1.set_title("Pairwise Register Agreement (%)", fontsize=12, fontweight="bold", pad=12)

        # Annotate cells
        for i in range(n_models):
            for j in range(n_models):
                val = matrix[i, j]
                color = "white" if val > 65 else "black"
                ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8.5, fontweight="semibold")

        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Agreement %", fontsize=9)

        # --- 2. Spills & Move Elimination Bar Chart ---
        x = np.arange(n_models)
        width = 0.35

        spills = [stats_dict[m].get("avg_spills", stats_dict[m].get("spills", 0)) for m in model_keys]
        moves = [stats_dict[m].get("avg_move_elim_rate_pct", stats_dict[m].get("move_elim_rate_pct", 0)) for m in model_keys]

        rects1 = ax2.bar(x - width/2, spills, width, label="Avg Spills", color="#ef4444", alpha=0.85, edgecolor="#b91c1c")
        rects2 = ax2.bar(x + width/2, moves, width, label="Move Elim (%)", color="#10b981", alpha=0.85, edgecolor="#047857")

        ax2.set_ylabel("Count / Rate (%)", fontsize=10, fontweight="bold")
        ax2.set_title("Spills vs Move Elimination Rate", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax2.legend(frameon=True)
        ax2.grid(axis="y", linestyle="--", alpha=0.35)

        # Annotate bar values
        for rect in rects1:
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def export_comparison_json(
        self,
        comparison_result: Dict[str, Any],
        prog: Program,
        cfg: ControlFlowGraph,
        ig: InterferenceGraph,
        output_path: str = "web/comparison_data.json"
    ) -> str:
        """
        Exports comparison results to web/comparison_data.json,
        including nodes, graph edges, and multi-model assignments.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        features = ig.get_node_features()
        nodes = []
        for v in ig.variables:
            feat = features[v.name]
            node_dict = {
                "id": v.name,
                "spill_cost": feat.spill_cost,
                "loop_depth": feat.loop_depth,
                "degree": feat.degree,
                "move_degree": feat.move_degree,
                "assignments": {m: comparison_result["assignments"][m].get(v.name, "SPILL") for m in comparison_result["model_keys"]}
            }
            nodes.append(node_dict)

        payload = {
            "program_name": prog.name,
            "num_registers": self.num_registers,
            "instructions": [str(i) for i in prog.instructions],
            "cfg_blocks": [
                {
                    "block_id": b.block_id,
                    "label": b.label,
                    "loop_depth": b.loop_depth,
                    "instructions": [str(i) for i in b.instructions],
                    "successors": [s.block_id for s in b.successors]
                }
                for b in cfg.blocks
            ],
            "nodes": nodes,
            "interference_edges": [{"source": u, "target": v, "type": "interference"} for u, v in ig.interference_edges],
            "coalescing_edges": [{"source": u, "target": v, "type": "coalescing"} for u, v in ig.coalescing_edges],
            "model_keys": comparison_result["model_keys"],
            "model_stats": comparison_result["model_stats"],
            "register_agreement_matrix": comparison_result["register_agreement_matrix"],
            "spill_agreement_matrix": comparison_result["spill_agreement_matrix"],
            "spill_confusion_matrices": comparison_result["spill_confusion_matrices"],
            "coalescing_jaccard_matrix": comparison_result["coalescing_jaccard_matrix"],
            "disagreement_nodes": comparison_result["disagreement_nodes"]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Sync embedded_data.js if exporting to web/ directory
        embedded_path = os.path.join(os.path.dirname(output_path), "embedded_data.js")
        default_path = os.path.join(os.path.dirname(output_path), "data.json")
        default_payload = {}
        if os.path.exists(default_path):
            try:
                with open(default_path, "r", encoding="utf-8") as df:
                    default_payload = json.load(df)
            except Exception:
                pass
        try:
            with open(embedded_path, "w", encoding="utf-8") as ef:
                ef.write("// Auto-generated embedded compiler datasets for instant offline/file-protocol rendering\n")
                ef.write(f"window.__DEFAULT_DATASET__ = {json.dumps(default_payload)};\n")
                ef.write(f"window.__COMPARISON_DATASET__ = {json.dumps(payload)};\n")
        except Exception:
            pass

        return output_path
