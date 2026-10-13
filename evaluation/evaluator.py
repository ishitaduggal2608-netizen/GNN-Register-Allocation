"""
Comparative Benchmark Evaluator for Register Allocation Strategies.
Compares GNN-guided Allocators (R-GCN, R-GAT, R-SAGE, R-GIN, GCN, GraphSAGE, GAT)
vs Chaitin-Briggs Baseline vs Random Allocator.
Tracks pre-repair vs post-repair validity, conflict rates, wall-clock timing, and multi-seed statistics.
"""

from typing import List, Dict, Tuple, Any, Optional
import random
import time
import torch
import torch.nn as nn
import numpy as np

from compiler.ir import Program
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import ChaitinBriggsAllocator
from dataset.generator import SyntheticIRGenerator
from models.gnn_allocator import (
    RelationalGNNRegisterAllocator,
    GCNSpillPredictor,
    SAGESpillPredictor,
    GATSpillPredictor
)
from models.rgat_allocator import RelationalGATRegisterAllocator
from models.rsage_allocator import RelationalSAGERegisterAllocator
from models.gin_allocator import RelationalGINRegisterAllocator
from models.repair import repair_conflicts
from models.trainer import GNNTrainer
from evaluation.metrics import CompilerAllocationMetrics, EvaluatedMetrics
from agents.base import extract_graph_tensors


class BenchmarkEvaluator:
    """Runs end-to-end benchmark comparisons across multiple programs."""
    def __init__(self, num_registers: int = 4):
        self.num_registers: int = num_registers
        self.registers: List[str] = [f"R{i}" for i in range(num_registers)]

    def run_gnn_allocation(
        self, model: nn.Module, ig: InterferenceGraph, apply_repair: bool = True
    ) -> Tuple[Dict[str, str], int, float]:
        """
        Runs full GNN inference and physical register assignment on an interference graph.
        Returns:
            register_assignment: Dict mapping variable name -> 'R0'..'R(K-1)' or 'SPILL'
            coloring_conflicts: number of adjacent nodes sharing same physical register
            inference_time_seconds: wall-clock time
        """
        model.eval()
        start_time = time.perf_counter()

        x, interf_adj, coal_adj, var_order = extract_graph_tensors(ig)

        # Apply normalisation stats if present on model
        norm_stats = getattr(model, "normalization_stats", None)
        if norm_stats is not None:
            mean = norm_stats.get("mean")
            std = norm_stats.get("std")
            if mean is not None and std is not None:
                mean_t = torch.tensor(mean, dtype=torch.float32, device=x.device)
                std_t = torch.tensor(std, dtype=torch.float32, device=x.device).clamp(min=1e-6)
                if mean_t.numel() == x.size(-1):
                    x = (x - mean_t) / std_t

        with torch.no_grad():
            color_logits, _ = model(x, interf_adj, coal_adj)
            pred_classes = torch.argmax(color_logits, dim=-1)

        N = len(var_order)
        spill_class = self.num_registers

        # Pre-repair conflicts count
        pre_conflicts = 0
        for i in range(N):
            for j in range(i + 1, N):
                if interf_adj[i, j] > 0:
                    ci, cj = pred_classes[i].item(), pred_classes[j].item()
                    if ci < spill_class and cj < spill_class and ci == cj:
                        pre_conflicts += 1

        final_classes = pred_classes
        post_conflicts = pre_conflicts

        if apply_repair and pre_conflicts > 0:
            repaired_classes, remaining_conflicts = repair_conflicts(
                interf_adj, pred_classes, num_registers=self.num_registers, spill_class=spill_class
            )
            final_classes = repaired_classes
            post_conflicts = remaining_conflicts

        register_assignment = {}
        for idx, var_name in enumerate(var_order):
            cls_id = final_classes[idx].item()
            if cls_id == spill_class:
                register_assignment[var_name] = "SPILL"
            else:
                register_assignment[var_name] = f"R{cls_id}"

        inference_time = time.perf_counter() - start_time
        return register_assignment, post_conflicts, inference_time

    def run_random_allocation(self, ig: InterferenceGraph) -> Dict[str, str]:
        """Baseline random allocator assigning random physical registers or spill."""
        assignment = {}
        choices = self.registers + ["SPILL"]
        for var in ig.variables:
            assignment[var.name] = random.choice(choices)
        return assignment

    def benchmark_batch(
        self,
        model: nn.Module,
        test_samples: List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, Any]]
    ) -> Dict[str, List[EvaluatedMetrics]]:
        """Benchmarks GNN model, Chaitin-Briggs, and Random Allocator across a batch of programs."""
        cb_allocator = ChaitinBriggsAllocator(num_registers=self.num_registers)
        results = {"Chaitin-Briggs": [], "GNN": [], "Random": []}

        for prog, cfg, liveness, ig, _ in test_samples:
            # 1. Chaitin-Briggs
            cb_result = cb_allocator.allocate(ig)
            cb_metrics = CompilerAllocationMetrics.evaluate(
                "Chaitin-Briggs", ig, cb_result.register_assignment, cb_result.coalesced_pairs, self.num_registers
            )
            results["Chaitin-Briggs"].append(cb_metrics)

            # 2. GNN Allocator
            gnn_assignment, _, _ = self.run_gnn_allocation(model, ig, apply_repair=True)
            gnn_metrics = CompilerAllocationMetrics.evaluate(
                "GNN-Guided", ig, gnn_assignment, [], self.num_registers
            )
            results["GNN"].append(gnn_metrics)

            # 3. Random Baseline
            rand_assignment = self.run_random_allocation(ig)
            rand_metrics = CompilerAllocationMetrics.evaluate(
                "Random", ig, rand_assignment, [], self.num_registers
            )
            results["Random"].append(rand_metrics)

        return results

    def compare_model_architectures(
        self,
        dataset: Any,
        val_dataset: Any,
        test_samples: List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, Any]],
        seeds: List[int] = [0, 1, 2],
        epochs: int = 15,
        use_relational: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Model Comparison:
        Evaluates R-GCN, R-GAT, R-SAGE, R-GIN (or classic baselines) over multiple random seeds, reporting mean +- std.
        """
        if use_relational:
            architectures = [
                ("R-GCN", RelationalGNNRegisterAllocator),
                ("R-GAT", RelationalGATRegisterAllocator),
                ("R-SAGE", RelationalSAGERegisterAllocator),
                ("R-GIN", RelationalGINRegisterAllocator)
            ]
        else:
            architectures = [
                ("GCN", GCNSpillPredictor),
                ("GraphSAGE", SAGESpillPredictor),
                ("GAT", GATSpillPredictor),
                ("R-GCN", RelationalGNNRegisterAllocator)
            ]

        model_results = {}

        for name, cls in architectures:
            acc_list = []
            spill_list = []
            time_list = []

            for s in seeds:
                torch.manual_seed(s)
                model = cls(in_channels=6, num_registers=self.num_registers)
                trainer = GNNTrainer(model, dataset, val_dataset=val_dataset)
                trainer.train(num_epochs=epochs, verbose=False)

                eval_metrics = trainer.evaluate()
                acc_list.append(eval_metrics["accuracy"])

                # Run timing and spill benchmark on test samples
                spill_count = 0
                t_total = 0.0
                for prog, cfg, liveness, ig, _ in test_samples:
                    asgn, _, t_infer = self.run_gnn_allocation(model, ig, apply_repair=True)
                    spill_count += sum(1 for r in asgn.values() if r == "SPILL")
                    t_total += t_infer

                spill_list.append(spill_count / max(len(test_samples), 1))
                time_list.append(t_total / max(len(test_samples), 1))

            model_results[name] = {
                "mean_acc": float(np.mean(acc_list)),
                "std_acc": float(np.std(acc_list)),
                "mean_spills": float(np.mean(spill_list)),
                "std_spills": float(np.std(spill_list)),
                "mean_inference_time_ms": float(np.mean(time_list) * 1000.0),
            }

        return model_results

    def compare_four_relational_models(
        self,
        models_dict: Dict[str, nn.Module],
        test_samples: List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmarks all four trained relational models against Chaitin-Briggs and Random baselines.
        Returns a metrics table dictionary suitable for display.
        """
        cb_allocator = ChaitinBriggsAllocator(num_registers=self.num_registers)
        metrics_store: Dict[str, Dict[str, List[float]]] = {}

        all_keys = list(models_dict.keys()) + ["Chaitin-Briggs", "Random"]
        for k in all_keys:
            metrics_store[k] = {"spills": [], "cost": [], "moves": [], "conflicts": [], "time_ms": []}

        for prog, cfg, liveness, ig, _ in test_samples:
            # 1. Trained GNN models
            for m_name, model in models_dict.items():
                t0 = time.perf_counter()
                asgn, conf, t_infer = self.run_gnn_allocation(model, ig, apply_repair=True)
                t_ms = (time.perf_counter() - t0) * 1000.0
                m = CompilerAllocationMetrics.evaluate(m_name, ig, asgn, [], self.num_registers)
                metrics_store[m_name]["spills"].append(m.total_spills)
                metrics_store[m_name]["cost"].append(m.total_spill_cost)
                metrics_store[m_name]["moves"].append(m.move_elimination_rate_pct)
                metrics_store[m_name]["conflicts"].append(m.coloring_conflicts)
                metrics_store[m_name]["time_ms"].append(t_ms)

            # 2. Chaitin-Briggs
            t0 = time.perf_counter()
            cb_res = cb_allocator.allocate(ig)
            t_ms = (time.perf_counter() - t0) * 1000.0
            cb_m = CompilerAllocationMetrics.evaluate("Chaitin-Briggs", ig, cb_res.register_assignment, cb_res.coalesced_pairs, self.num_registers)
            metrics_store["Chaitin-Briggs"]["spills"].append(cb_m.total_spills)
            metrics_store["Chaitin-Briggs"]["cost"].append(cb_m.total_spill_cost)
            metrics_store["Chaitin-Briggs"]["moves"].append(cb_m.move_elimination_rate_pct)
            metrics_store["Chaitin-Briggs"]["conflicts"].append(cb_m.coloring_conflicts)
            metrics_store["Chaitin-Briggs"]["time_ms"].append(t_ms)

            # 3. Random Allocator
            t0 = time.perf_counter()
            rnd_asgn = self.run_random_allocation(ig)
            t_ms = (time.perf_counter() - t0) * 1000.0
            rnd_m = CompilerAllocationMetrics.evaluate("Random", ig, rnd_asgn, [], self.num_registers)
            metrics_store["Random"]["spills"].append(rnd_m.total_spills)
            metrics_store["Random"]["cost"].append(rnd_m.total_spill_cost)
            metrics_store["Random"]["moves"].append(rnd_m.move_elimination_rate_pct)
            metrics_store["Random"]["conflicts"].append(rnd_m.coloring_conflicts)
            metrics_store["Random"]["time_ms"].append(t_ms)

        summary = {}
        for name, data in metrics_store.items():
            summary[name] = {
                "avg_spills": round(float(np.mean(data["spills"])), 2),
                "avg_spill_cost": round(float(np.mean(data["cost"])), 2),
                "avg_move_elim_pct": round(float(np.mean(data["moves"])), 2),
                "avg_conflicts": round(float(np.mean(data["conflicts"])), 2),
                "avg_time_ms": round(float(np.mean(data["time_ms"])), 2),
            }

        return summary

    @staticmethod
    def summarize_benchmark(results: Dict[str, List[EvaluatedMetrics]]) -> Dict[str, Dict[str, float]]:
        """Computes aggregate benchmark averages across test programs."""
        summary = {}
        for strategy, metrics_list in results.items():
            n = len(metrics_list)
            avg_spills = sum(m.total_spills for m in metrics_list) / max(n, 1)
            avg_spill_cost = sum(m.total_spill_cost for m in metrics_list) / max(n, 1)
            avg_move_elim = sum(m.move_elimination_rate_pct for m in metrics_list) / max(n, 1)
            avg_conflicts = sum(m.coloring_conflicts for m in metrics_list) / max(n, 1)

            summary[strategy] = {
                "avg_spills": round(avg_spills, 2),
                "avg_spill_cost": round(avg_spill_cost, 2),
                "avg_move_elim_pct": round(avg_move_elim, 2),
                "avg_conflicts": round(avg_conflicts, 2)
            }
        return summary
