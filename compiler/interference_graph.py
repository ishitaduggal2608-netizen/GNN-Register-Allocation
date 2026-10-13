"""
Interference Graph and Coalescing Graph Generator module.
Constructs multi-relational graph topology with compiler node features.
"""

from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
import networkx as nx
from .ir import Program, Variable, OpCode, Instruction
from .cfg import ControlFlowGraph
from .liveness import LivenessAnalyzer


class NodeFeature:
    """Structure for node feature vector passed to GNN model."""
    def __init__(
        self,
        var: Variable,
        spill_cost: float,
        loop_depth: int,
        degree: int,
        move_degree: int,
        live_range_length: int,
        use_count: int
    ):
        self.var: Variable = var
        self.spill_cost: float = spill_cost
        self.loop_depth: int = loop_depth
        self.degree: int = degree
        self.move_degree: int = move_degree
        self.live_range_length: int = live_range_length
        self.use_count: int = use_count

    def to_vector(self) -> np.ndarray:
        # Standardized feature vector representation
        return np.array([
            self.spill_cost,
            float(self.loop_depth),
            float(self.degree),
            float(self.move_degree),
            float(self.live_range_length),
            float(self.use_count)
        ], dtype=np.float32)


class InterferenceGraph:
    """Multi-relational graph containing Interference Edges and Coalescing (Move) Edges."""
    def __init__(self, program: Program, cfg: ControlFlowGraph, liveness: LivenessAnalyzer):
        self.program: Program = program
        self.cfg: ControlFlowGraph = cfg
        self.liveness: LivenessAnalyzer = liveness

        self.variables: List[Variable] = program.get_non_const_vars()
        self.var_map: Dict[str, Variable] = {v.name: v for v in self.variables}
        self.var_to_idx: Dict[str, int] = {v.name: i for i, v in enumerate(self.variables)}

        self.interference_edges: Set[Tuple[str, str]] = set()
        self.coalescing_edges: Set[Tuple[str, str]] = set()
        self._degree: Dict[str, int] = {v.name: 0 for v in self.variables}
        self._move_degree: Dict[str, int] = {v.name: 0 for v in self.variables}

        self.spill_costs: Dict[str, float] = {v.name: 0.0 for v in self.variables}
        self.loop_depths: Dict[str, int] = {v.name: 0 for v in self.variables}
        self.live_range_lengths: Dict[str, int] = {v.name: 0 for v in self.variables}
        self.use_counts: Dict[str, int] = {v.name: 0 for v in self.variables}

        self._build_graph()

    def _build_graph(self):
        """Constructs edges and calculates compiler features."""
        # 1. Calculate live ranges and interference edges
        for inst in self.program.instructions:
            defs = inst.get_defined_vars()
            uses = inst.get_used_vars()
            live_out = self.liveness.get_live_variables_at(inst.line_no)

            # Update live range counts and use counts
            for v in live_out:
                self.live_range_lengths[v.name] += 1
            for v in defs.union(uses):
                self.use_counts[v.name] += 1

            # Determine loop nesting depth for variable
            block = self._get_block_for_inst(inst)
            depth = block.loop_depth if block else 0

            # Spill cost formula: count * 10^loop_depth
            cost_weight = 10.0 ** depth
            for v in defs.union(uses):
                self.spill_costs[v.name] += cost_weight
                self.loop_depths[v.name] = max(self.loop_depths[v.name], depth)

            # 2. Add interference edges:
            # For each defined var d in inst, d interferes with every v in live_out except in a MOVE d <- s
            if inst.op == OpCode.MOVE:
                target, src = inst.target, inst.arg1
                if target and src and not target.is_const and not src.is_const and target.name != src.name:
                    # Move affinity coalescing edge
                    edge = tuple(sorted([target.name, src.name]))
                    self.coalescing_edges.add(edge)

                for d in defs:
                    for l in live_out:
                        if d != l and (inst.op != OpCode.MOVE or l != inst.arg1):
                            edge = tuple(sorted([d.name, l.name]))
                            self.interference_edges.add(edge)
            else:
                for d in defs:
                    for l in live_out:
                        if d != l:
                            edge = tuple(sorted([d.name, l.name]))
                            self.interference_edges.add(edge)

        self._recompute_degrees()

    def _recompute_degrees(self):
        """Precomputes edge degrees once after graph construction."""
        self._degree = {v.name: 0 for v in self.variables}
        self._move_degree = {v.name: 0 for v in self.variables}

        for u, v in self.interference_edges:
            if u in self._degree:
                self._degree[u] += 1
            if v in self._degree:
                self._degree[v] += 1

        for u, v in self.coalescing_edges:
            if u in self._move_degree:
                self._move_degree[u] += 1
            if v in self._move_degree:
                self._move_degree[v] += 1

    def _get_block_for_inst(self, inst: Instruction):
        return self.cfg.inst_to_block.get(inst)

    def get_interference_degree(self, var_name: str) -> int:
        return self._degree.get(var_name, 0)

    def get_move_degree(self, var_name: str) -> int:
        return self._move_degree.get(var_name, 0)

    def get_node_features(self) -> Dict[str, NodeFeature]:
        features = {}
        for v in self.variables:
            name = v.name
            deg = self.get_interference_degree(name)
            mdeg = self.get_move_degree(name)
            features[name] = NodeFeature(
                var=v,
                spill_cost=self.spill_costs[name],
                loop_depth=self.loop_depths[name],
                degree=deg,
                move_degree=mdeg,
                live_range_length=self.live_range_lengths[name],
                use_count=self.use_counts[name]
            )
        return features

    def get_raw_feature_matrix(self) -> np.ndarray:
        """Returns unnormalized N x F compiler feature matrix for GNN input."""
        features_dict = self.get_node_features()
        matrix = []
        for v in self.variables:
            feat = features_dict[v.name]
            matrix.append([
                feat.spill_cost,
                float(feat.loop_depth),
                float(feat.degree),
                float(feat.move_degree),
                float(feat.live_range_length),
                float(feat.use_count)
            ])
        return np.array(matrix, dtype=np.float32)

    def get_feature_matrix(self, normalization_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Returns N x F features, using training-set normalization stats when provided."""
        matrix = self.get_raw_feature_matrix()
        if normalization_stats is None:
            return matrix

        means = np.array(normalization_stats["mean"], dtype=np.float32).reshape(1, -1)
        stds = np.array(normalization_stats["std"], dtype=np.float32).reshape(1, -1)
        stds[stds == 0] = 1.0
        return (matrix - means) / stds

    def to_networkx(self) -> Tuple[nx.Graph, nx.Graph]:
        """Returns NetworkX graphs for interference and coalescing."""
        g_interf = nx.Graph()
        g_coal = nx.Graph()

        for v in self.variables:
            g_interf.add_node(v.name)
            g_coal.add_node(v.name)

        for u, v in self.interference_edges:
            g_interf.add_edge(u, v)

        for u, v in self.coalescing_edges:
            g_coal.add_edge(u, v)

        return g_interf, g_coal
