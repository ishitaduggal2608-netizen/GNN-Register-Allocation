"""
Chaitin-Briggs Graph Coloring Register Allocator & Move Coalescing Baseline.
Implements George/Briggs conservative coalescing and spill weight heuristics.
"""

from typing import Dict, Set, List, Tuple, Optional
import networkx as nx
from .interference_graph import InterferenceGraph
from .ir import Variable, Program, OpCode


class AllocationResult:
    """Holds allocation results: register mapping, spilled nodes, and eliminated moves."""
    def __init__(self):
        self.register_assignment: Dict[str, str] = {}  # var_name -> 'R0', 'R1', etc.
        self.spilled_vars: Set[str] = set()            # spilled variable names
        self.coalesced_pairs: List[Tuple[str, str]] = []  # coalesced variable pairs
        self.eliminated_moves: int = 0
        self.total_spill_cost: float = 0.0

    def __repr__(self) -> str:
        return (f"AllocationResult(assigned={len(self.register_assignment)}, "
                f"spilled={len(self.spilled_vars)}, coalesced={len(self.coalesced_pairs)}, "
                f"eliminated_moves={self.eliminated_moves}, total_spill_cost={self.total_spill_cost:.2f})")


def variable_order_from_program(program: Program) -> List[str]:
    """Returns non-constant variables in first-appearance instruction order."""
    seen: Set[str] = set()
    ordered: List[str] = []

    for inst in program.instructions:
        for var in (inst.target, inst.arg1, inst.arg2):
            if var and not var.is_const and var.name not in seen:
                seen.add(var.name)
                ordered.append(var.name)

    for var in program.get_non_const_vars():
        if var.name not in seen:
            seen.add(var.name)
            ordered.append(var.name)

    return ordered


def canonicalize_register_assignment(var_order: List[str], register_assignment: Dict[str, str]) -> Dict[str, str]:
    """Relabels physical registers deterministically by first assignment appearance."""
    reg_remap: Dict[str, str] = {}
    canonical_assignment: Dict[str, str] = {}

    for name in list(var_order) + sorted(set(register_assignment) - set(var_order)):
        reg = register_assignment.get(name)
        if reg is None:
            continue
        if reg not in reg_remap:
            reg_remap[reg] = f"R{len(reg_remap)}"
        canonical_assignment[name] = reg_remap[reg]

    return canonical_assignment


def canonicalize_allocation_result(result: AllocationResult, var_order: List[str]) -> AllocationResult:
    """Returns a copy with permutation-equivalent register labels mapped to stable canonical labels."""
    canonical = AllocationResult()
    canonical.register_assignment = canonicalize_register_assignment(var_order, result.register_assignment)
    canonical.spilled_vars = set(result.spilled_vars)
    canonical.coalesced_pairs = list(result.coalesced_pairs)
    canonical.eliminated_moves = result.eliminated_moves
    canonical.total_spill_cost = result.total_spill_cost
    return canonical


class ChaitinBriggsAllocator:
    """Classic Chaitin-Briggs Optimistic Register Allocator with Coalescing."""
    def __init__(self, num_registers: int = 4):
        self.K: int = num_registers
        self.registers: List[str] = [f"R{i}" for i in range(num_registers)]

    def allocate(self, ig: InterferenceGraph) -> AllocationResult:
        """Executes full Chaitin-Briggs allocation pipeline."""
        result = AllocationResult()
        g_interf, g_coal = ig.to_networkx()

        # Copy graph for manipulation
        interf_graph = g_interf.copy()
        coal_graph = g_coal.copy()
        features = ig.get_node_features()

        # Track coalescing aliases (e.g. u merged into v)
        alias_map: Dict[str, str] = {v.name: v.name for v in ig.variables}

        def get_alias(name: str) -> str:
            while alias_map[name] != name:
                name = alias_map[name]
            return name

        # 1. Conservative Coalescing (Briggs / George criteria)
        coalesced_count = 0
        for u, v in list(coal_graph.edges()):
            u_alias = get_alias(u)
            v_alias = get_alias(v)

            if u_alias == v_alias or interf_graph.has_edge(u_alias, v_alias):
                continue

            # Briggs criterion: merged node has < K neighbors of degree >= K
            neighbors_u = set(interf_graph.neighbors(u_alias))
            neighbors_v = set(interf_graph.neighbors(v_alias))
            combined_neighbors = neighbors_u.union(neighbors_v)

            high_degree_count = sum(1 for n in combined_neighbors if interf_graph.degree(n) >= self.K)

            if high_degree_count < self.K:
                # Merge u_alias into v_alias
                alias_map[u_alias] = v_alias
                result.coalesced_pairs.append((u_alias, v_alias))
                coalesced_count += 1

                # Update interference graph edges
                for n in neighbors_u:
                    if n != v_alias and not interf_graph.has_edge(v_alias, n):
                        interf_graph.add_edge(v_alias, n)
                interf_graph.remove_node(u_alias)

        # Count eliminated MOVE instructions
        for inst in ig.program.instructions:
            if inst.op == OpCode.MOVE:
                t = get_alias(inst.target.name) if inst.target else None
                a = get_alias(inst.arg1.name) if inst.arg1 else None
                if t and a and t == a:
                    result.eliminated_moves += 1

        # 2. Simplify & Spill Loop (Stack creation)
        select_stack: List[str] = []
        spilled_nodes: Set[str] = set()
        active_nodes = set(interf_graph.nodes())

        work_graph = interf_graph.copy()

        while work_graph.nodes():
            # Find low-degree node (< K)
            low_degree = [n for n in work_graph.nodes() if work_graph.degree(n) < self.K]

            if low_degree:
                # Pick low-degree node to simplify
                node = low_degree[0]
                select_stack.append(node)
                work_graph.remove_node(node)
            else:
                # Potential Spill: Pick node with smallest spill weight (cost / degree)
                best_spill_node = None
                min_weight = float('inf')

                for n in work_graph.nodes():
                    deg = work_graph.degree(n)
                    cost = features[n].spill_cost if n in features else 1.0
                    weight = cost / float(max(deg, 1))

                    if weight < min_weight:
                        min_weight = weight
                        best_spill_node = n

                if best_spill_node:
                    select_stack.append(best_spill_node)
                    spilled_nodes.add(best_spill_node)
                    work_graph.remove_node(best_spill_node)

        # 3. Select / Coloring Phase
        assigned_colors: Dict[str, str] = {}

        while select_stack:
            node = select_stack.pop()

            # Find colors used by active neighbors in original interference graph
            neighbor_colors = set()
            for nbr in interf_graph.neighbors(node):
                if nbr in assigned_colors:
                    neighbor_colors.add(assigned_colors[nbr])

            # Pick first available register
            available = [r for r in self.registers if r not in neighbor_colors]

            if available:
                assigned_colors[node] = available[0]
                if node in spilled_nodes:
                    spilled_nodes.remove(node)  # Optimistic coloring succeeded!
            else:
                # Actual Spill
                result.spilled_vars.add(node)
                result.total_spill_cost += features[node].spill_cost if node in features else 1.0

        # Map back all variables (including coalesced aliases)
        for v in ig.variables:
            alias = get_alias(v.name)
            if alias in assigned_colors:
                result.register_assignment[v.name] = assigned_colors[alias]
            else:
                result.spilled_vars.add(v.name)

        return result
