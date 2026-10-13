"""
Compiler Allocation Metrics Module.
Calculates spill counts, spill costs, move elimination rates, and coloring conflict metrics.
"""

from typing import Dict, Set, List, Tuple, Any
from compiler.interference_graph import InterferenceGraph
from compiler.ir import OpCode, Variable


class EvaluatedMetrics:
    """Structure storing evaluation metrics for a register allocator."""
    def __init__(
        self,
        allocator_name: str,
        total_spills: int,
        total_spill_cost: float,
        eliminated_moves: int,
        total_moves: int,
        coloring_conflicts: int,
        registers_used: int,
        num_registers: int
    ):
        self.allocator_name: str = allocator_name
        self.total_spills: int = total_spills
        self.total_spill_cost: float = total_spill_cost
        self.eliminated_moves: int = eliminated_moves
        self.total_moves: int = total_moves
        self.coloring_conflicts: int = coloring_conflicts
        self.registers_used: int = registers_used
        self.num_registers: int = num_registers

        self.move_elimination_rate_pct: float = (eliminated_moves / max(total_moves, 1)) * 100.0
        self.register_utilization_pct: float = (registers_used / max(num_registers, 1)) * 100.0

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "allocator": self.allocator_name,
            "spill_count": self.total_spills,
            "spill_cost": round(self.total_spill_cost, 2),
            "eliminated_moves": self.eliminated_moves,
            "total_moves": self.total_moves,
            "move_elimination_rate_pct": round(self.move_elimination_rate_pct, 2),
            "coloring_conflicts": self.coloring_conflicts,
            "registers_used": f"{self.registers_used}/{self.num_registers}"
        }

    def __repr__() -> str:
        return (f"[{self.allocator_name}] Spills: {self.total_spills} (Cost: {self.total_spill_cost:.1f}) | "
                f"Moves Eliminated: {self.eliminated_moves}/{self.total_moves} ({self.move_elimination_rate_pct:.1f}%) | "
                f"Conflicts: {self.coloring_conflicts}")


class CompilerAllocationMetrics:
    """Computes compiler allocation metrics given an interference graph and register assignment."""
    @staticmethod
    def evaluate(
        allocator_name: str,
        ig: InterferenceGraph,
        register_assignment: Dict[str, str],  # var_name -> 'R0'..'R(K-1)' or 'SPILL'
        coalesced_pairs: List[Tuple[str, str]],
        num_registers: int = 4
    ) -> EvaluatedMetrics:
        features = ig.get_node_features()
        var_map = ig.var_map

        spills = 0
        total_spill_cost = 0.0
        used_regs = set()

        for v in ig.variables:
            name = v.name
            assign = register_assignment.get(name, "SPILL")
            if assign == "SPILL" or assign is None:
                spills += 1
                total_spill_cost += features[name].spill_cost if name in features else 1.0
            else:
                used_regs.add(assign)

        # Count total MOVE instructions in program
        total_moves = sum(1 for inst in ig.program.instructions if inst.op == OpCode.MOVE)

        # Count eliminated MOVE instructions where src and target got the same physical register or coalesced
        eliminated_moves = 0
        for inst in ig.program.instructions:
            if inst.op == OpCode.MOVE:
                t = inst.target.name if inst.target else None
                s = inst.arg1.name if inst.arg1 else None
                if t and s:
                    assign_t = register_assignment.get(t, "SPILL_T")
                    assign_s = register_assignment.get(s, "SPILL_S")
                    if assign_t == assign_s and not assign_t.startswith("SPILL"):
                        eliminated_moves += 1

        # Count coloring conflicts on interference edges
        coloring_conflicts = 0
        for u, v in ig.interference_edges:
            assign_u = register_assignment.get(u, "SPILL_U")
            assign_v = register_assignment.get(v, "SPILL_V")
            if assign_u == assign_v and not assign_u.startswith("SPILL"):
                coloring_conflicts += 1

        return EvaluatedMetrics(
            allocator_name=allocator_name,
            total_spills=spills,
            total_spill_cost=total_spill_cost,
            eliminated_moves=eliminated_moves,
            total_moves=total_moves,
            coloring_conflicts=coloring_conflicts,
            registers_used=len(used_regs),
            num_registers=num_registers
        )
