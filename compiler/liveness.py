"""
Dataflow Liveness Analysis module.
Computes live-in and live-out variable sets for basic blocks and instruction-level live ranges.
"""

from typing import Dict, Set, List
from .cfg import ControlFlowGraph, BasicBlock
from .ir import Variable, Instruction


class LivenessAnalyzer:
    """Performs backward iterative dataflow analysis on the Control Flow Graph."""
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.inst_live_in: Dict[int, Set[Variable]] = {}
        self.inst_live_out: Dict[int, Set[Variable]] = {}

        self.run_liveness_analysis()
        self.run_instruction_liveness()

    def run_liveness_analysis(self):
        """Iterative fixpoint dataflow analysis for basic block liveness."""
        # 1. Compute USE and DEF sets for every block
        for block in self.cfg.blocks:
            block.compute_gen_kill()
            block.live_in = set()
            block.live_out = set()

        changed = True
        iterations = 0
        max_iterations = 100

        # 2. Iterate backward until fixpoint
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for block in reversed(self.cfg.blocks):
                # OUT[B] = Union of IN[succ]
                new_out = set()
                for succ in block.successors:
                    new_out.update(succ.live_in)

                # IN[B] = USE[B] U (OUT[B] - DEF[B])
                new_in = block.use_set.union(new_out - block.def_set)

                if new_in != block.live_in or new_out != block.live_out:
                    block.live_in = new_in
                    block.live_out = new_out
                    changed = True

    def run_instruction_liveness(self):
        """Computes precise instruction-by-instruction live_in and live_out variable sets."""
        for block in self.cfg.blocks:
            if not block.instructions:
                continue

            # Start at bottom of basic block with block.live_out
            current_live = set(block.live_out)

            for inst in reversed(block.instructions):
                self.inst_live_out[inst.line_no] = set(current_live)

                # Live_in = (Live_out - DEF) U USE
                defs = inst.get_defined_vars()
                uses = inst.get_used_vars()

                current_live = (current_live - defs).union(uses)
                self.inst_live_in[inst.line_no] = set(current_live)

    def get_live_variables_at(self, line_no: int) -> Set[Variable]:
        """Returns the set of variables live at a given instruction line number."""
        return self.inst_live_out.get(line_no, set())
