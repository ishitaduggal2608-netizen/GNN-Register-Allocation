"""
Control Flow Graph (CFG) construction and Loop Analysis module.
Partitions TAC instructions into Basic Blocks and computes loop nesting depth.
"""

from typing import List, Dict, Set, Optional
import networkx as nx
from .ir import Program, Instruction, OpCode


class BasicBlock:
    """Represents a basic block of contiguous TAC instructions with single entry and exit."""
    def __init__(self, block_id: int, label: Optional[str] = None):
        self.block_id: int = block_id
        self.label: Optional[str] = label
        self.instructions: List[Instruction] = []
        self.predecessors: Set['BasicBlock'] = set()
        self.successors: Set['BasicBlock'] = set()
        self.loop_depth: int = 0

        # Liveness dataflow sets
        self.use_set: Set = set()
        self.def_set: Set = set()
        self.live_in: Set = set()
        self.live_out: Set = set()

    def add_instruction(self, inst: Instruction):
        self.instructions.append(inst)

    def compute_gen_kill(self):
        """Computes USE (gen) and DEF (kill) sets for the basic block."""
        self.use_set.clear()
        self.def_set.clear()

        for inst in self.instructions:
            # Uses before def in this block
            for u in inst.get_used_vars():
                if u not in self.def_set:
                    self.use_set.add(u)
            # Defs in this block
            for d in inst.get_defined_vars():
                self.def_set.add(d)

    def __repr__(self) -> str:
        lbl = f" ({self.label})" if self.label else ""
        return f"Block_{self.block_id}{lbl}[insts={len(self.instructions)}, loop_depth={self.loop_depth}]"

    def __hash__(self) -> int:
        return hash(self.block_id)

    def __eq__(self, other) -> bool:
        if isinstance(other, BasicBlock):
            return self.block_id == other.block_id
        return False


class ControlFlowGraph:
    """Builds and analyzes the Control Flow Graph of a TAC Program."""
    def __init__(self, program: Program):
        self.program: Program = program
        self.blocks: List[BasicBlock] = []
        self.entry_block: Optional[BasicBlock] = None
        self.exit_block: Optional[BasicBlock] = None
        self.label_to_block: Dict[str, BasicBlock] = {}
        self.inst_to_block: Dict[Instruction, BasicBlock] = {}
        self.nx_graph: nx.DiGraph = nx.DiGraph()

        self._build_cfg()
        self._analyze_loops()

    def _build_cfg(self):
        """Partitions program instructions into basic blocks and connects edges."""
        if not self.program.instructions:
            return

        # 1. Identify block leaders
        leaders: Set[int] = {0}  # First instruction is a leader
        label_map: Dict[str, int] = {}

        for idx, inst in enumerate(self.program.instructions):
            if inst.op == OpCode.LABEL and inst.label:
                leaders.add(idx)
                label_map[inst.label] = idx

        for idx, inst in enumerate(self.program.instructions):
            if inst.op in (OpCode.JUMP, OpCode.BEQ, OpCode.BNE):
                if inst.target_label in label_map:
                    leaders.add(label_map[inst.target_label])
                if idx + 1 < len(self.program.instructions):
                    leaders.add(idx + 1)
            elif inst.op == OpCode.RETURN:
                if idx + 1 < len(self.program.instructions):
                    leaders.add(idx + 1)

        sorted_leaders = sorted(list(leaders))
        # 2. Form basic blocks
        for i, start_idx in enumerate(sorted_leaders):
            end_idx = sorted_leaders[i + 1] if i + 1 < len(sorted_leaders) else len(self.program.instructions)
            first_inst = self.program.instructions[start_idx]

            lbl = first_inst.label if first_inst.op == OpCode.LABEL else None
            block = BasicBlock(block_id=i, label=lbl)

            for inst_idx in range(start_idx, end_idx):
                inst = self.program.instructions[inst_idx]
                block.add_instruction(inst)
                self.inst_to_block[inst] = block

            if lbl:
                self.label_to_block[lbl] = block

            self.blocks.append(block)

        if self.blocks:
            self.entry_block = self.blocks[0]

        # 3. Add control flow edges between blocks
        for block in self.blocks:
            last_inst = block.instructions[-1]
            if last_inst.op == OpCode.JUMP:
                if last_inst.target_label in self.label_to_block:
                    target_block = self.label_to_block[last_inst.target_label]
                    self._add_edge(block, target_block)
            elif last_inst.op in (OpCode.BEQ, OpCode.BNE):
                # Branch edge
                if last_inst.target_label in self.label_to_block:
                    target_block = self.label_to_block[last_inst.target_label]
                    self._add_edge(block, target_block)
                # Fall-through edge
                next_block_id = block.block_id + 1
                if next_block_id < len(self.blocks):
                    self._add_edge(block, self.blocks[next_block_id])
            elif last_inst.op == OpCode.RETURN:
                pass  # Terminal block
            else:
                # Fall-through to next block
                next_block_id = block.block_id + 1
                if next_block_id < len(self.blocks):
                    self._add_edge(block, self.blocks[next_block_id])

        # Build networkx graph representation
        for block in self.blocks:
            self.nx_graph.add_node(block.block_id, block=block)
            for succ in block.successors:
                self.nx_graph.add_edge(block.block_id, succ.block_id)

    def _add_edge(self, src: BasicBlock, dst: BasicBlock):
        src.successors.add(dst)
        dst.predecessors.add(src)

    def _analyze_loops(self):
        """Computes loop nesting depth for basic blocks using natural loop detection via back-edges."""
        if not self.entry_block or not self.nx_graph.nodes:
            return

        # Simple Cycle detection to assign loop depth to blocks in loops
        try:
            cycles = list(nx.simple_cycles(self.nx_graph))
            for cycle in cycles:
                for block_id in cycle:
                    self.blocks[block_id].loop_depth += 1
        except Exception:
            pass
