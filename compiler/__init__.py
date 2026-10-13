"""
Compiler Subsystem for GNN Register Allocation & Spill Coalescing.
Provides Intermediate Representation (TAC), Control Flow Graph (CFG),
Liveness Analysis, Interference & Coalescing Graph Builder, and Baseline Chaitin-Briggs Allocator.
"""

from .ir import Instruction, OpCode, Program, Variable
from .cfg import BasicBlock, ControlFlowGraph
from .liveness import LivenessAnalyzer
from .interference_graph import InterferenceGraph, NodeFeature
from .chaitin_briggs import ChaitinBriggsAllocator, AllocationResult

__all__ = [
    "Instruction",
    "OpCode",
    "Program",
    "Variable",
    "BasicBlock",
    "ControlFlowGraph",
    "LivenessAnalyzer",
    "InterferenceGraph",
    "NodeFeature",
    "ChaitinBriggsAllocator",
    "AllocationResult",
]
