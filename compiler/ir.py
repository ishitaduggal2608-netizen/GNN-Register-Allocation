"""
Intermediate Representation (TAC - Three-Address Code) module.
Defines virtual registers, constants, instruction opcodes, and TAC instructions.
"""

from enum import Enum, auto
from typing import Optional, List, Set, Dict, Any


class OpCode(Enum):
    ASSIGN = "ASSIGN"      # target = arg1
    ADD = "ADD"            # target = arg1 + arg2
    SUB = "SUB"            # target = arg1 - arg2
    MUL = "MUL"            # target = arg1 * arg2
    LOAD = "LOAD"          # target = MEM[arg1]
    STORE = "STORE"        # MEM[target] = arg1
    JUMP = "JUMP"          # goto target_label
    BEQ = "BEQ"            # if arg1 == arg2 goto target_label
    BNE = "BNE"            # if arg1 != arg2 goto target_label
    MOVE = "MOVE"          # target = arg1 (coalescing candidate)
    CALL = "CALL"          # target = CALL arg1
    RETURN = "RETURN"      # return arg1
    LABEL = "LABEL"        # label target_label


class Variable:
    """Represents a virtual register or variable in compiler IR."""
    def __init__(self, name: str, is_const: bool = False, const_val: Optional[int] = None):
        self.name: str = name
        self.is_const: bool = is_const
        self.const_val: Optional[int] = const_val
        self.physical_reg: Optional[str] = None
        self.spilled: bool = False
        self.spill_slot: Optional[int] = None

    def __repr__(self) -> str:
        return f"Var({self.name})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)


class Instruction:
    """Represents a single Three-Address Code (TAC) instruction."""
    def __init__(
        self,
        op: OpCode,
        target: Optional[Variable] = None,
        arg1: Optional[Variable] = None,
        arg2: Optional[Variable] = None,
        label: Optional[str] = None,
        target_label: Optional[str] = None,
        line_no: int = 0
    ):
        self.op: OpCode = op
        self.target: Optional[Variable] = target
        self.arg1: Optional[Variable] = arg1
        self.arg2: Optional[Variable] = arg2
        self.label: Optional[str] = label
        self.target_label: Optional[str] = target_label
        self.line_no: int = line_no

    def get_defined_vars(self) -> Set[Variable]:
        """Returns set of variables defined (written) by this instruction."""
        defs = set()
        if self.target and not self.target.is_const:
            if self.op not in (OpCode.JUMP, OpCode.BEQ, OpCode.BNE, OpCode.LABEL, OpCode.STORE):
                defs.add(self.target)
        return defs

    def get_used_vars(self) -> Set[Variable]:
        """Returns set of variables used (read) by this instruction."""
        uses = set()
        if self.arg1 and not self.arg1.is_const:
            uses.add(self.arg1)
        if self.arg2 and not self.arg2.is_const:
            uses.add(self.arg2)
        if self.op == OpCode.STORE and self.target and not self.target.is_const:
            uses.add(self.target)
        if self.op in (OpCode.BEQ, OpCode.BNE) and self.target and not self.target.is_const:
            uses.add(self.target)
        return uses

    def is_move(self) -> bool:
        """Returns True if instruction is a MOVE (coalescing candidate)."""
        return self.op == OpCode.MOVE

    def __repr__(self) -> str:
        if self.op == OpCode.LABEL:
            return f"{self.label}:"
        elif self.op == OpCode.ASSIGN:
            return f"L{self.line_no:02d}: {self.target.name} = {self.arg1.name}"
        elif self.op in (OpCode.ADD, OpCode.SUB, OpCode.MUL):
            op_str = "+" if self.op == OpCode.ADD else ("-" if self.op == OpCode.SUB else "*")
            return f"L{self.line_no:02d}: {self.target.name} = {self.arg1.name} {op_str} {self.arg2.name}"
        elif self.op == OpCode.MOVE:
            return f"L{self.line_no:02d}: MOVE {self.target.name} <- {self.arg1.name}"
        elif self.op == OpCode.LOAD:
            return f"L{self.line_no:02d}: {self.target.name} = LOAD [{self.arg1.name}]"
        elif self.op == OpCode.STORE:
            return f"L{self.line_no:02d}: STORE [{self.target.name}] = {self.arg1.name}"
        elif self.op == OpCode.JUMP:
            return f"L{self.line_no:02d}: JUMP {self.target_label}"
        elif self.op in (OpCode.BEQ, OpCode.BNE):
            comp = "==" if self.op == OpCode.BEQ else "!="
            return f"L{self.line_no:02d}: IF {self.arg1.name} {comp} {self.arg2.name} GOTO {self.target_label}"
        elif self.op == OpCode.CALL:
            return f"L{self.line_no:02d}: {self.target.name} = CALL {self.arg1.name}"
        elif self.op == OpCode.RETURN:
            val = self.arg1.name if self.arg1 else "void"
            return f"L{self.line_no:02d}: RETURN {val}"
        return f"L{self.line_no:02d}: {self.op.value}"


class Program:
    """Container for a TAC program or function body."""
    def __init__(self, name: str = "main"):
        self.name: str = name
        self.instructions: List[Instruction] = []
        self.variables: Dict[str, Variable] = {}

    def get_or_create_var(self, name: str, is_const: bool = False, const_val: Optional[int] = None) -> Variable:
        if name not in self.variables:
            self.variables[name] = Variable(name, is_const, const_val)
        return self.variables[name]

    def add_instruction(self, inst: Instruction):
        inst.line_no = len(self.instructions)
        self.instructions.append(inst)

    def get_non_const_vars(self) -> List[Variable]:
        return [v for v in self.variables.values() if not v.is_const]

    def __repr__(self) -> str:
        lines = [f"Program: {self.name}"]
        lines.extend([str(inst) for inst in self.instructions])
        return "\n".join(lines)
