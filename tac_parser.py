"""Parser for a practical subset of Three-Address Code used by the web demo.

Supported examples:
    x = 5
    y = x
    z = x + y   # also - and *
    L1:
    goto L1
    if x == y goto L1
    if x != 0 goto L1
    return z

This is intentionally a small TAC parser, not a general C compiler.
"""
import re
from compiler.ir import Program, Instruction, OpCode

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
CONST = r"-?\d+"

class TACParseError(ValueError):
    pass


def parse_tac(source: str) -> Program:
    if not isinstance(source, str) or not source.strip():
        raise TACParseError("Enter at least one TAC instruction.")

    program = Program(name="web_input")
    lines = []
    for line_no, raw in enumerate(source.splitlines(), 1):
        line = raw.split("#", 1)[0].split("//", 1)[0].strip()
        if line:
            lines.append((line_no, line))

    def value(token: str):
        token = token.strip()
        if re.fullmatch(CONST, token):
            key = f"const_neg_{token[1:]}" if token.startswith("-") else f"const_{token}"
            if key not in program.variables:
                return program.get_or_create_var(key, is_const=True, const_val=int(token))
            return program.variables[key]
        if not re.fullmatch(IDENT, token):
            raise TACParseError(f"Invalid value '{token}'. Use a variable name or integer constant.")
        return program.get_or_create_var(token)

    # Validate labels first so branch targets can be checked.
    labels = set()
    for original_no, line in lines:
        m = re.fullmatch(rf"({IDENT}):", line)
        if m:
            if m.group(1) in labels:
                raise TACParseError(f"Line {original_no}: duplicate label '{m.group(1)}'.")
            labels.add(m.group(1))

    for original_no, line in lines:
        try:
            m = re.fullmatch(rf"({IDENT}):", line)
            if m:
                program.add_instruction(Instruction(OpCode.LABEL, label=m.group(1)))
                continue

            m = re.fullmatch(rf"goto\s+({IDENT})", line, re.IGNORECASE)
            if m:
                if m.group(1) not in labels:
                    raise TACParseError(f"Line {original_no}: label '{m.group(1)}' does not exist.")
                program.add_instruction(Instruction(OpCode.JUMP, target_label=m.group(1)))
                continue

            m = re.fullmatch(rf"if\s+({IDENT}|{CONST})\s*(==|!=)\s*({IDENT}|{CONST})\s+goto\s+({IDENT})", line, re.IGNORECASE)
            if m:
                left, op, right, label = m.groups()
                if label not in labels:
                    raise TACParseError(f"Line {original_no}: label '{label}' does not exist.")
                program.add_instruction(Instruction(OpCode.BEQ if op == "==" else OpCode.BNE,
                                                    arg1=value(left), arg2=value(right), target_label=label))
                continue

            m = re.fullmatch(rf"return(?:\s+({IDENT}|{CONST}))?", line, re.IGNORECASE)
            if m:
                program.add_instruction(Instruction(OpCode.RETURN, arg1=value(m.group(1)) if m.group(1) else None))
                continue

            m = re.fullmatch(rf"({IDENT})\s*=\s*({IDENT}|{CONST})\s*([+*-])\s*({IDENT}|{CONST})", line)
            if m:
                target, left, op, right = m.groups()
                opcode = {"+": OpCode.ADD, "-": OpCode.SUB, "*": OpCode.MUL}[op]
                program.add_instruction(Instruction(opcode, target=value(target), arg1=value(left), arg2=value(right)))
                continue

            m = re.fullmatch(rf"({IDENT})\s*=\s*LOAD\s*\[\s*({IDENT}|{CONST})\s*\]", line, re.IGNORECASE)
            if m:
                target, address = m.groups()
                program.add_instruction(Instruction(OpCode.LOAD, target=value(target), arg1=value(address)))
                continue

            m = re.fullmatch(rf"STORE\s*\[\s*({IDENT}|{CONST})\s*\]\s*=\s*({IDENT}|{CONST})", line, re.IGNORECASE)
            if m:
                address, stored = m.groups()
                program.add_instruction(Instruction(OpCode.STORE, target=value(address), arg1=value(stored)))
                continue

            m = re.fullmatch(rf"({IDENT})\s*=\s*CALL\s+({IDENT})", line, re.IGNORECASE)
            if m:
                target, function = m.groups()
                program.add_instruction(Instruction(OpCode.CALL, target=value(target), arg1=value(function)))
                continue

            m = re.fullmatch(rf"({IDENT})\s*=\s*({IDENT}|{CONST})", line)
            if m:
                target_name, src_name = m.groups()
                target, src = value(target_name), value(src_name)
                if src.is_const:
                    program.add_instruction(Instruction(OpCode.ASSIGN, target=target, arg1=src))
                else:
                    program.add_instruction(Instruction(OpCode.MOVE, target=target, arg1=src))
                continue

            raise TACParseError(f"Line {original_no}: unsupported instruction: '{line}'.")
        except TACParseError:
            raise
        except Exception as exc:
            raise TACParseError(f"Line {original_no}: could not parse '{line}': {exc}") from exc

    if not program.instructions:
        raise TACParseError("No valid TAC instructions were found.")
    return program
