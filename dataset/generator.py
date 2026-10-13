"""
Synthetic IR Generator for Compiler Register Allocation benchmarking.
Generates synthetic functions with loops, conditionals, move chains, and variable interferences.
"""

import random
from typing import List, Tuple
from compiler.ir import Program, Instruction, OpCode, Variable
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import (
    ChaitinBriggsAllocator,
    AllocationResult,
    canonicalize_allocation_result,
    variable_order_from_program
)


class SyntheticIRGenerator:
    """Generates synthetic TAC programs for training and evaluating GNN register allocators."""
    def __init__(self, seed: int = 42):
        self.seed: int = seed
        random.seed(seed)

    def generate_program(
        self,
        num_vars: int = 15,
        loop_prob: float = 0.4,
        move_prob: float = 0.3,
        num_instructions: int = 30
    ) -> Program:
        """Generates a synthetic TAC program function."""
        prog = Program(name=f"func_v{num_vars}_{random.randint(1000, 9999)}")

        # Create variable pool
        var_names = [f"v{i}" for i in range(num_vars)]
        vars_list = [prog.get_or_create_var(name) for name in var_names]

        # Initialize variables
        for v in vars_list[:min(3, num_vars)]:
            c = prog.get_or_create_var(f"const_{random.randint(1, 100)}", is_const=True, const_val=random.randint(1, 10))
            prog.add_instruction(Instruction(OpCode.ASSIGN, target=v, arg1=c))

        label_counter = 0

        inst_count = 0
        while inst_count < num_instructions:
            # Decide construct: Loop, Branch, Move, or Arithmetic
            r = random.random()

            if r < loop_prob and inst_count + 6 < num_instructions:
                # Generate a loop:
                # L_start:
                # v_cond = v_i - const
                # BEQ v_cond, const_0 -> L_end
                # ... loop body ...
                # JUMP L_start
                # L_end:
                label_start = f"L_loop_start_{label_counter}"
                label_end = f"L_loop_end_{label_counter}"
                label_counter += 1

                v_iter = random.choice(vars_list)
                v_target = random.choice(vars_list)
                c_step = prog.get_or_create_var(f"const_1", is_const=True, const_val=1)

                prog.add_instruction(Instruction(OpCode.LABEL, label=label_start))
                prog.add_instruction(Instruction(OpCode.SUB, target=v_target, arg1=v_iter, arg2=c_step))
                prog.add_instruction(Instruction(OpCode.BEQ, arg1=v_target, arg2=c_step, target_label=label_end))

                # Loop body instructions
                for _ in range(random.randint(2, 4)):
                    v_a = random.choice(vars_list)
                    v_b = random.choice(vars_list)
                    v_c = random.choice(vars_list)
                    op = random.choice([OpCode.ADD, OpCode.SUB, OpCode.MUL])
                    prog.add_instruction(Instruction(op, target=v_a, arg1=v_b, arg2=v_c))
                    inst_count += 1

                prog.add_instruction(Instruction(OpCode.JUMP, target_label=label_start))
                prog.add_instruction(Instruction(OpCode.LABEL, label=label_end))
                inst_count += 5

            elif r < loop_prob + move_prob:
                # Generate MOVE instructions (affinity edges for coalescing)
                v_dst = random.choice(vars_list)
                v_src = random.choice([v for v in vars_list if v != v_dst] or vars_list)
                prog.add_instruction(Instruction(OpCode.MOVE, target=v_dst, arg1=v_src))
                inst_count += 1

            else:
                # Generate standard arithmetic operation
                v_dst = random.choice(vars_list)
                v_arg1 = random.choice(vars_list)
                v_arg2 = random.choice(vars_list)
                op = random.choice([OpCode.ADD, OpCode.SUB, OpCode.MUL])
                prog.add_instruction(Instruction(op, target=v_dst, arg1=v_arg1, arg2=v_arg2))
                inst_count += 1

        # Return last variable
        ret_var = random.choice(vars_list)
        prog.add_instruction(Instruction(OpCode.RETURN, arg1=ret_var))

        return prog

    def generate_dataset(
        self,
        num_samples: int = 50,
        min_vars: int = 8,
        max_vars: int = 25,
        num_registers: int = 4
    ) -> List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, AllocationResult]]:
        """Generates a batch of dataset samples containing TAC, CFG, IG, and ground-truth Chaitin-Briggs allocation."""
        dataset = []
        for i in range(num_samples):
            n_vars = random.randint(min_vars, max_vars)
            prog = self.generate_program(num_vars=n_vars, num_instructions=random.randint(20, 45))

            cfg = ControlFlowGraph(prog)
            liveness = LivenessAnalyzer(cfg)
            ig = InterferenceGraph(prog, cfg, liveness)

            allocator = ChaitinBriggsAllocator(num_registers=num_registers)
            ground_truth = allocator.allocate(ig)
            # Graph coloring is permutation-invariant; canonical labels keep equivalent
            # Chaitin-Briggs colorings from becoming contradictory CE targets.
            ground_truth = canonicalize_allocation_result(ground_truth, variable_order_from_program(prog))

            dataset.append((prog, cfg, liveness, ig, ground_truth))

        return dataset

    def export_synthetic_dataset_to_json(
        self,
        num_samples: int = 50,
        output_dir: str = "data/raw_graphs",
        num_registers: int = 4
    ) -> List[str]:
        """Generates synthetic dataset and exports all samples as JSON files in Phase 2.5 schema."""
        import os
        from compiler.export import export_graph_to_json

        os.makedirs(output_dir, exist_ok=True)
        samples = self.generate_dataset(num_samples=num_samples, num_registers=num_registers)
        filepaths = []

        for i, (prog, cfg, liveness, ig, gt) in enumerate(samples):
            graph_id = f"func_{i:03d}"
            filepath = os.path.join(output_dir, f"{graph_id}.json")
            export_graph_to_json(ig, gt, graph_id=graph_id, num_registers=num_registers, filepath=filepath)
            filepaths.append(filepath)

        return filepaths
