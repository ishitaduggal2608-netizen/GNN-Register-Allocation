"""
Unit Tests for Compiler Subsystem (IR, CFG, Liveness, Interference Graph).
"""

import unittest
from compiler.ir import Program, Instruction, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph


class TestCompilerSubsystem(unittest.TestCase):
    def setUp(self):
        self.prog = Program(name="test_prog")
        v0 = self.prog.get_or_create_var("v0")
        v1 = self.prog.get_or_create_var("v1")
        v2 = self.prog.get_or_create_var("v2")
        c1 = self.prog.get_or_create_var("const_1", is_const=True, const_val=1)
        c5 = self.prog.get_or_create_var("const_5", is_const=True, const_val=5)

        self.prog.add_instruction(Instruction(OpCode.ASSIGN, target=v0, arg1=c1))
        self.prog.add_instruction(Instruction(OpCode.ASSIGN, target=v1, arg1=c5))
        self.prog.add_instruction(Instruction(OpCode.ADD, target=v2, arg1=v0, arg2=v1))
        self.prog.add_instruction(Instruction(OpCode.RETURN, arg1=v2))

    def test_cfg_construction(self):
        cfg = ControlFlowGraph(self.prog)
        self.assertTrue(len(cfg.blocks) > 0)
        self.assertEqual(cfg.entry_block.block_id, 0)

    def test_liveness_analysis(self):
        cfg = ControlFlowGraph(self.prog)
        liveness = LivenessAnalyzer(cfg)
        self.assertTrue(len(liveness.inst_live_out) > 0)

    def test_interference_graph_building(self):
        cfg = ControlFlowGraph(self.prog)
        liveness = LivenessAnalyzer(cfg)
        ig = InterferenceGraph(self.prog, cfg, liveness)

        self.assertEqual(len(ig.variables), 3)
        features = ig.get_node_features()
        self.assertIn("v0", features)
        self.assertIn("v1", features)
        self.assertIn("v2", features)

        feat_matrix = ig.get_feature_matrix()
        self.assertEqual(feat_matrix.shape, (3, 6))

    def test_cached_degrees_match_naive_edge_scans(self):
        prog = Program(name="degree_prog")
        a = prog.get_or_create_var("a")
        b = prog.get_or_create_var("b")
        c = prog.get_or_create_var("c")
        d = prog.get_or_create_var("d")
        one = prog.get_or_create_var("const_1", is_const=True, const_val=1)

        prog.add_instruction(Instruction(OpCode.ASSIGN, target=a, arg1=one))
        prog.add_instruction(Instruction(OpCode.ASSIGN, target=b, arg1=one))
        prog.add_instruction(Instruction(OpCode.MOVE, target=c, arg1=a))
        prog.add_instruction(Instruction(OpCode.ADD, target=d, arg1=c, arg2=b))
        prog.add_instruction(Instruction(OpCode.RETURN, arg1=d))

        cfg = ControlFlowGraph(prog)
        liveness = LivenessAnalyzer(cfg)
        ig = InterferenceGraph(prog, cfg, liveness)

        for var in ig.variables:
            name = var.name
            naive_interf_degree = sum(1 for u, v in ig.interference_edges if u == name or v == name)
            naive_move_degree = sum(1 for u, v in ig.coalescing_edges if u == name or v == name)
            self.assertEqual(ig.get_interference_degree(name), naive_interf_degree)
            self.assertEqual(ig.get_move_degree(name), naive_move_degree)


if __name__ == "__main__":
    unittest.main()
