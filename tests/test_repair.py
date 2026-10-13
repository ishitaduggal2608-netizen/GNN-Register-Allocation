"""
Unit tests for JSON Exporter/Importer, Post-Hoc Conflict Repair, and Coalesce Classifier.
"""

import os
import unittest
import torch
import numpy as np

from compiler.ir import Program, Instruction, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import ChaitinBriggsAllocator
from compiler.export import export_graph_to_json, load_graph_from_json
from models.repair import repair_conflicts
from models.gnn_allocator import (
    GCNSpillPredictor,
    SAGESpillPredictor,
    GATSpillPredictor,
    RelationalGNNRegisterAllocator,
    CoalesceClassifier
)


class TestRepairAndExport(unittest.TestCase):
    def setUp(self):
        self.prog = Program("test_func")
        a = self.prog.get_or_create_var("a")
        b = self.prog.get_or_create_var("b")
        c = self.prog.get_or_create_var("c")
        d = self.prog.get_or_create_var("d")
        v1 = self.prog.get_or_create_var("const_1", is_const=True, const_val=1)

        self.prog.add_instruction(Instruction(OpCode.ASSIGN, target=a, arg1=v1))
        self.prog.add_instruction(Instruction(OpCode.ASSIGN, target=b, arg1=v1))
        self.prog.add_instruction(Instruction(OpCode.ADD, target=c, arg1=a, arg2=b))
        self.prog.add_instruction(Instruction(OpCode.MOVE, target=d, arg1=c))
        self.prog.add_instruction(Instruction(OpCode.RETURN, arg1=d))

        self.cfg = ControlFlowGraph(self.prog)
        self.liveness = LivenessAnalyzer(self.cfg)
        self.ig = InterferenceGraph(self.prog, self.cfg, self.liveness)
        self.allocator = ChaitinBriggsAllocator(num_registers=2)
        self.gt_res = self.allocator.allocate(self.ig)

    def test_json_export_import(self):
        tmp_json = "data/raw_graphs/test_sample.json"
        export_payload = export_graph_to_json(
            self.ig, self.gt_res, graph_id="test_001", num_registers=2, filepath=tmp_json
        )
        self.assertEqual(export_payload["graph_id"], "test_001")
        self.assertEqual(export_payload["num_registers"], 2)
        self.assertIn("nodes", export_payload)
        self.assertIn("interference_edges", export_payload)

        # Import verification
        loaded = load_graph_from_json(tmp_json)
        self.assertEqual(loaded["graph_id"], "test_001")
        self.assertEqual(len(loaded["nodes"]), len(export_payload["nodes"]))

        # Cleanup
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    def test_greedy_conflict_repair(self):
        # Create a graph with 3 mutually interfering nodes assigned color 0 (conflict)
        adj = torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        pred_labels = torch.tensor([0, 0, 0], dtype=torch.long)  # All color 0 (conflict!)

        repaired, conflicts = repair_conflicts(adj, pred_labels, num_registers=2, spill_class=2)
        self.assertGreater(conflicts, 0)
        # Check no two adjacent nodes have same non-spill color
        c0, c1, c2 = repaired[0].item(), repaired[1].item(), repaired[2].item()
        if c0 < 2 and c1 < 2:
            self.assertNotEqual(c0, c1)
        if c1 < 2 and c2 < 2:
            self.assertNotEqual(c1, c2)

    def test_model_forward_passes(self):
        x = torch.randn(4, 6)
        adj = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

        for model_cls in [GCNSpillPredictor, SAGESpillPredictor, GATSpillPredictor, RelationalGNNRegisterAllocator]:
            model = model_cls(in_channels=6, num_registers=2)
            logits, scores = model(x, adj, adj)
            self.assertEqual(logits.shape, (4, 3))  # 2 registers + 1 spill class
            self.assertEqual(scores.shape, (4, 4))

    def test_coalesce_classifier(self):
        embeds = torch.randn(4, 64)
        move_edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        classifier = CoalesceClassifier(node_embed_dim=64)
        logits = classifier(embeds, move_edges)
        self.assertEqual(logits.shape, (2,))


if __name__ == "__main__":
    unittest.main()
