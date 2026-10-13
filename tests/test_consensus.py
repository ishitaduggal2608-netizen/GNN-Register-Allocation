"""
Unit and Integration Tests for ConsensusArbiter and Safe Coalescing.
Verifies confidence-weighted soft voting, deterministic tie-breaking,
conservative Briggs/George coalescing validation, and conflict repair validity.
"""

import unittest
import torch
import networkx as nx

from compiler.ir import Program, Instruction, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from dataset.generator import SyntheticIRGenerator
from agents.rgcn_agent import RelationalAgent
from agents.gat_agent import AttentionAgent
from agents.sage_agent import NeighbourhoodAgent
from agents.gcn_agent import PressureAgent
from agents.consensus import ConsensusArbiter, ConsensusResult


class TestConsensusArbiter(unittest.TestCase):
    def setUp(self):
        generator = SyntheticIRGenerator(seed=42)
        self.prog = generator.generate_program(num_vars=12, num_instructions=30)
        self.cfg = ControlFlowGraph(self.prog)
        self.liveness = LivenessAnalyzer(self.cfg)
        self.ig = InterferenceGraph(self.prog, self.cfg, self.liveness)
        self.K = 4

        self.rgcn = RelationalAgent(num_registers=self.K, in_channels=6, hidden_dim=32)
        self.gat = AttentionAgent(num_registers=self.K, in_channels=6, hidden_dim=32)
        self.sage = NeighbourhoodAgent(num_registers=self.K, in_channels=6, hidden_dim=32)
        self.gcn = PressureAgent(num_registers=self.K, in_channels=6, hidden_dim=32)

    def test_multi_agent_arbitration(self):
        pred_rgcn = self.rgcn.predict_from_ig(self.ig)
        pred_gat = self.gat.predict_from_ig(self.ig)
        pred_sage = self.sage.predict_from_ig(self.ig)
        pred_gcn = self.gcn.predict_from_ig(self.ig)

        predictions = {
            "rgcn": pred_rgcn,
            "gat": pred_gat,
            "sage": pred_sage,
            "gcn": pred_gcn
        }

        arbiter = ConsensusArbiter(num_registers=self.K)
        result = arbiter.arbitrate(predictions, self.ig)

        self.assertIsInstance(result, ConsensusResult)
        N = len(self.ig.variables)
        self.assertEqual(len(result.register_assignment), N)

        # Zero coloring conflicts on assigned physical registers
        conflicts = 0
        for u, v in self.ig.interference_edges:
            c_u = result.register_assignment.get(u)
            c_v = result.register_assignment.get(v)
            if c_u and c_v and c_u != "SPILL" and c_v != "SPILL":
                if c_u == c_v:
                    conflicts += 1
        self.assertEqual(conflicts, 0, "Consensus allocator must have zero coloring conflicts")

    def test_deterministic_tie_breaking(self):
        # Two identical predictions must yield identical results across multiple runs
        pred_rgcn = self.rgcn.predict_from_ig(self.ig)
        pred_gat = self.gat.predict_from_ig(self.ig)
        predictions = {"rgcn": pred_rgcn, "gat": pred_gat}

        arbiter = ConsensusArbiter(num_registers=self.K)
        res1 = arbiter.arbitrate(predictions, self.ig)
        res2 = arbiter.arbitrate(predictions, self.ig)

        self.assertEqual(res1.register_assignment, res2.register_assignment)
        self.assertEqual(res1.spilled_vars, res2.spilled_vars)
        self.assertEqual(res1.coalesced_pairs, res2.coalesced_pairs)

    def test_conservative_coalesce_safety(self):
        arbiter = ConsensusArbiter(num_registers=self.K)
        g = nx.Graph()
        # Create a K-clique (size 4: a, b, c, d)
        for u in ["a", "b", "c", "d"]:
            for v in ["a", "b", "c", "d"]:
                if u != v:
                    g.add_edge(u, v)

        # Directly interfering nodes must NEVER be coalesced
        self.assertFalse(arbiter.evaluate_conservative_coalesce("a", "b", g, self.K))

        # Separate node e and f with no neighbors: safe to coalesce
        g.add_node("e")
        g.add_node("f")
        self.assertTrue(arbiter.evaluate_conservative_coalesce("e", "f", g, self.K))


if __name__ == "__main__":
    unittest.main()
