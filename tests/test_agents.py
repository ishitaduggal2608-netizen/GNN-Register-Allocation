"""
Unit and Integration Tests for Pattern Detection Agents and Registry.
Tests base class contracts, RelationalAgent (Agent 1), tensor extraction,
entropy-confidence calculation, and backward-compatible checkpoint loading.
"""

import os
import unittest
import torch
import numpy as np

from compiler.ir import Program, Instruction, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from dataset.generator import SyntheticIRGenerator
from agents.base import (
    PatternAgent,
    AgentPrediction,
    extract_graph_tensors,
    compute_normalized_entropy_confidence
)
from agents.rgcn_agent import RelationalAgent
from agents.registry import (
    get_agent_class,
    create_agent,
    load_agent_checkpoint,
    list_registered_agents
)


class TestAgentsBase(unittest.TestCase):
    def setUp(self):
        generator = SyntheticIRGenerator(seed=42)
        self.prog = generator.generate_program(num_vars=10, num_instructions=25)
        self.cfg = ControlFlowGraph(self.prog)
        self.liveness = LivenessAnalyzer(self.cfg)
        self.ig = InterferenceGraph(self.prog, self.cfg, self.liveness)

    def test_extract_graph_tensors(self):
        x, interf_adj, coal_adj, var_names = extract_graph_tensors(self.ig)
        N = len(self.ig.variables)

        self.assertEqual(x.shape, (N, 6))
        self.assertEqual(interf_adj.shape, (N, N))
        self.assertEqual(coal_adj.shape, (N, N))
        self.assertEqual(len(var_names), N)

        # Symmetry checks
        self.assertTrue(torch.allclose(interf_adj, interf_adj.T))
        self.assertTrue(torch.allclose(coal_adj, coal_adj.T))

    def test_normalized_entropy_confidence(self):
        # Uniform distribution -> max entropy -> confidence 0.0
        uniform = torch.full((3, 5), 0.2)
        conf_uniform = compute_normalized_entropy_confidence(uniform)
        self.assertTrue(torch.allclose(conf_uniform, torch.zeros(3), atol=1e-5))

        # One-hot distribution -> min entropy -> confidence 1.0
        one_hot = torch.zeros((3, 5))
        one_hot[0, 0] = 1.0
        one_hot[1, 2] = 1.0
        one_hot[2, 4] = 1.0
        conf_one_hot = compute_normalized_entropy_confidence(one_hot)
        self.assertTrue(torch.allclose(conf_one_hot, torch.ones(3), atol=1e-5))


class TestRelationalAgent(unittest.TestCase):
    def setUp(self):
        generator = SyntheticIRGenerator(seed=42)
        self.prog = generator.generate_program(num_vars=10, num_instructions=25)
        self.cfg = ControlFlowGraph(self.prog)
        self.liveness = LivenessAnalyzer(self.cfg)
        self.ig = InterferenceGraph(self.prog, self.cfg, self.liveness)

    def test_forward_and_prediction_shapes(self):
        agent = RelationalAgent(num_registers=4, in_channels=6, hidden_dim=32)
        pred = agent.predict_from_ig(self.ig)
        N = len(self.ig.variables)
        K = 4

        self.assertIsInstance(pred, AgentPrediction)
        self.assertEqual(pred.color_probs.shape, (N, K + 1))
        self.assertEqual(pred.spill_prob.shape, (N,))
        self.assertEqual(pred.pred_classes.shape, (N,))
        self.assertEqual(pred.confidence.shape, (N,))
        self.assertIsNotNone(pred.coalesce_score)
        self.assertEqual(pred.coalesce_score.shape, (N, N))

        # Confidences bounded in [0, 1]
        self.assertTrue(torch.all(pred.confidence >= 0.0))
        self.assertTrue(torch.all(pred.confidence <= 1.0))

    def test_explain_payload(self):
        agent = RelationalAgent(num_registers=4, in_channels=6, hidden_dim=32)
        x, interf_adj, coal_adj, _ = extract_graph_tensors(self.ig)
        exp = agent.explain(x, interf_adj, coal_adj, ig=self.ig)

        self.assertEqual(exp["agent"], "RelationalAgent")
        self.assertEqual(exp["architecture"], "rgcn")
        self.assertIn("move_chains_detected", exp)
        self.assertIn("top_coalescing_candidates", exp)
        self.assertIn("relational_edge_ratio", exp)

    def test_save_and_load_round_trip(self):
        agent = RelationalAgent(num_registers=4, in_channels=6, hidden_dim=32)
        checkpoint_path = "checkpoints/test_rgcn.pt"

        try:
            agent.save(checkpoint_path)
            loaded_agent = load_agent_checkpoint(checkpoint_path)

            self.assertIsInstance(loaded_agent, RelationalAgent)
            self.assertEqual(loaded_agent.num_registers, 4)
            self.assertEqual(loaded_agent.hidden_dim, 32)

            pred_orig = agent.predict_from_ig(self.ig)
            pred_loaded = loaded_agent.predict_from_ig(self.ig)

            self.assertTrue(torch.allclose(pred_orig.color_probs, pred_loaded.color_probs, atol=1e-5))
            self.assertTrue(torch.allclose(pred_orig.spill_prob, pred_loaded.spill_prob, atol=1e-5))
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def test_load_legacy_checkpoint(self):
        # Must load legacy gnn_allocator.pt without errors
        legacy_path = "gnn_allocator.pt"
        if os.path.exists(legacy_path):
            agent = load_agent_checkpoint(legacy_path)
            self.assertIsInstance(agent, RelationalAgent)
            pred = agent.predict_from_ig(self.ig)
            self.assertEqual(pred.color_probs.shape[1], 5)  # K=4 -> 5 classes


class TestRegistry(unittest.TestCase):
    def test_registered_agents(self):
        cls = get_agent_class("rgcn")
        self.assertEqual(cls, RelationalAgent)
        cls_alias = get_agent_class("r-gcn")
        self.assertEqual(cls_alias, RelationalAgent)

    def test_create_agent(self):
        agent = create_agent("rgcn", num_registers=4)
        self.assertIsInstance(agent, RelationalAgent)


if __name__ == "__main__":
    unittest.main()
