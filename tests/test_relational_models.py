"""
Unit & Integration Tests for Multi-Relational Allocator Architectures:
1. Relational GCN (R-GCN)
2. Relational GAT (R-GAT)
3. Relational GraphSAGE (R-SAGE)
4. Relational GIN (R-GIN)
Verifies forward pass output shapes, GraphColoringLoss computation,
save/load checkpoint round-trips, and ModelComparator execution.
"""

import os
import tempfile
import unittest
import torch

from models.gnn_allocator import RelationalGNNRegisterAllocator
from models.rgat_allocator import RelationalGATRegisterAllocator
from models.rsage_allocator import RelationalSAGERegisterAllocator
from models.gin_allocator import RelationalGINRegisterAllocator
from models.base_allocator import GraphColoringLoss
from models.trainer import GNNTrainer, create_model_by_type
from evaluation.model_comparator import ModelComparator
from compiler.ir import Program
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from dataset.generator import SyntheticIRGenerator


class TestRelationalModels(unittest.TestCase):
    def setUp(self):
        self.N = 5
        self.K = 4
        self.x = torch.randn(self.N, 6)
        self.interf_adj = torch.zeros(self.N, self.N)
        for i in range(self.N):
            self.interf_adj[i, (i + 1) % self.N] = 1.0
            self.interf_adj[(i + 1) % self.N, i] = 1.0

        self.coal_adj = torch.zeros(self.N, self.N)
        self.coal_adj[0, 2] = 1.0
        self.coal_adj[2, 0] = 1.0

        self.target_colors = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self.coalesce_labels = self.coal_adj.clone()

    def _check_forward_shapes(self, model_cls, name):
        model = model_cls(in_node_features=6, hidden_dim=32, num_registers=self.K, num_layers=2)
        model.eval()
        with torch.no_grad():
            color_logits, coalesce_logits = model(self.x, self.interf_adj, self.coal_adj)
        self.assertEqual(color_logits.shape, (self.N, self.K + 1), f"{name} color_logits wrong shape")
        self.assertEqual(coalesce_logits.shape, (self.N, self.N), f"{name} coalesce_logits wrong shape")
        embeds = model.get_node_embeddings(self.x, self.interf_adj, self.coal_adj)
        self.assertEqual(embeds.shape, (self.N, 32), f"{name} embeds wrong shape")

    def test_rgcn_forward_shapes(self):
        self._check_forward_shapes(RelationalGNNRegisterAllocator, "rgcn")

    def test_rgat_forward_shapes(self):
        self._check_forward_shapes(RelationalGATRegisterAllocator, "rgat")

    def test_rsage_forward_shapes(self):
        self._check_forward_shapes(RelationalSAGERegisterAllocator, "rsage")

    def test_gin_forward_shapes(self):
        self._check_forward_shapes(RelationalGINRegisterAllocator, "gin")

    def _check_loss(self, model_cls):
        model = model_cls(in_node_features=6, hidden_dim=32, num_registers=self.K, num_layers=2)
        criterion = GraphColoringLoss(conflict_weight=0.5, coalesce_weight=0.3)
        color_logits, coalesce_logits = model(self.x, self.interf_adj, self.coal_adj)
        loss, loss_dict = criterion(
            color_logits, coalesce_logits, self.target_colors, self.interf_adj, self.coal_adj, self.coalesce_labels
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("conflict_loss", loss_dict)
        self.assertIn("coalesce_loss", loss_dict)

        loss.backward()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_rgcn_loss_computation(self):
        self._check_loss(RelationalGNNRegisterAllocator)

    def test_rgat_loss_computation(self):
        self._check_loss(RelationalGATRegisterAllocator)

    def test_rsage_loss_computation(self):
        self._check_loss(RelationalSAGERegisterAllocator)

    def test_gin_loss_computation(self):
        self._check_loss(RelationalGINRegisterAllocator)

    def _check_save_load(self, arch_key):
        model = create_model_by_type(arch_key, in_channels=6, hidden_dim=32, num_registers=self.K)
        trainer = GNNTrainer(model=model)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, f"{arch_key}_test.pt")
            trainer.save_checkpoint(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))

            loaded_model = GNNTrainer.load_checkpoint(ckpt_path)
            self.assertEqual(loaded_model.num_registers, self.K)

            model.eval()
            loaded_model.eval()
            with torch.no_grad():
                c1, _ = model(self.x, self.interf_adj, self.coal_adj)
                c2, _ = loaded_model(self.x, self.interf_adj, self.coal_adj)
            self.assertTrue(torch.allclose(c1, c2, atol=1e-5))

    def test_rgcn_save_load(self):
        self._check_save_load("rgcn")

    def test_rgat_save_load(self):
        self._check_save_load("rgat")

    def test_rsage_save_load(self):
        self._check_save_load("rsage")

    def test_gin_save_load(self):
        self._check_save_load("gin")

    def test_model_comparator_execution(self):
        generator = SyntheticIRGenerator(seed=42)
        prog = generator.generate_program(num_vars=10, num_instructions=20)
        cfg = ControlFlowGraph(prog)
        liveness = LivenessAnalyzer(cfg)
        ig = InterferenceGraph(prog, cfg, liveness)

        K = 4
        models = {
            "rgcn": create_model_by_type("rgcn", in_channels=6, hidden_dim=32, num_registers=K),
            "rgat": create_model_by_type("rgat", in_channels=6, hidden_dim=32, num_registers=K),
            "rsage": create_model_by_type("rsage", in_channels=6, hidden_dim=32, num_registers=K),
            "gin": create_model_by_type("gin", in_channels=6, hidden_dim=32, num_registers=K),
        }

        comparator = ModelComparator(num_registers=K)
        res = comparator.compare_single_program(models, ig, prog)

        self.assertIn("model_keys", res)
        for key in ["rgcn", "rgat", "rsage", "gin", "chaitin_briggs", "random"]:
            self.assertIn(key, res["model_keys"])

        self.assertIn("register_agreement_matrix", res)
        for m in res["model_keys"]:
            self.assertEqual(res["register_agreement_matrix"][m][m], 100.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_path = os.path.join(tmpdir, "test_comp.png")
            comparator.generate_comparison_plot(res, output_path=fig_path)
            self.assertTrue(os.path.exists(fig_path))
            self.assertGreater(os.path.getsize(fig_path), 10000)

            json_path = os.path.join(tmpdir, "test_comp.json")
            comparator.export_comparison_json(res, prog, cfg, ig, output_path=json_path)
            self.assertTrue(os.path.exists(json_path))
            self.assertGreater(os.path.getsize(json_path), 1000)


if __name__ == "__main__":
    unittest.main()
