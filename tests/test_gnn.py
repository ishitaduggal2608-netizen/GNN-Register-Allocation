"""
Unit Tests for Relational GNN Model Architecture and Loss Function.
"""

import unittest
import torch
from models.gnn_allocator import RelationalGNNRegisterAllocator, GraphColoringLoss


class TestGNNModel(unittest.TestCase):
    def test_forward_pass(self):
        N = 5
        K = 4
        in_dim = 6
        hidden_dim = 16

        model = RelationalGNNRegisterAllocator(in_node_features=in_dim, hidden_dim=hidden_dim, num_registers=K)

        x = torch.randn(N, in_dim)
        interf_adj = torch.eye(N)
        coal_adj = torch.zeros(N, N)

        color_logits, coalesce_logits = model(x, interf_adj, coal_adj)

        self.assertEqual(color_logits.shape, (N, K + 1))
        self.assertEqual(coalesce_logits.shape, (N, N))

    def test_loss_function(self):
        N = 4
        K = 4
        criterion = GraphColoringLoss()

        color_logits = torch.randn(N, K + 1)
        coalesce_logits = torch.randn(N, N)
        target_colors = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        interf_adj = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float32)
        coal_adj = torch.zeros((N, N), dtype=torch.float32)
        coalesce_labels = torch.zeros((N, N), dtype=torch.float32)

        loss, loss_dict = criterion(color_logits, coalesce_logits, target_colors, interf_adj, coal_adj, coalesce_labels)

        self.assertTrue(loss.item() > 0)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("conflict_loss", loss_dict)

    def test_loss_accepts_raw_logits_and_decreases(self):
        N = 4
        K = 2
        criterion = GraphColoringLoss(conflict_weight=0.0, coalesce_weight=1.0, pos_weight=torch.tensor([2.0]))

        color_logits = torch.nn.Parameter(torch.zeros(N, K + 1))
        coalesce_logits = torch.nn.Parameter(torch.tensor([
            [0.0, -3.0, 0.0, 0.0],
            [-3.0, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, -3.0],
            [0.0, 0.0, -3.0, 0.0],
        ]))
        target_colors = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        interf_adj = torch.zeros((N, N), dtype=torch.float32)
        coal_adj = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        coalesce_labels = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        optimizer = torch.optim.SGD([color_logits, coalesce_logits], lr=0.5)
        initial_loss = None
        final_loss = None
        for step in range(8):
            optimizer.zero_grad()
            loss, _ = criterion(
                color_logits,
                coalesce_logits,
                target_colors,
                interf_adj,
                coal_adj,
                coalesce_labels
            )
            self.assertTrue(torch.isfinite(loss))
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
