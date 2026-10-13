
import unittest
import torch

from models.repair import repair_conflicts


class TestAllocationConstraints(unittest.TestCase):

    def verify_assignments(self, adjacency, assignments, num_registers):
        """
        Verify that each interfering pair either uses different
        physical registers or has at least one spilled endpoint.
        """
        spill_label = num_registers

        self.assertEqual(len(assignments), adjacency.shape[0])

        for i in range(len(assignments)):
            self.assertIn(
                assignments[i],
                list(range(num_registers)) + [spill_label],
                f"Invalid register assignment for node {i}",
            )

        for i in range(adjacency.shape[0]):
            for j in range(i + 1, adjacency.shape[1]):
                if adjacency[i, j].item() != 0:
                    self.assertTrue(
                        assignments[i] == spill_label
                        or assignments[j] == spill_label
                        or assignments[i] != assignments[j],
                        f"Interference violation between nodes {i} and {j}",
                    )

    def test_all_register_counts(self):
        # Five variables with four interference edges.
        adjacency = torch.tensor([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)

        # All variables initially predict R0.
        predictions = torch.zeros(5, dtype=torch.long)

        for k in (1, 2, 3, 4):
            with self.subTest(register_count=k):
                repaired, _ = repair_conflicts(
                    adjacency,
                    predictions,
                    num_registers=k,
                    spill_class=k,
                )

                self.verify_assignments(
                    adjacency,
                    repaired.tolist(),
                    num_registers=k,
                )


if __name__ == "__main__":
    unittest.main()
