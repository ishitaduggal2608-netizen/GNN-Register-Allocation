
import unittest
import torch

from models.repair import repair_conflicts


class TestLiveAllocationCorrectness(unittest.TestCase):

    def test_interfering_variables_do_not_share_register(self):
        # a, b, and c all interfere with one another.
        adjacency = torch.tensor([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=torch.float32)

        # Initial predictions: all variables want register R0.
        predicted_classes = torch.tensor([0, 0, 0])

        repaired, _ = repair_conflicts(
            adjacency,
            predicted_classes,
            num_registers=1,
            spill_class=1,
        )

        # Every assigned register must be valid.
        self.assertTrue(torch.all(repaired >= 0).item())
        self.assertTrue(torch.all(repaired <= 1).item())

        # With only one physical register, at most one of these
        # mutually interfering variables can remain in R0.
        register_assignments = repaired.tolist()
        self.assertLessEqual(register_assignments.count(0), 1)

        # Every interfering pair must have different physical
        # registers, or at least one endpoint must be spilled.
        for i in range(3):
            for j in range(i + 1, 3):
                if adjacency[i, j] == 1:
                    self.assertTrue(
                        repaired[i].item() == 1
                        or repaired[j].item() == 1
                        or repaired[i].item() != repaired[j].item()
                    )


if __name__ == "__main__":
    unittest.main()
