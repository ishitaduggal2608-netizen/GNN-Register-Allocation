"""
End-to-End Integration Tests for Synthetic Generation, Dataset, Model Training, and Evaluation.
"""

import unittest
from dataset.generator import SyntheticIRGenerator
from dataset.dataset import InterferenceGraphDataset, split_raw_samples
from models.gnn_allocator import RelationalGNNRegisterAllocator
from models.trainer import GNNTrainer
from evaluation.evaluator import BenchmarkEvaluator


class TestEndToEndPipeline(unittest.TestCase):
    def test_train_val_test_split_has_no_overlap(self):
        generator = SyntheticIRGenerator(seed=321)
        raw_samples = generator.generate_dataset(num_samples=20, min_vars=5, max_vars=8, num_registers=4)

        raw_train, raw_val, raw_test = split_raw_samples(raw_samples, seed=321)
        train_ids = {id(sample[0]) for sample in raw_train}
        val_ids = {id(sample[0]) for sample in raw_val}
        test_ids = {id(sample[0]) for sample in raw_test}

        self.assertEqual(len(raw_train) + len(raw_val) + len(raw_test), len(raw_samples))
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_full_pipeline(self):
        # 1. Dataset generation
        generator = SyntheticIRGenerator(seed=123)
        raw_samples = generator.generate_dataset(num_samples=5, min_vars=5, max_vars=10, num_registers=4)
        self.assertEqual(len(raw_samples), 5)
        raw_train, raw_val, _ = split_raw_samples(raw_samples, seed=123)

        # 2. PyTorch Dataset
        dataset = InterferenceGraphDataset(raw_train, num_registers=4)
        val_dataset = InterferenceGraphDataset(
            raw_val, num_registers=4, normalization_stats=dataset.normalization_stats
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(val_dataset), 1)

        # 3. Model & Trainer
        model = RelationalGNNRegisterAllocator(in_node_features=6, hidden_dim=16, num_registers=4)
        trainer = GNNTrainer(model, dataset, val_dataset=val_dataset, lr=1e-2)

        history = trainer.train(num_epochs=2, verbose=False)
        self.assertEqual(len(history["train_loss"]), 2)

        # 4. Benchmark evaluation
        evaluator = BenchmarkEvaluator(num_registers=4)
        results = evaluator.benchmark_batch(model, raw_samples)
        self.assertIn("GNN", results)
        self.assertIn("Chaitin-Briggs", results)


if __name__ == "__main__":
    unittest.main()
