"""
Feature Importance Ablation Study Module for GNN Register Allocator.
Trains best architecture with feature subsets masked out to quantify individual feature importance.
"""

from typing import Dict, List, Any
import numpy as np
import torch
from models.gnn_allocator import RelationalGNNRegisterAllocator
from models.trainer import GNNTrainer
from dataset.dataset import InterferenceGraphDataset, GraphDataSample


FEATURE_NAMES = [
    "spill_cost",
    "loop_depth",
    "degree",
    "move_degree",
    "live_range_length",
    "use_count"
]


class FeatureAblationStudy:
    """Executes ablation experiments by zeroing out specific node features."""
    def __init__(self, num_registers: int = 4):
        self.num_registers: int = num_registers

    def mask_dataset_feature(
        self, dataset: InterferenceGraphDataset, mask_feature_idx: int
    ) -> InterferenceGraphDataset:
        """Returns a copy of dataset with specified feature column zeroed out."""
        masked_samples = []
        for sample in dataset.samples:
            x_masked = sample.node_features.clone()
            x_masked[:, mask_feature_idx] = 0.0
            masked_sample = GraphDataSample(
                node_features=x_masked,
                interf_adj=sample.interf_adj,
                coal_adj=sample.coal_adj,
                target_colors=sample.target_colors,
                coalesce_labels=sample.coalesce_labels,
                var_names=sample.var_names,
                num_registers=sample.num_registers
            )
            masked_samples.append(masked_sample)

        new_ds = InterferenceGraphDataset([], num_registers=self.num_registers)
        new_ds.samples = masked_samples
        new_ds.normalization_stats = getattr(dataset, "normalization_stats", None)
        return new_ds

    def run_ablation_study(
        self,
        dataset: InterferenceGraphDataset,
        val_dataset: InterferenceGraphDataset,
        epochs: int = 15,
        seed: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs ablation study removing one feature at a time and training R-GCN.
        Returns metrics summary dictionary.
        """
        results = {}

        # 1. Full feature set baseline
        torch.manual_seed(seed)
        full_model = RelationalGNNRegisterAllocator(num_registers=self.num_registers)
        trainer = GNNTrainer(full_model, dataset, val_dataset=val_dataset)
        trainer.train(num_epochs=epochs, verbose=False)
        full_eval = trainer.evaluate()
        results["Full Features"] = full_eval

        # 2. Single feature ablation experiments
        for idx, feat_name in enumerate(FEATURE_NAMES):
            torch.manual_seed(seed)
            ablated_ds = self.mask_dataset_feature(dataset, idx)
            ablated_val_ds = self.mask_dataset_feature(val_dataset, idx)
            model = RelationalGNNRegisterAllocator(num_registers=self.num_registers)
            ab_trainer = GNNTrainer(model, ablated_ds, val_dataset=ablated_val_ds)
            ab_trainer.train(num_epochs=epochs, verbose=False)
            eval_metrics = ab_trainer.evaluate()
            results[f"No {feat_name}"] = eval_metrics

        return results
