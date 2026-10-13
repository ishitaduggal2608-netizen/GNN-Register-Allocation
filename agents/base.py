"""
Base Pattern Detection Agent Architecture for Compiler Register Allocation.
Defines PatternAgent abstract base class, AgentPrediction container,
and shared compiler interference graph tensor extraction routines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from compiler.interference_graph import InterferenceGraph
from dataset.dataset import InterferenceGraphDataset, GraphDataSample, compute_feature_normalization_stats


@dataclass
class AgentPrediction:
    """Standardized prediction dataclass returned by all Pattern Detection Agents."""
    color_probs: torch.Tensor               # [N, K+1] class probabilities
    spill_prob: torch.Tensor                # [N] spill probability (class K)
    coalesce_score: Optional[torch.Tensor]  # [N, N] move coalescing likelihood matrix
    confidence: torch.Tensor                # [N] normalized certainty score in [0.0, 1.0]
    pred_classes: torch.Tensor              # [N] argmax predicted register or spill index
    priority: Optional[torch.Tensor] = None # [N] optional simplification/difficulty priority score
    explanation: Optional[Dict[str, Any]] = None  # Human-interpretable detected pattern payload


def extract_graph_tensors(
    ig: InterferenceGraph,
    normalization_stats: Optional[Dict[str, List[float]]] = None,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Extracts node features, interference adjacency, and coalescing adjacency tensors
    from an InterferenceGraph compiler structure.

    Shared helper factored out from BenchmarkEvaluator to eliminate duplicate tensor-building logic.

    Returns:
        x: [N, 6] float32 node feature tensor
        interf_adj: [N, N] float32 symmetric interference adjacency tensor
        coal_adj: [N, N] float32 symmetric move coalescing adjacency tensor
        var_names: List of variable names corresponding to indices 0..N-1
    """
    feat_matrix = ig.get_feature_matrix(normalization_stats=normalization_stats)
    x = torch.tensor(feat_matrix, dtype=torch.float32, device=device)

    N = len(ig.variables)
    var_names = [v.name for v in ig.variables]
    var_to_idx = {v.name: i for i, v in enumerate(ig.variables)}

    interf_adj = torch.zeros((N, N), dtype=torch.float32, device=device)
    for u, v in ig.interference_edges:
        if u in var_to_idx and v in var_to_idx:
            i, j = var_to_idx[u], var_to_idx[v]
            interf_adj[i, j] = 1.0
            interf_adj[j, i] = 1.0

    coal_adj = torch.zeros((N, N), dtype=torch.float32, device=device)
    for u, v in ig.coalescing_edges:
        if u in var_to_idx and v in var_to_idx:
            i, j = var_to_idx[u], var_to_idx[v]
            coal_adj[i, j] = 1.0
            coal_adj[j, i] = 1.0

    return x, interf_adj, coal_adj, var_names


def compute_normalized_entropy_confidence(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Calculates per-node prediction confidence as (1.0 - normalized_entropy).
    Uniform distribution over C classes yields confidence 0.0.
    One-hot deterministic distribution yields confidence 1.0.

    Args:
        probs: [N, C] tensor of probabilities summing to 1 along dim -1.
        eps: Small epsilon to avoid log(0).
    """
    C = probs.size(-1)
    if C <= 1:
        return torch.ones(probs.size(0), dtype=torch.float32, device=probs.device)
    
    entropy = -torch.sum(probs * torch.log(probs.clamp(min=eps)), dim=-1)  # [N]
    max_entropy = math.log(float(C))
    norm_entropy = (entropy / max_entropy).clamp(min=0.0, max=1.0)
    return (1.0 - norm_entropy).clamp(min=0.0, max=1.0)


class PatternAgent(nn.Module, ABC):
    """
    Abstract base class for all Pattern Detection Agents in the compiler allocator.
    Each agent detects a unique structural or program pattern and produces standardized
    color logits, spill probabilities, and explanations.
    """
    OWNER: str = "UNASSIGNED"

    def __init__(
        self,
        name: str,
        architecture: str,
        specialty: str,
        num_registers: int = 4,
        in_channels: int = 6,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.name: str = name
        self.architecture: str = architecture
        self.specialty: str = specialty
        self.num_registers: int = num_registers
        self.in_channels: int = in_channels
        self.hidden_dim: int = hidden_dim
        self.num_classes: int = num_registers + 1  # K registers + 1 spill class
        self.normalization_stats: Optional[Dict[str, List[float]]] = None
        self.model: Optional[nn.Module] = None

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Base forward pass.
        Returns:
            color_logits: [N, K+1]
            coalesce_logits: Optional [N, N]
        """
        raise NotImplementedError

    def predict(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None,
        ig: Optional[InterferenceGraph] = None
    ) -> AgentPrediction:
        """
        Runs inference and packages the output into a standardized AgentPrediction container.
        """
        self.eval()
        with torch.no_grad():
            color_logits, coalesce_logits = self.forward(x, interf_adj, coal_adj)
            color_probs = F.softmax(color_logits, dim=-1)
            spill_prob = color_probs[:, self.num_registers]
            pred_classes = torch.argmax(color_logits, dim=-1)
            confidence = compute_normalized_entropy_confidence(color_probs)

            coalesce_score = None
            if coalesce_logits is not None:
                coalesce_score = torch.sigmoid(coalesce_logits)

            explanation = self.explain(x, interf_adj, coal_adj, ig=ig)

        return AgentPrediction(
            color_probs=color_probs,
            spill_prob=spill_prob,
            coalesce_score=coalesce_score,
            confidence=confidence,
            pred_classes=pred_classes,
            explanation=explanation
        )

    def predict_from_ig(self, ig: InterferenceGraph, device: str = "cpu") -> AgentPrediction:
        """Convenience method to execute agent inference directly on a compiler InterferenceGraph."""
        x, interf_adj, coal_adj, _ = extract_graph_tensors(
            ig, normalization_stats=self.normalization_stats, device=device
        )
        return self.predict(x, interf_adj, coal_adj, ig=ig)

    @abstractmethod
    def explain(
        self,
        x: torch.Tensor,
        interf_adj: torch.Tensor,
        coal_adj: Optional[torch.Tensor] = None,
        ig: Optional[InterferenceGraph] = None
    ) -> Dict[str, Any]:
        """Produces human-readable diagnostic analysis of patterns detected in the graph."""
        raise NotImplementedError

    def fit(
        self,
        train_ds: InterferenceGraphDataset,
        val_ds: Optional[InterferenceGraphDataset] = None,
        epochs: int = 20,
        seed: int = 42,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = False,
        device: str = "cpu"
    ) -> Dict[str, List[float]]:
        """
        Trains the agent's internal model using identical data splits and seeds.
        """
        # Reproducibility seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.normalization_stats = getattr(train_ds, "normalization_stats", None)
        from models.trainer import GNNTrainer
        trainer = GNNTrainer(
            model=self.model if self.model is not None else self,
            dataset=train_ds,
            val_dataset=val_ds,
            lr=lr,
            weight_decay=weight_decay,
            device=device
        )
        history = trainer.train(num_epochs=epochs, verbose=verbose)
        return history

    def describe(self) -> Dict[str, Any]:
        """Returns structured metadata for the agent."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "specialty": self.specialty,
            "owner": self.OWNER,
            "num_registers": self.num_registers,
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim
        }

    def save(self, filepath: str) -> None:
        """
        Saves a self-describing checkpoint containing architecture metadata,
        hyperparameters, normalization stats, and state dict.
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        payload = {
            "architecture": self.architecture,
            "agent_class": self.__class__.__name__,
            "name": self.name,
            "specialty": self.specialty,
            "owner": self.OWNER,
            "num_registers": self.num_registers,
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "normalization_stats": self.normalization_stats,
            "model_state_dict": self.model.state_dict() if self.model is not None else self.state_dict()
        }
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "PatternAgent":
        """Loads agent from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        agent = cls(
            num_registers=checkpoint.get("num_registers", 4),
            in_channels=checkpoint.get("in_channels", 6),
            hidden_dim=checkpoint.get("hidden_dim", 64)
        )
        if "normalization_stats" in checkpoint:
            agent.normalization_stats = checkpoint["normalization_stats"]

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if agent.model is not None:
            agent.model.load_state_dict(state_dict)
        else:
            agent.load_state_dict(state_dict)
        agent.eval()
        return agent
