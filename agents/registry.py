"""
Agent Architecture Registry & Checkpoint Loader.
Supports unified creation, lookup, and self-describing checkpoint loading across
all 4 Pattern Detection Agents, with full backward-compatibility for legacy R-GCN checkpoints.
"""

from typing import Dict, Type, Any, Optional
import os
import pickle
import warnings
import torch

from .base import PatternAgent
from .rgcn_agent import RelationalAgent
from .gat_agent import AttentionAgent
from .sage_agent import NeighbourhoodAgent
from .gcn_agent import PressureAgent

# Architecture Registry mapping string aliases to Agent classes
_REGISTRY: Dict[str, Type[PatternAgent]] = {
    "rgcn": RelationalAgent,
    "r-gcn": RelationalAgent,
    "relational": RelationalAgent,
    "gat": AttentionAgent,
    "rgat": AttentionAgent,
    "r-gat": AttentionAgent,
    "attention": AttentionAgent,
    "sage": NeighbourhoodAgent,
    "rsage": NeighbourhoodAgent,
    "r-sage": NeighbourhoodAgent,
    "graphsage": NeighbourhoodAgent,
    "neighbourhood": NeighbourhoodAgent,
    "neighborhood": NeighbourhoodAgent,
    "structure": NeighbourhoodAgent,
    "gcn": PressureAgent,
    "gin": PressureAgent,
    "rgin": PressureAgent,
    "pressure": PressureAgent,
}


def register_agent(alias: str, agent_class: Type[PatternAgent]) -> None:
    """Registers an agent class under an architectural alias."""
    _REGISTRY[alias.lower()] = agent_class


def get_agent_class(arch_or_name: str) -> Type[PatternAgent]:
    """Resolves an architectural name or alias to its corresponding PatternAgent class."""
    key = arch_or_name.lower().strip()
    if key not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Unknown agent architecture '{arch_or_name}'. Available: {available}")
    return _REGISTRY[key]


def list_registered_agents() -> Dict[str, str]:
    """Returns dictionary of registered agent names and class names."""
    return {k: v.__name__ for k, v in _REGISTRY.items()}


def create_agent(
    architecture: str,
    num_registers: int = 4,
    in_channels: int = 6,
    hidden_dim: int = 64,
    **kwargs
) -> PatternAgent:
    """Factory helper to instantiate any registered PatternAgent."""
    agent_cls = get_agent_class(architecture)
    return agent_cls(
        num_registers=num_registers,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        **kwargs
    )


def load_agent_checkpoint(filepath: str, device: str = "cpu") -> PatternAgent:
    """
    Loads an agent from a saved checkpoint file.
    Self-describing checkpoints specify their architecture and hyperparameters.
    Legacy checkpoints without an 'architecture' field (e.g. gnn_allocator.pt)
    are seamlessly loaded as RelationalAgent (R-GCN).
    """
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except Exception as e:
        checkpoint = torch.load(filepath, map_location=device)

    # Legacy compatibility: if checkpoint is a pure state_dict or lacks architecture field, default to rgcn
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {filepath} is not a valid dictionary.")

    arch = checkpoint.get("architecture", "rgcn")
    agent_cls = get_agent_class(arch)

    num_registers = checkpoint.get("num_registers", 4)
    in_channels = checkpoint.get("in_channels", checkpoint.get("in_node_features", 6))
    hidden_dim = checkpoint.get("hidden_dim", 64)

    agent = agent_cls(
        num_registers=num_registers,
        in_channels=in_channels,
        hidden_dim=hidden_dim
    )

    if "normalization_stats" in checkpoint:
        agent.normalization_stats = checkpoint["normalization_stats"]

    # Retrieve weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "agent_state_dict" in checkpoint:
        state_dict = checkpoint["agent_state_dict"]
    else:
        state_dict = checkpoint

    if agent.model is not None:
        agent.model.load_state_dict(state_dict)
    else:
        agent.load_state_dict(state_dict)

    agent.to(device)
    agent.eval()
    return agent
