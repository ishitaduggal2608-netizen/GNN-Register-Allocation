"""
Agents Package for Compiler Pattern Detection and Register Allocation.
Exports PatternAgent base class, AgentPrediction container, individual pattern agents,
and the unified agent registry.
"""

from .base import PatternAgent, AgentPrediction, extract_graph_tensors, compute_normalized_entropy_confidence
from .rgcn_agent import RelationalAgent
from .gat_agent import AttentionAgent
from .sage_agent import NeighbourhoodAgent
from .gcn_agent import PressureAgent
from .registry import register_agent, get_agent_class, create_agent, load_agent_checkpoint, list_registered_agents

__all__ = [
    "PatternAgent",
    "AgentPrediction",
    "extract_graph_tensors",
    "compute_normalized_entropy_confidence",
    "RelationalAgent",
    "AttentionAgent",
    "NeighbourhoodAgent",
    "PressureAgent",
    "register_agent",
    "get_agent_class",
    "create_agent",
    "load_agent_checkpoint",
    "list_registered_agents",
]
