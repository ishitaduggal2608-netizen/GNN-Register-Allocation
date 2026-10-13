"""Phase 8: train a move-edge coalescing head over frozen spill-GNN embeddings."""

from __future__ import annotations

import torch

try:
    from .models import CoalesceClassifier
except ImportError:
    from models import CoalesceClassifier


def train_coalesce_head(encoder, train_loader, validation_loader, embedding_dim: int,
                        epochs: int = 30, learning_rate: float = 1e-3, device=None):
    """Freezes ``encoder`` and optimizes only the pairwise MLP head."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    head = CoalesceClassifier(embedding_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)

    for _ in range(epochs):
        head.train()
        for batch in train_loader:
            if batch.y_coalesce.numel() == 0:
                continue
            batch = batch.to(device)
            with torch.no_grad():
                embeddings = encoder.encode(batch.x, batch.edge_index)
            logits = head(embeddings, batch.move_edge_index)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.y_coalesce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head, evaluate_coalesce_head(encoder, head, validation_loader, device)


@torch.no_grad()
def evaluate_coalesce_head(encoder, head, loader, device) -> dict[str, float]:
    head.eval()
    correct = total = 0
    for batch in loader:
        if batch.y_coalesce.numel() == 0:
            continue
        batch = batch.to(device)
        predictions = (torch.sigmoid(head(encoder.encode(batch.x, batch.edge_index), batch.move_edge_index)) > 0.5)
        correct += int((predictions == batch.y_coalesce.bool()).sum())
        total += batch.y_coalesce.numel()
    return {"accuracy": 0.0 if total == 0 else correct / total, "num_move_edges": total}
