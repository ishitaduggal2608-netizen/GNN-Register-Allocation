"""Local web app server: serves the dashboard and exposes POST /api/allocate."""
import json
import os
import random
import sys
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import torch

from tac_parser import parse_tac, TACParseError
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import ChaitinBriggsAllocator
from agents.registry import create_agent, load_agent_checkpoint
from agents.consensus import ConsensusArbiter
from models.repair import repair_conflicts

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
AGENT_NAMES = ["rgcn", "gat", "sage", "gcn"]



def _load_agents(num_registers):
    agents, statuses = {}, {}

    # Select the checkpoint folder for the requested register count
    if num_registers == 4:
        checkpoint_dir = ROOT / "checkpoints"
    else:
        checkpoint_dir = ROOT / f"checkpoints_k{num_registers}"

    for name in AGENT_NAMES:
        candidates = [checkpoint_dir / f"{name}.pt"]

        # Legacy fallback for RGCN at K=4 only
        if num_registers == 4 and name == "rgcn":
            candidates.append(ROOT / "gnn_allocator.pt")

        loaded = False

        for checkpoint in candidates:
            if not checkpoint.exists():
                continue

            try:
                agent = load_agent_checkpoint(
                    str(checkpoint), device="cpu"
                )

                # Reject checkpoints trained for a different K
                if getattr(agent, "num_registers", None) != num_registers:
                    statuses[name] = (
                        f"incompatible register count: {checkpoint.name}"
                    )
                    continue

                agents[name] = agent
                statuses[name] = f"trained checkpoint: {checkpoint.name}"
                loaded = True
                break

            except Exception as exc:
                statuses[name] = (
                    f"checkpoint unavailable ({checkpoint.name}): {exc}"
                )

        if not loaded:
            agents[name] = create_agent(
                name, num_registers=num_registers
            )
            statuses.setdefault(
                name,
                "no compatible trained checkpoint; untrained model initialized"
            )

    return agents, statuses

def build_result(source, num_registers):
    program = parse_tac(source)
    cfg = ControlFlowGraph(program)
    liveness = LivenessAnalyzer(cfg)
    ig = InterferenceGraph(program, cfg, liveness)

    if not ig.variables:
        raise TACParseError("This program has no variables to allocate. Add an assignment such as 'x = 1'.")

    agents, model_status = _load_agents(num_registers)
    predictions = {name: agent.predict_from_ig(ig) for name, agent in agents.items()}
    consensus = ConsensusArbiter(num_registers=num_registers).arbitrate(predictions, ig)
    cb_result = ChaitinBriggsAllocator(num_registers=num_registers).allocate(ig)

    n = len(ig.variables)
    var_to_idx = {v.name: i for i, v in enumerate(ig.variables)}
    adj = torch.zeros((n, n), dtype=torch.float32)
    for u, v in ig.interference_edges:
        if u in var_to_idx and v in var_to_idx:
            i, j = var_to_idx[u], var_to_idx[v]
            adj[i, j] = adj[j, i] = 1.0

    assignments = {}
    for name, pred in predictions.items():
        repaired, _ = repair_conflicts(adj, pred.pred_classes, num_registers=num_registers, spill_class=num_registers)
        assignments[name] = {
            v.name: (f"R{int(repaired[var_to_idx[v.name]].item())}" if int(repaired[var_to_idx[v.name]].item()) < num_registers else "SPILL")
            for v in ig.variables
        }
    assignments["consensus"] = dict(consensus.register_assignment)
    assignments["cb"] = {v.name: cb_result.register_assignment.get(v.name, "SPILL") for v in ig.variables}

    rng = random.Random(42)
    random_labels = torch.tensor([rng.randrange(num_registers + 1) for _ in range(n)], dtype=torch.long)
    random_repaired, _ = repair_conflicts(adj, random_labels, num_registers=num_registers, spill_class=num_registers)
    assignments["random"] = {v.name: (f"R{int(random_repaired[var_to_idx[v.name]].item())}" if int(random_repaired[var_to_idx[v.name]].item()) < num_registers else "SPILL") for v in ig.variables}

    model_stats = {}
    for model_name, assignment in assignments.items():
        spilled = [v.name for v in ig.variables if assignment.get(v.name) == "SPILL"]
        cost = sum(ig.spill_costs.get(name, 1.0) for name in spilled)
        moves_eliminated = 0
        for inst in program.instructions:
            if inst.op.value == "MOVE" and inst.target and inst.arg1:
                left = assignment.get(inst.target.name)
                right = assignment.get(inst.arg1.name)
                if left == right and left != "SPILL":
                    moves_eliminated += 1
        model_stats[model_name] = {
            "spills": len(spilled),
            "spill_cost": round(float(cost), 2),
            "moves_eliminated": moves_eliminated,
        }

    features = ig.get_node_features()
    nodes = []
    for variable in ig.variables:
        name, idx = variable.name, var_to_idx[variable.name]
        votes = {m: assignments[m][name] for m in AGENT_NAMES}
        is_disagreement = len(set(votes.values())) > 1
        spill_probs, confidence = {}, {}
        for m, pred in predictions.items():
            spill_probs[m] = round(float(pred.spill_prob[idx].item()), 4)
            confidence[m] = round(float(pred.confidence[idx].item()), 4)
        spill_probs["consensus"] = round(float(consensus.consensus_probs[idx, num_registers].item()), 4)
        confidence["consensus"] = round(float(torch.max(consensus.consensus_probs[idx]).item()), 4)
        assignments_for_node = {m: assignment[name] for m, assignment in assignments.items()}
        feat = features[name]
        nodes.append({
            "id": name, "spill_cost": feat.spill_cost, "loop_depth": feat.loop_depth,
            "degree": feat.degree, "move_degree": feat.move_degree,
            "live_range_length": feat.live_range_length, "use_count": feat.use_count,
            "assignments": assignments_for_node,
            "spill_probs": spill_probs, "confidence": confidence,
            "is_disagreement": is_disagreement,
            "gnn_assignment": assignments["consensus"][name],
            "cb_assignment": assignments["cb"][name],
        })

    cfg_blocks = [{
        "block_id": block.block_id, "label": block.label,
        "loop_depth": block.loop_depth,
        "instructions": [str(inst) for inst in block.instructions],
        "successors": [succ.block_id for succ in block.successors],
    } for block in cfg.blocks]

    # Recompute simple pairwise agreement values for the interactive result.
    pairwise = {}
    for i, first in enumerate(AGENT_NAMES):
        for second in AGENT_NAMES[i + 1:]:
            pct = 100.0 * sum(assignments[first][v.name] == assignments[second][v.name] for v in ig.variables) / max(n, 1)
            pairwise[f"{first}-{second}"] = round(pct, 1)
    unanimous = 100.0 * sum(len({assignments[m][v.name] for m in AGENT_NAMES}) == 1 for v in ig.variables) / max(n, 1)
    disagreements = [{"id": v.name, "assignments": {m: assignments[m][v.name] for m in AGENT_NAMES}} for v in ig.variables if len({assignments[m][v.name] for m in AGENT_NAMES}) > 1]

    return {
        "program_name": program.name,
        "num_registers": num_registers,
        "instructions": [str(inst) for inst in program.instructions],
        "cfg_blocks": cfg_blocks,
        "nodes": nodes,
        "interference_edges": [{"source": u, "target": v, "type": "interference"} for u, v in sorted(ig.interference_edges)],
        "coalescing_edges": [{"source": u, "target": v, "type": "coalescing"} for u, v in sorted(ig.coalescing_edges)],
        "attention_edges": [],
        "models": ["consensus", "rgcn", "gat", "sage", "gcn", "cb", "random"],
        "model_keys": ["consensus", "rgcn", "gat", "sage", "gcn", "cb", "random"],
        "model_metadata": {
            "consensus": {"name": "Consensus Ensemble", "arch": "Ensemble", "specialty": "Confidence-weighted voting across the available agents."},
            "rgcn": {"name": "Relational GCN (R-GCN)", "arch": "R-GCN", "specialty": "Relational graph message passing."},
            "gat": {"name": "Attention Agent (GAT)", "arch": "GAT", "specialty": "Attention-based neighbourhood aggregation."},
            "sage": {"name": "GraphSAGE Agent", "arch": "GraphSAGE", "specialty": "Neighbourhood feature aggregation."},
            "gcn": {"name": "GCN Agent", "arch": "GCN", "specialty": "Graph convolution baseline."},
            "cb": {"name": "Chaitin-Briggs", "arch": "Classical", "specialty": "Classical graph-colouring allocator."},
            "random": {"name": "Random Allocator", "arch": "Random", "specialty": "Random assignment baseline with conflict repair."},
        },
        "model_stats": model_stats,
        "disagreement_nodes": disagreements,
        "pairwise_agreement": pairwise,
        "unanimous_pct": round(unanimous, 1),
        "explanations": {name: pred.explanation for name, pred in predictions.items()},
        "cb_coalesced_pairs": [list(pair) for pair in cb_result.coalesced_pairs],
        "cb_spills": sorted(list(cb_result.spilled_vars)),
        "gnn_spills": [name for name, assignment in assignments["consensus"].items() if assignment == "SPILL"],
        "model_status": model_status,
        "source_mode": "live_tac_input",
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path.rstrip("/") != "/api/allocate":
            self.send_error(404, "Unknown API endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 200_000:
                raise ValueError("Request body is empty or too large (maximum 200 KB).")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            source = payload.get("tac", "")
            registers = int(payload.get("registers", 4))
            if registers < 1 or registers > 16:
                raise ValueError("Register count must be between 1 and 16.")
            result = build_result(source, registers)
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (TACParseError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.log_error("Allocation failed: %s", exc)
            self._send_json(500, {"error": f"Allocation failed: {exc}"})

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(os.environ.get("REGALLOC_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("\nGNN Register Allocation live web app")
    print(f"Open http://127.0.0.1:{port} in your browser")
    print("Paste TAC into the Live Allocation panel and click Run Allocation.")
    print("Press Ctrl+C to stop the server.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
