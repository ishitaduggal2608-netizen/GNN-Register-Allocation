
import sys
from pathlib import Path

# Add the main project folder to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.registry import load_agent_checkpoint

for k in [1, 2, 3, 4]:
    folder = Path("checkpoints") if k == 4 else Path(f"checkpoints_k{k}")
    print(f"\n--- K = {k} ---")

    for name in ["rgcn", "gat", "sage", "gcn"]:
        path = folder / f"{name}.pt"

        if not path.exists():
            print(f"{name}: FILE MISSING")
            continue

        try:
            agent = load_agent_checkpoint(str(path), device="cpu")
            actual_k = getattr(agent, "num_registers", None)

            if actual_k == k:
                print(f"{name}: OK")
            else:
                print(f"{name}: REGISTER MISMATCH (loaded K={actual_k})")

        except Exception as exc:
            print(f"{name}: ERROR - {exc}")
