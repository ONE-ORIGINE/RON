from __future__ import annotations
import ast, json
from pathlib import Path
from ron.experiments.collision_compare import CollisionExperimentConfig, run_collision_comparison

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"Linear", "Embedding", "GRU", "LSTM", "Transformer", "MultiheadAttention"}
SCAN_DIRS = [ROOT/"ron"/"core", ROOT/"ron"/"runtime", ROOT/"ron"/"memory", ROOT/"ron"/"graph", ROOT/"ron"/"networks", ROOT/"ron"/"experiments"]

def _attr_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))

def forbidden_calls():
    hits = {k: 0 for k in sorted(FORBIDDEN)}
    locations = []
    for root in SCAN_DIRS:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _attr_name(node.func)
                    last = name.split(".")[-1]
                    if last in hits:
                        hits[last] += 1
                        locations.append({"file": str(py.relative_to(ROOT)), "line": getattr(node, "lineno", None), "call": name})
    return hits, locations

def main():
    hits, locations = forbidden_calls()
    summary = run_collision_comparison(CollisionExperimentConfig(steps=3, batch_size=6, out_dir="runs/v20_2_validation_compare"))
    final = summary["final"]
    result = {
        "version": "v20.2",
        "purpose": "comparative training/evaluation on CollisionWorld + high-quality graphics; HTML interfaces removed",
        "interfaces_removed": not (ROOT/"interfaces").exists(),
        "pure_core_no_classical_disguise": all(v == 0 for v in hits.values()),
        "forbidden_module_call_hits": hits,
        "forbidden_locations": locations,
        "comparison_ran": True,
        "variants": sorted(final.keys()),
        "pure_final_loss": final["pure"]["total_loss"],
        "triaxial_final_loss": final["triaxial"]["total_loss"],
        "hybrid_final_loss": final["hybrid"]["total_loss"],
        "pure_final_action_acc": final["pure"]["action_acc"],
        "triaxial_final_action_acc": final["triaxial"]["action_acc"],
        "hybrid_final_action_acc": final["hybrid"]["action_acc"],
        "csv": summary["csv"],
        "summary": summary,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
