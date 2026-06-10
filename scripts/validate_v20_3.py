from __future__ import annotations
import ast, json
from pathlib import Path
from ron.experiments.scientific_protocol import ScientificProtocolConfig, run_scientific_protocol
from ron.experiments.packet_analysis import PacketAnalysisConfig, run_packet_analysis

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
    protocol = run_scientific_protocol(ScientificProtocolConfig(
        steps=1,
        batch_size=3,
        seeds=1,
        base_seed=3030,
        out_dir="runs/v20_3_validation_protocol",
        device="cpu",
    ))
    packet = run_packet_analysis(PacketAnalysisConfig(samples=6, out_dir="runs/v20_3_validation_packet"))
    variants = sorted({r["variant"] for r in protocol["final_rows"]})
    result = {
        "version": "v20.3",
        "purpose": "scientific protocol layer: replicated comparison, statistics, packet analysis, publication graphics",
        "pure_core_no_classical_disguise": all(v == 0 for v in hits.values()),
        "forbidden_module_call_hits": hits,
        "forbidden_locations": locations,
        "protocol_ran": True,
        "seeds": protocol["config"]["seeds"],
        "steps": protocol["config"]["steps"],
        "variants": variants,
        "has_metric_summary": bool(protocol["metric_summary"]),
        "has_comparisons": bool(protocol["comparisons"]),
        "packet_analysis_ran": "confusion_proxy" in packet,
        "evidence_level": protocol["evidence_grade"]["level"],
        "six_hour_guidance": protocol["evidence_grade"]["six_hour_guidance"],
        "metric_summary": protocol["metric_summary"],
        "comparisons": protocol["comparisons"],
        "packet_analysis": packet,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
