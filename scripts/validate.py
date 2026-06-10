from __future__ import annotations
import ast, json, subprocess, sys
from pathlib import Path
from ron import RONSystem, validate_packet, packet_schema
from ron.hybrid import HybridRouter, FamilySignal, NeuralFamily

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"Linear", "Embedding", "GRU", "LSTM", "Transformer", "MultiheadAttention"}

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
    for py in ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _attr_name(node.func)
                last = name.split(".")[-1]
                if last in hits:
                    hits[last] += 1
                    locations.append({"file": str(py.relative_to(ROOT)), "line": getattr(node, "lineno", None), "call": name})
    return hits, locations

def main():
    hits, loc = forbidden_calls()
    ron = RONSystem(
        mode="ron_pure",
        profile="active_geodesic",
        device="cpu",
        lite_cells=4,
        memory_slots=8,
        grid=1,
        cube=3,
        moment_degree=1,
        signature_degree=1,
        harmonic_degree=1,
        internal_cycles=1,
    )
    out = ron.infer_triplet(2, 5, 7, mask=3)
    packet_validation = validate_packet(out.packet)
    router = HybridRouter()
    route = router.route([
        FamilySignal(NeuralFamily.RON, "conductor", {}, may_decide=True),
        FamilySignal(NeuralFamily.TRIAXIAL_BRUTE, "teacher", {}, may_decide=False),
        FamilySignal(NeuralFamily.CLASSICAL, "codec", {}, may_decide=False),
        FamilySignal(NeuralFamily.CLASSICAL, "hidden_decider", {}, may_decide=True),
    ])
    result = {
        "version": "v20",
        "purpose": "final clean RON repository",
        "no_classical_disguise": all(v == 0 for v in hits.values()),
        "forbidden_module_call_hits": hits,
        "forbidden_locations": loc,
        "runtime_ok": True,
        "default_profile": out.summary.get("profile"),
        "default_policy": out.policy,
        "packet_contract_ok": packet_validation.ok,
        "packet_contract_errors": packet_validation.errors,
        "packet_schema_field_count": len(packet_schema()),
        "hybrid_folder_present": (ROOT / "ron" / "hybrid").exists(),
        "hybrid_rejects_hidden_classical_decider": len(route.rejected) == 1 and route.rejected[0].role == "hidden_decider",
        "active_pressure_strength_positive": float(out.summary.get("active_pressure_strength", 0.0)) > 0,
        "active_memory_writes": out.summary.get("memory_summary", {}).get("used_slots", 0) > 0,
        "packet": out.packet.as_dict(),
        "summary": out.summary,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
