from __future__ import annotations
import ast, json
from pathlib import Path
from ron.datasets.synthetic_dynamics import make_dynamics_batch
from ron.datasets.collision_world import CollisionWorldConfig, make_collision_batch
from ron.networks import RONPureDynamicsNet, RONTriaxialTeacherNet, RONHybridTriFamilyNet
from ron.hybrid.triaxial_brute import OptionalTriaxialBrute

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"Linear", "Embedding", "GRU", "LSTM", "Transformer", "MultiheadAttention"}
PURE_SCAN_DIRS = [
    ROOT / "ron" / "core",
    ROOT / "ron" / "runtime",
    ROOT / "ron" / "memory",
    ROOT / "ron" / "graph",
    ROOT / "ron" / "bridges",
    ROOT / "ron" / "networks",
]

def _attr_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if isinstance(node, ast.Name): parts.append(node.id)
    return ".".join(reversed(parts))

def forbidden_calls(paths):
    hits = {k: 0 for k in sorted(FORBIDDEN)}
    locations = []
    for root in paths:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
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
    hits, loc = forbidden_calls(PURE_SCAN_DIRS)
    dyn = make_dynamics_batch(batch_size=4, seed=1, device="cpu")
    col = make_collision_batch(CollisionWorldConfig(batch_size=4, seed=2), device="cpu")
    kwargs = dict(lite_cells=4, memory_slots=8, grid=1, cube=3, moment_degree=1, signature_degree=1, harmonic_degree=1, internal_cycles=1, device="cpu")

    pure = RONPureDynamicsNet(**kwargs).infer_axes(dyn.axes, dyn.energy, dyn.mask)
    tri = RONTriaxialTeacherNet(**kwargs).infer_axes(dyn.axes, dyn.energy, dyn.mask)
    hyb = RONHybridTriFamilyNet(**kwargs).infer_states(col.x_state, col.y_state, col.z_state, col.mask)
    brute_path = ROOT.parent / "triaxial_neuron.py"
    brute = OptionalTriaxialBrute(str(brute_path) if brute_path.exists() else None)
    if brute.available():
        try:
            brute.load()
            brute_summary = brute.summary()
        except Exception as e:
            brute_summary = {"loaded": False, "error": str(e), "source_path": str(brute_path)}
    else:
        brute_summary = brute.summary()

    result = {
        "version": "v20.1",
        "purpose": "datasets + concrete networks + optional triaxial brute wrapper + first interfaces/visuals",
        "pure_core_no_classical_disguise": all(v == 0 for v in hits.values()),
        "pure_core_forbidden_call_hits": hits,
        "pure_core_forbidden_locations": loc,
        "synthetic_dynamics_ok": dyn.axes.shape == (4,3,3) and dyn.energy.shape == (4,3),
        "collision_world_ok": col.axes.shape == (4,3,3) and col.energy.shape == (4,3),
        "ron_pure_network_ok": pure.packet is not None and pure.summary.get("mode") == "ron_pure",
        "ron_triaxial_network_ok": tri.teacher.get("recommended_mode") is not None and tri.summary.get("triaxial_teacher_mode_count") == 6,
        "ron_hybrid_trifamily_ok": hyb.summary.get("hybrid_accepted_count") == 3 and hyb.summary.get("hybrid_rejected_count") == 0,
        "hybrid_route": hyb.hybrid_route,
        "optional_triaxial_brute": brute_summary,
        "pure_summary": pure.summary,
        "triaxial_summary": tri.summary,
        "hybrid_summary": hyb.summary,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
