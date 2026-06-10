from __future__ import annotations
from ron.runtime.ron_system import RONSystem

def test_clean_active_geodesic_runtime():
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
    d = out.packet.as_dict()
    assert isinstance(d["action_face"], str)
    assert out.summary["active_pressure_strength"] > 0
    assert out.summary["active_changed_packet"] in (True, False)
    assert out.policy["allow_classical_core"] is False
