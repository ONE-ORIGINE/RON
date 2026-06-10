from __future__ import annotations
from ron.runtime.ron_system import RONSystem
from ron.runtime.packet_contract import validate_packet, packet_schema

def test_packet_contract_runtime():
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
    val = validate_packet(out.packet)
    assert val.ok, val.errors
    assert "action_face" in packet_schema()
