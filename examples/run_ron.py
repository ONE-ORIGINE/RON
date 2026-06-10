from __future__ import annotations
import json
from ron import RONSystem, validate_packet

def main():
    ron = RONSystem(
        mode="ron_pure",
        profile="active_geodesic",
        device="cpu",
        lite_cells=8,
        memory_slots=16,
        grid=1,
        cube=3,
        moment_degree=1,
        signature_degree=1,
        harmonic_degree=1,
        internal_cycles=1,
    )
    out = ron.infer_triplet(2, 5, 7, mask=3)
    validation = validate_packet(out.packet)
    print(json.dumps({
        "packet_valid": validation.ok,
        "packet": out.packet.as_dict(),
        "summary": out.summary,
        "trace": [t.__dict__ for t in out.trace],
        "policy": out.policy,
    }, indent=2))

if __name__ == "__main__":
    main()
