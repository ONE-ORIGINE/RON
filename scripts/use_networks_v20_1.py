from __future__ import annotations
import argparse, json
from ron.datasets.synthetic_dynamics import make_dynamics_batch
from ron.datasets.collision_world import CollisionWorldConfig, make_collision_batch
from ron.networks import RONPureDynamicsNet, RONTriaxialTeacherNet, RONHybridTriFamilyNet

def main():
    ap = argparse.ArgumentParser("Use RON V20.1 networks")
    ap.add_argument("--network", choices=["pure", "triaxial", "hybrid"], default="hybrid")
    ap.add_argument("--dataset", choices=["dynamics", "collision"], default="collision")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.dataset == "collision":
        batch = make_collision_batch(CollisionWorldConfig(batch_size=4, seed=42), device="cpu")
    else:
        batch = make_dynamics_batch(batch_size=4, seed=42, device="cpu")

    kwargs = dict(lite_cells=8, memory_slots=16, grid=1, cube=3, moment_degree=1, signature_degree=1, harmonic_degree=1, internal_cycles=1, device="cpu")
    if args.network == "pure":
        net = RONPureDynamicsNet(**kwargs)
        out = net.infer_axes(batch.axes, batch.energy, batch.mask)
        payload = {"network": "pure", "packet": out.packet.as_dict(), "summary": out.summary}
    elif args.network == "triaxial":
        net = RONTriaxialTeacherNet(**kwargs)
        out = net.infer_axes(batch.axes, batch.energy, batch.mask)
        payload = {"network": "triaxial", "packet": out.packet.as_dict(), "summary": out.summary, "teacher": out.teacher}
    else:
        net = RONHybridTriFamilyNet(**kwargs)
        out = net.infer_states(batch.x_state, batch.y_state, batch.z_state, batch.mask)
        payload = {"network": "hybrid", "packet": out.packet.as_dict(), "summary": out.summary, "teacher": out.teacher, "hybrid_route": out.hybrid_route}
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
