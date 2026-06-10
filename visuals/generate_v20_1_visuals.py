from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
from ron.datasets.collision_world import CollisionWorldConfig, make_collision_batch
from ron.networks import RONHybridTriFamilyNet

OUT = Path("visuals/output")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    batch = make_collision_batch(CollisionWorldConfig(batch_size=8, seed=7), device="cpu")
    net = RONHybridTriFamilyNet(device="cpu", lite_cells=8, memory_slots=16, grid=1, cube=3, moment_degree=1, signature_degree=1, harmonic_degree=1, internal_cycles=1)
    out = net.infer_states(batch.x_state, batch.y_state, batch.z_state, batch.mask)
    packet = out.packet.as_dict()

    # 1. XYZ trajectory axes for first sample.
    axes = batch.axes[0].detach().cpu()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    labels = ["X passé", "Y présent", "Z futur"]
    for i in range(3):
        ax.quiver(0,0,0, float(axes[i,0]), float(axes[i,1]), float(axes[i,2]), length=1.0, normalize=True)
        ax.text(float(axes[i,0]), float(axes[i,1]), float(axes[i,2]), labels[i])
    ax.set_title("RON axes X/Y/Z")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.savefig(OUT/"ron_axes_xyz.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. Port route vector.
    fig = plt.figure()
    faces = ["+X","-X","+Y","-Y","+Z","-Z"]
    plt.bar(faces, packet["route_vector"])
    plt.title("RONPacket route_vector")
    plt.ylabel("probabilité")
    fig.savefig(OUT/"ron_packet_ports.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3. Diagnostic bars.
    fig = plt.figure()
    keys = ["active_geodesic_cost", "active_transport_cost", "active_phi_flux", "active_quaternion_order"]
    vals = [float(out.summary.get(k, 0.0)) for k in keys]
    plt.bar(keys, vals)
    plt.xticks(rotation=25, ha="right")
    plt.title("Diagnostics RON active_geodesic")
    fig.savefig(OUT/"ron_active_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 4. Hybrid route graph as simple staged diagram.
    fig = plt.figure()
    stages = ["classical\ncodec", "triaxial\nteacher", "RON\nconductor", "RONPacket"]
    xs = [0, 1, 2, 3]
    ys = [0, 0, 0, 0]
    plt.scatter(xs, ys, s=800)
    for x, label in zip(xs, stages):
        plt.text(x, 0, label, ha="center", va="center")
    for i in range(3):
        plt.arrow(xs[i]+0.18, 0, 0.55, 0, head_width=0.05, length_includes_head=True)
    plt.axis("off")
    plt.title("Réseau hybride tri-famille")
    fig.savefig(OUT/"ron_hybrid_trifamily_flow.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (OUT/"visual_summary.json").write_text(json.dumps({
        "packet": packet,
        "summary": out.summary,
        "hybrid_route": out.hybrid_route,
        "generated": [
            "ron_axes_xyz.png",
            "ron_packet_ports.png",
            "ron_active_diagnostics.png",
            "ron_hybrid_trifamily_flow.png",
        ]
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "files": sorted(p.name for p in OUT.iterdir())}, indent=2))

if __name__ == "__main__":
    main()
