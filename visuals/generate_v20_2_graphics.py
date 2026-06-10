from __future__ import annotations
from pathlib import Path
import csv, json
import math
import matplotlib.pyplot as plt
import torch
from ron.experiments.collision_compare import CollisionExperimentConfig, run_collision_comparison
from ron.datasets.collision_world import CollisionWorldConfig, make_collision_batch
from ron.networks import RONHybridTriFamilyNet

OUT = Path("visuals/v20_2")
OUT.mkdir(parents=True, exist_ok=True)

def _load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            for k, v in list(rr.items()):
                if k not in ("variant",):
                    rr[k] = float(v)
            rows.append(rr)
    return rows

def _series(rows, variant, key):
    s = [r for r in rows if r["variant"] == variant]
    return [r["step"] for r in s], [r[key] for r in s]

def _line_chart(rows, key, title, ylabel, filename):
    fig = plt.figure(figsize=(8.5, 5.2))
    for variant in ["pure", "triaxial", "hybrid"]:
        x, y = _series(rows, variant, key)
        plt.plot(x, y, marker="o", label=variant)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.25)
    fig.savefig(OUT/filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _bar_final(summary, key, title, ylabel, filename):
    fig = plt.figure(figsize=(7.5, 5))
    names = ["pure", "triaxial", "hybrid"]
    vals = [float(summary["final"][n][key]) for n in names]
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    fig.savefig(OUT/filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _trajectory_plot():
    batch = make_collision_batch(CollisionWorldConfig(batch_size=1, seed=99), device="cpu")
    axes = batch.axes[0].detach().cpu()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    labels = ["X passé", "Y présent", "Z futur"]
    for i, label in enumerate(labels):
        ax.quiver(0, 0, 0, float(axes[i,0]), float(axes[i,1]), float(axes[i,2]), length=1.0, normalize=True)
        ax.text(float(axes[i,0]), float(axes[i,1]), float(axes[i,2]), label)
    ax.plot([float(v) for v in axes[:,0]], [float(v) for v in axes[:,1]], [float(v) for v in axes[:,2]], marker="o")
    ax.set_title("Trajectoire orientationnelle RON X → Y → Z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.savefig(OUT/"v20_2_orientation_trajectory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def _packet_route_plot():
    batch = make_collision_batch(CollisionWorldConfig(batch_size=6, seed=77), device="cpu")
    net = RONHybridTriFamilyNet(device="cpu", lite_cells=8, memory_slots=16, grid=1, cube=3, moment_degree=1, signature_degree=1, harmonic_degree=1, internal_cycles=1)
    out = net.infer_states(batch.x_state, batch.y_state, batch.z_state, batch.mask)
    packet = out.packet.as_dict()
    fig = plt.figure(figsize=(7.5, 4.8))
    faces = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    plt.bar(faces, packet["route_vector"])
    plt.title("RONPacket — distribution des ports")
    plt.ylabel("probabilité")
    fig.savefig(OUT/"v20_2_packet_route_vector.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def _hybrid_flow_plot():
    fig = plt.figure(figsize=(10, 4))
    stages = ["Classical\ncodec", "Triaxial\nteacher", "RON\nactive geodesic", "RONPacket"]
    xs = [0, 1.8, 3.6, 5.4]
    ys = [0, 0, 0, 0]
    plt.scatter(xs, ys, s=1800)
    for x, label in zip(xs, stages):
        plt.text(x, 0, label, ha="center", va="center")
    for i in range(len(xs)-1):
        plt.arrow(xs[i]+0.35, 0, 1.05, 0, head_width=0.08, length_includes_head=True)
    plt.ylim(-0.7, 0.7)
    plt.axis("off")
    plt.title("RON V20.2 — architecture hybride tri-famille")
    fig.savefig(OUT/"v20_2_hybrid_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    summary = run_collision_comparison(CollisionExperimentConfig(steps=6, batch_size=10, out_dir="runs/v20_2_collision_compare"))
    rows = _load_rows(summary["csv"])

    _line_chart(rows, "total_loss", "RON V20.2 — perte CollisionWorld", "loss", "v20_2_loss_curves.png")
    _line_chart(rows, "action_acc", "RON V20.2 — action accuracy", "accuracy", "v20_2_action_accuracy.png")
    _line_chart(rows, "residual", "RON V20.2 — residual cost", "residual", "v20_2_residual_curve.png")
    _line_chart(rows, "release", "RON V20.2 — release dynamics", "release", "v20_2_release_curve.png")
    _line_chart(rows, "geodesic_loss", "RON V20.2 — geodesic term", "geodesic loss", "v20_2_geodesic_curve.png")
    _bar_final(summary, "action_acc", "Final action accuracy", "accuracy", "v20_2_final_action_accuracy.png")
    _bar_final(summary, "total_loss", "Final loss", "loss", "v20_2_final_loss.png")
    _trajectory_plot()
    _packet_route_plot()
    _hybrid_flow_plot()

    report = {
        "summary": summary,
        "generated": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT/"v20_2_visual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "generated": report["generated"]}, indent=2))

if __name__ == "__main__":
    main()
