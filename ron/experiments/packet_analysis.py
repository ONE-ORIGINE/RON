from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch
from ron.datasets.collision_world import CollisionWorldConfig, make_collision_batch
from ron.networks import RONHybridTriFamilyNet, RONPureDynamicsNet, RONTriaxialTeacherNet

@dataclass
class PacketAnalysisConfig:
    samples: int = 24
    seed: int = 909
    out_dir: str = "runs/v20_3_packet_analysis"
    network: str = "hybrid"

def _confusion(preds, targets, n=6):
    m = [[0 for _ in range(n)] for _ in range(n)]
    for p, t in zip(preds, targets):
        m[int(t)][int(p)] += 1
    return m

def run_packet_analysis(cfg: PacketAnalysisConfig | None = None):
    cfg = cfg or PacketAnalysisConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = make_collision_batch(CollisionWorldConfig(batch_size=cfg.samples, seed=cfg.seed), device="cpu")
    kwargs = dict(device="cpu", lite_cells=8, memory_slots=16, grid=1, cube=3, moment_degree=1, signature_degree=1, harmonic_degree=1, internal_cycles=1)

    if cfg.network == "pure":
        net = RONPureDynamicsNet(**kwargs)
        out = net.infer_axes(batch.axes, batch.energy, batch.mask)
    elif cfg.network == "triaxial":
        net = RONTriaxialTeacherNet(**kwargs)
        out = net.infer_axes(batch.axes, batch.energy, batch.mask)
    else:
        net = RONHybridTriFamilyNet(**kwargs)
        out = net.infer_states(batch.x_state, batch.y_state, batch.z_state, batch.mask)

    # The official network packet is one aggregate packet from the batch path.
    packet = out.packet.as_dict()
    summary = out.summary
    # For a simple aggregate confusion proxy, use the same chosen action against all targets.
    pred = [packet["action_index"]] * cfg.samples
    target = batch.action_face.detach().cpu().tolist()
    cm = _confusion(pred, target)

    report = {
        "config": asdict(cfg),
        "packet": packet,
        "summary": summary,
        "target_distribution": {str(i): int((batch.action_face == i).sum()) for i in range(6)},
        "confusion_proxy": cm,
        "note": "This is packet-level aggregate analysis, not a full per-sample decoder.",
    }
    (out_dir / "packet_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
