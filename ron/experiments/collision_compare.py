from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv, json
import torch
import torch.nn.functional as F
from ron.core.whole_core import RONWholeTriAxialCore
from ron.core.geodesic_pressure import geodesic_pressure
from ron.datasets.collision_world import CollisionWorldConfig, make_collision_batch
from ron.networks.ron_hybrid_trifamily import SimpleStateCodec

@dataclass
class CollisionExperimentConfig:
    steps: int = 6
    batch_size: int = 10
    seed: int = 404
    lr: float = 0.028
    out_dir: str = "runs/v20_2_collision_compare"
    device: str = "cpu"
    vocab: int = 12
    grid: int = 1
    cube: int = 3
    moment_degree: int = 1
    signature_degree: int = 1
    harmonic_degree: int = 1
    internal_cycles: int = 1

def _accuracy(logits, target):
    return float((logits.argmax(dim=-1) == target).float().mean().detach().cpu())

def _diag(out, key, default=0.0):
    v = out.diagnostics.get(key, None)
    if v is None:
        return torch.as_tensor(default, device=out.main.device, dtype=out.main.dtype)
    return v if torch.is_tensor(v) else torch.as_tensor(v, device=out.main.device, dtype=out.main.dtype)

def _loss(out, batch, axes, energy, variant: str):
    route = (
        F.cross_entropy(out.main, batch.action_face.long())
        + 0.45 * F.cross_entropy(out.cause, batch.cause_face.long())
        + 0.45 * F.cross_entropy(out.consequence, batch.consequence_face.long())
    )
    residual = _diag(out, "residual_cost").mean()
    phi = _diag(out, "phi_pressure").abs().mean()
    dissonance = _diag(out, "dissonance").mean()
    entropy_penalty = 1.0 - _diag(out, "port_entropy", 1.0).mean()
    release = _diag(out, "release").mean()
    release_target = batch.should_release.float().mean().to(release.device)
    release_loss = (release - release_target).pow(2)
    total = route + 0.08*residual + 0.04*phi + 0.04*dissonance + 0.03*entropy_penalty + 0.06*release_loss

    geo_cost = torch.zeros((), device=out.main.device)
    trans_cost = torch.zeros((), device=out.main.device)
    if variant in ("triaxial", "hybrid"):
        geo = geodesic_pressure(axes, energy, strength=0.0, cube_n=3)
        geo_cost = torch.log1p(geo.geodesic_cost).mean()
        trans_cost = torch.log1p(geo.transport_cost).mean()
        total = total + 0.035*geo_cost + 0.020*trans_cost
    return total, {
        "route_loss": float(route.detach().cpu()),
        "residual": float(residual.detach().cpu()),
        "phi_pressure": float(phi.detach().cpu()),
        "dissonance": float(dissonance.detach().cpu()),
        "port_entropy": float(_diag(out, "port_entropy", 1.0).detach().cpu()),
        "release": float(release.detach().cpu()),
        "geodesic_loss": float(geo_cost.detach().cpu()),
        "transport_loss": float(trans_cost.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "action_acc": _accuracy(out.main, batch.action_face),
        "cause_acc": _accuracy(out.cause, batch.cause_face),
        "consequence_acc": _accuracy(out.consequence, batch.consequence_face),
    }

def _model(cfg: CollisionExperimentConfig, seed_offset: int):
    torch.manual_seed(cfg.seed + seed_offset)
    return RONWholeTriAxialCore(
        vocab=cfg.vocab,
        grid=cfg.grid,
        cube=cfg.cube,
        moment_degree=cfg.moment_degree,
        signature_degree=cfg.signature_degree,
        harmonic_degree=cfg.harmonic_degree,
        internal_cycles=cfg.internal_cycles,
        seed=cfg.seed + seed_offset,
    ).to(cfg.device)

def run_collision_comparison(cfg: CollisionExperimentConfig | None = None):
    cfg = cfg or CollisionExperimentConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codec = SimpleStateCodec()
    variants = {
        "pure": _model(cfg, 0),
        "triaxial": _model(cfg, 11),
        "hybrid": _model(cfg, 23),
    }
    opts = {k: torch.optim.AdamW(v.parameters(), lr=cfg.lr, weight_decay=1e-4) for k, v in variants.items()}
    rows = []

    for step in range(cfg.steps + 1):
        batch = make_collision_batch(CollisionWorldConfig(batch_size=cfg.batch_size, seed=cfg.seed + step), device=cfg.device)
        for name, model in variants.items():
            model.train(step > 0)
            if name == "hybrid":
                axes, energy = codec(batch.x_state, batch.y_state, batch.z_state)
                axes, energy = axes.to(cfg.device), energy.to(cfg.device)
            else:
                axes, energy = batch.axes.to(cfg.device), batch.energy.to(cfg.device)

            out = model(axes=axes, mask=batch.mask.to(cfg.device), adaptive_cycles=True)
            loss, parts = _loss(out, batch, axes, energy, name)
            if step > 0:
                opts[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opts[name].step()
                if hasattr(model, "project_parameters_"):
                    model.project_parameters_()
            row = {"step": step, "variant": name, **parts}
            rows.append(row)

    # Save CSV and JSON.
    csv_path = out_dir / "collision_compare_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final = {name: [r for r in rows if r["variant"] == name][-1] for name in variants}
    summary = {
        "config": asdict(cfg),
        "rows": rows,
        "final": final,
        "csv": str(csv_path),
    }
    (out_dir / "collision_compare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
