from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv, json, time
from ron.experiments.collision_compare import CollisionExperimentConfig, run_collision_comparison
from ron.experiments.statistics import mean_std, bootstrap_ci, paired_differences, cohen_d

def _unique_out_dir(path: Path) -> Path:
    """Avoid overwriting protocol outputs by creating a timestamped folder when needed."""
    if not path.exists():
        return path
    markers = ["all_steps_metrics.csv", "final_metrics_by_seed.csv", "scientific_protocol_summary.json"]
    if not any((path / m).exists() for m in markers):
        return path
    import time
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{stamp}")
    i = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{stamp}_{i:02d}")
        i += 1
    return candidate

@dataclass
class ScientificProtocolConfig:
    steps: int = 12
    batch_size: int = 12
    seeds: int = 5
    base_seed: int = 1000
    lr: float = 0.028
    out_dir: str = "runs/v20_3_scientific_protocol"
    device: str = "cpu"

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _variant_series(final_rows, metric):
    variants = sorted(set(r["variant"] for r in final_rows))
    return {v: [float(r[metric]) for r in final_rows if r["variant"] == v] for v in variants}

def run_scientific_protocol(cfg: ScientificProtocolConfig | None = None):
    cfg = cfg or ScientificProtocolConfig()
    out_dir = _unique_out_dir(Path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    final_rows = []
    start = time.time()

    for i in range(cfg.seeds):
        seed = cfg.base_seed + i * 17
        run_dir = out_dir / f"seed_{seed}"
        comp = run_collision_comparison(CollisionExperimentConfig(
            steps=cfg.steps,
            batch_size=cfg.batch_size,
            seed=seed,
            lr=cfg.lr,
            out_dir=str(run_dir),
            device=cfg.device,
        ))
        for r in comp["rows"]:
            rr = dict(r)
            rr["seed"] = seed
            all_rows.append(rr)
        for variant, r in comp["final"].items():
            rr = dict(r)
            rr["seed"] = seed
            final_rows.append(rr)

    all_csv = out_dir / "all_steps_metrics.csv"
    final_csv = out_dir / "final_metrics_by_seed.csv"
    _write_csv(all_csv, all_rows)
    _write_csv(final_csv, final_rows)

    metric_summary = {}
    for metric in ["total_loss", "action_acc", "cause_acc", "consequence_acc", "residual", "release", "geodesic_loss"]:
        series = _variant_series(final_rows, metric)
        metric_summary[metric] = {
            v: {
                **mean_std(vals),
                "ci95": bootstrap_ci(vals, seed=cfg.base_seed + len(metric)),
            }
            for v, vals in series.items()
        }

    comparisons = {}
    for metric, prefer_lower in [("total_loss", True), ("action_acc", False), ("cause_acc", False), ("consequence_acc", False)]:
        series = _variant_series(final_rows, metric)
        if "pure" in series:
            for challenger in ["triaxial", "hybrid"]:
                if challenger in series:
                    # diff is challenger - pure. For loss, negative is better. For accuracy, positive is better.
                    diff = paired_differences(series[challenger], series["pure"])
                    comparisons[f"{challenger}_minus_pure_{metric}"] = {
                        "mean_diff": mean_std(diff)["mean"],
                        "std_diff": mean_std(diff)["std"],
                        "ci95": bootstrap_ci(diff, seed=cfg.base_seed + 31),
                        "cohen_d": cohen_d(diff),
                        "better_count": sum((d < 0 if prefer_lower else d > 0) for d in diff),
                        "n": len(diff),
                        "prefer_lower": prefer_lower,
                    }

    summary = {
        "version": "v20.3",
        "purpose": "scientific reproducibility protocol for RON variants",
        "config": asdict(cfg),
        "elapsed_seconds": time.time() - start,
        "all_steps_csv": str(all_csv),
        "final_csv": str(final_csv),
        "metric_summary": metric_summary,
        "comparisons": comparisons,
        "final_rows": final_rows,
        "evidence_grade": {
            "level": "prototype_reproducible_protocol",
            "meaning": "CPU-safe replicated evidence; not a universal proof and not a peer-reviewed large-scale benchmark.",
            "six_hour_guidance": "Use more seeds and steps within the 6h budget; prefer seeds>=10 and steps>=100 if hardware allows.",
        },
    }
    (out_dir / "scientific_protocol_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
