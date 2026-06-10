from __future__ import annotations
import argparse, json, time
import torch
from ron.experiments.scientific_protocol import ScientificProtocolConfig, run_scientific_protocol

def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA demandé mais indisponible.")
    return choice

def main():
    ap = argparse.ArgumentParser("RON V20.3 six-hour oriented protocol")
    ap.add_argument("--budget-hours", type=float, default=6.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--profile", choices=["safe", "balanced", "strong"], default="balanced")
    ap.add_argument("--out-dir", default="runs/v20_3_six_hour_protocol")
    args = ap.parse_args()
    device = resolve_device(args.device)

    # Conservative presets. They do not stop exactly at 6h; they choose a run
    # expected to fit common hardware. User can tune manually if needed.
    if device == "cuda":
        presets = {
            "safe": dict(steps=300, batch_size=128, seeds=10),
            "balanced": dict(steps=600, batch_size=256, seeds=10),
            "strong": dict(steps=1000, batch_size=256, seeds=15),
        }
    else:
        presets = {
            "safe": dict(steps=200, batch_size=64, seeds=10),
            "balanced": dict(steps=500, batch_size=64, seeds=10),
            "strong": dict(steps=1000, batch_size=64, seeds=10),
        }
    params = presets[args.profile]
    started = time.time()
    summary = run_scientific_protocol(ScientificProtocolConfig(
        steps=params["steps"],
        batch_size=params["batch_size"],
        seeds=params["seeds"],
        out_dir=args.out_dir,
        device=device,
    ))
    print(json.dumps({
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "profile": args.profile,
        "budget_hours_requested": args.budget_hours,
        "chosen_params": params,
        "elapsed_seconds": time.time() - started,
        "metric_summary": summary["metric_summary"],
        "comparisons": summary["comparisons"],
        "final_csv": summary["final_csv"],
    }, indent=2))

if __name__ == "__main__":
    main()
