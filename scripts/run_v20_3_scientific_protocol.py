from __future__ import annotations
import argparse, json
import torch
from ron.experiments.scientific_protocol import ScientificProtocolConfig, run_scientific_protocol

def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA demandé mais torch.cuda.is_available() == False. Lancez scripts/check_device.py.")
    return choice

def main():
    ap = argparse.ArgumentParser("RON V20.3 scientific protocol")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.028)
    ap.add_argument("--out-dir", default="runs/v20_3_scientific_protocol")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--cpu", action="store_true", help="Alias legacy: force CPU.")
    args = ap.parse_args()
    device = "cpu" if args.cpu else resolve_device(args.device)
    summary = run_scientific_protocol(ScientificProtocolConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seeds=args.seeds,
        base_seed=args.base_seed,
        lr=args.lr,
        out_dir=args.out_dir,
        device=device,
    ))
    print(json.dumps({
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "config": summary["config"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "metric_summary": summary["metric_summary"],
        "comparisons": summary["comparisons"],
        "final_csv": summary["final_csv"],
    }, indent=2))

if __name__ == "__main__":
    main()
