from __future__ import annotations
import argparse, json
import torch
from ron.learning.native_trainer import RONNativeTrainer, NativeTrainConfig

def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    return choice

def main():
    ap = argparse.ArgumentParser("RON V20 native small training")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.035)
    ap.add_argument("--out-dir", default="runs/native_v20")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--cpu", action="store_true", help="legacy alias: force CPU")
    args = ap.parse_args()
    device = "cpu" if args.cpu else resolve_device(args.device)
    cfg = NativeTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        out_dir=args.out_dir,
    )
    summary = RONNativeTrainer(cfg).run()
    print(json.dumps({"device": device, **summary}, indent=2))

if __name__ == "__main__":
    main()
