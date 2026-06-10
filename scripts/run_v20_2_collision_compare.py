from __future__ import annotations
import argparse, json
from ron.experiments.collision_compare import CollisionExperimentConfig, run_collision_comparison

def main():
    ap = argparse.ArgumentParser("RON V20.2 collision comparison")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.028)
    ap.add_argument("--out-dir", default="runs/v20_2_collision_compare")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = CollisionExperimentConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, out_dir=args.out_dir, device="cpu")
    summary = run_collision_comparison(cfg)
    print(json.dumps({
        "config": summary["config"],
        "final": summary["final"],
        "csv": summary["csv"],
    }, indent=2))

if __name__ == "__main__":
    main()
