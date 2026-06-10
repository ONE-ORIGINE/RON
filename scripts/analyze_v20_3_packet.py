from __future__ import annotations
import argparse, json
from ron.experiments.packet_analysis import PacketAnalysisConfig, run_packet_analysis

def main():
    ap = argparse.ArgumentParser("RON V20.3 packet analysis")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--network", choices=["pure", "triaxial", "hybrid"], default="hybrid")
    ap.add_argument("--out-dir", default="runs/v20_3_packet_analysis")
    args = ap.parse_args()
    report = run_packet_analysis(PacketAnalysisConfig(samples=args.samples, network=args.network, out_dir=args.out_dir))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
