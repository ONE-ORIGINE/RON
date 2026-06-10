from __future__ import annotations
import argparse, json
import torch
from ron import RONSystem, validate_packet

def resolve_device(choice: str):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    return choice

def main():
    ap = argparse.ArgumentParser("RON V20 inference")
    ap.add_argument("--x", type=int, default=2)
    ap.add_argument("--y", type=int, default=5)
    ap.add_argument("--z", type=int, default=7)
    ap.add_argument("--mask", type=int, default=3)
    ap.add_argument("--profile", default="active_geodesic", choices=["stable", "geodesic_symphony", "active_geodesic"])
    ap.add_argument("--mode", default="ron_pure", choices=["ron_pure", "ron_triaxial", "ron_hybrid"])
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--cpu", action="store_true", help="legacy alias: force CPU")
    args = ap.parse_args()
    device = "cpu" if args.cpu else resolve_device(args.device)
    ron = RONSystem(
        mode=args.mode,
        profile=args.profile,
        device=device,
        lite_cells=8,
        memory_slots=16,
        grid=1,
        cube=3,
        moment_degree=1,
        signature_degree=1,
        harmonic_degree=1,
        internal_cycles=1,
    )
    out = ron.infer_triplet(args.x, args.y, args.z, mask=args.mask)
    pv = validate_packet(out.packet)
    print(json.dumps({
        "device": device,
        "packet_contract_ok": pv.ok,
        "packet_contract_errors": pv.errors,
        "packet": out.packet.as_dict(),
        "summary": out.summary,
        "trace": [t.__dict__ for t in out.trace],
    }, indent=2))

if __name__ == "__main__":
    main()
