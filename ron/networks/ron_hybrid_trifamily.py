from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.hybrid import HybridRouter, FamilySignal, NeuralFamily
from ron.networks.ron_triaxial_teacher import RONTriaxialTeacherNet

@dataclass
class RONHybridOutput:
    packet: object
    summary: dict
    hybrid_route: dict
    teacher: dict

class SimpleStateCodec:
    """Boundary codec for numeric states.

    This is a deterministic non-learning codec: states -> axes/energy.
    It is included to demonstrate the hybrid family interface without making a
    classical neural core.
    """
    def __call__(self, x_state: Tensor, y_state: Tensor, z_state: Tensor):
        def axis(s):
            v = s[..., :3] + 0.15*s[..., 3:6]
            return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        axes = torch.stack([axis(x_state), axis(y_state), axis(z_state)], dim=1)
        energy = torch.stack([
            0.35 + x_state[..., 3:6].norm(dim=-1),
            0.45 + y_state[..., 3:6].norm(dim=-1),
            0.40 + z_state[..., 3:6].norm(dim=-1),
        ], dim=1).clamp(0.05, 2.0)
        return axes, energy

class RONHybridTriFamilyNet:
    """Three-family hybrid network.

    classical codec -> triaxial diagnostic -> RON decision.
    Classical/triaxial signals are accepted only as boundary/teacher signals.
    """
    def __init__(self, device: str | None = None, **kwargs):
        self.codec = SimpleStateCodec()
        self.ron_teacher = RONTriaxialTeacherNet(device=device, **kwargs)
        self.router = HybridRouter(allow_boundary_codecs=True, allow_triaxial_teacher=True)

    def infer_states(self, x_state: Tensor, y_state: Tensor, z_state: Tensor, mask: Tensor | None = None) -> RONHybridOutput:
        axes, energy = self.codec(x_state, y_state, z_state)
        out = self.ron_teacher.infer_axes(axes, energy, mask)
        route = self.router.route([
            FamilySignal(NeuralFamily.CLASSICAL, "state_codec", {"kind": "deterministic_numeric"}, confidence=0.5, may_decide=False),
            FamilySignal(NeuralFamily.TRIAXIAL_BRUTE, "xyz_teacher", out.teacher, confidence=0.5, may_decide=False),
            FamilySignal(NeuralFamily.RON, "active_geodesic_conductor", out.summary, confidence=1.0, may_decide=True),
        ])
        summary = dict(out.summary)
        summary.update({
            "hybrid_accepted_count": len(route.accepted),
            "hybrid_rejected_count": len(route.rejected),
            "hybrid_policy_ok": len(route.rejected) == 0,
        })
        return RONHybridOutput(
            packet=out.packet,
            summary=summary,
            hybrid_route={
                "accepted": [(s.family.value, s.role) for s in route.accepted],
                "rejected": [(s.family.value, s.role) for s in route.rejected],
                "reason": route.reason,
            },
            teacher=out.teacher,
        )
