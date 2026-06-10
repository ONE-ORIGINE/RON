from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.graph.cortical_bus import boundary_to_axis

@dataclass
class GovernedBus:
    transported: dict[str, Tensor]
    gated: dict[str, Tensor]
    role_gates: dict[str, float]
    conductor_axis: Tensor
    total_flow_before: float
    total_flow_after: float

class RONRoleGovernor:
    """RON-native role governor.

    The governor does not add a dense layer. It gates transported port-energy
    per population according to residual, release and dissonance. It is the
    V20 way to make conductor feedback less arbitrary and more stable.
    """
    def __init__(self, min_gate: float=0.20, max_gate: float=1.35):
        self.min_gate = min_gate
        self.max_gate = max_gate

    def role_gate(self, role: str, residual: float, release: float, dissonance: float) -> float:
        # RON-native policy:
        # - spatial/temporal stay active under uncertainty;
        # - causal/memory strengthen when residual/dissonance are high;
        # - value/conductor strengthen when release is plausible.
        if role in ("causal", "memory"):
            g = 0.75 + 0.55*residual + 0.35*dissonance
        elif role in ("spatial", "temporal", "sensory"):
            g = 0.85 + 0.25*(1.0-release) + 0.20*dissonance
        elif role == "value":
            g = 0.70 + 0.45*release - 0.20*residual
        elif role == "conductor":
            g = 0.80 + 0.35*release + 0.20*(1.0-dissonance)
        else:
            g = 1.0
        return float(max(self.min_gate, min(self.max_gate, g)))

    def apply(self, transported: dict[str, Tensor], residual: float, release: float, dissonance: float) -> GovernedBus:
        gated: dict[str, Tensor] = {}
        gates: dict[str, float] = {}
        for role, boundary in transported.items():
            gate = self.role_gate(role, residual, release, dissonance)
            gates[role] = gate
            gated[role] = boundary * gate

        before = sum(float(v.abs().mean().detach().cpu()) for v in transported.values()) if transported else 0.0
        after = sum(float(v.abs().mean().detach().cpu()) for v in gated.values()) if gated else 0.0

        if "conductor" in gated and gated["conductor"].abs().sum() > 0:
            axis = boundary_to_axis(gated["conductor"])
        elif gated:
            # If conductor received no direct signal, merge all gated boundaries.
            merged = torch.stack([v for v in gated.values()]).mean(0)
            axis = boundary_to_axis(merged)
        else:
            axis = torch.zeros(1, 3)
            axis[:, 2] = 1.0
        return GovernedBus(gated, gated, gates, axis, before, after)
