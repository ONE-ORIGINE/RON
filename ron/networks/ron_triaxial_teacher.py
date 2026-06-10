from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.bridges.triaxial_lens import RONTriaxialLens
from ron.networks.ron_pure_dynamics import RONPureDynamicsNet

@dataclass
class RONTriaxialTeacherOutput:
    packet: object
    summary: dict
    teacher: dict

class RONTriaxialTeacherNet:
    """RON + triaxial diagnostic network.

    It uses the RON-native TriaxialLens by default. A brute triaxial teacher can
    be plugged in later through ron.hybrid.triaxial_brute, but RON remains the
    conductor and packet source.
    """
    def __init__(self, device: str | None = None, **kwargs):
        self.ron = RONPureDynamicsNet(device=device, **kwargs)
        self.lens = RONTriaxialLens()

    def infer_axes(self, axes: Tensor, energy: Tensor, mask: Tensor | None = None) -> RONTriaxialTeacherOutput:
        lens = self.lens(axes.to(self.ron.device))
        out = self.ron.infer_axes(axes, energy, mask)
        summary = dict(out.summary)
        summary.update({
            "triaxial_teacher_coherence_gap": lens.coherence_gap,
            "triaxial_teacher_recommended_mode": lens.recommended_mode,
            "triaxial_teacher_mode_count": len(lens.mode_costs),
        })
        return RONTriaxialTeacherOutput(out.packet, summary, {
            "coherence_gap": lens.coherence_gap,
            "recommended_mode": lens.recommended_mode,
            "mode_costs": lens.mode_costs,
            "missing_axis_pressure": lens.missing_axis_pressure,
        })
