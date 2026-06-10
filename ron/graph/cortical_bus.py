from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.ports import face_normals
from .rich_synapse import RichRONSynapseBank

@dataclass
class BusState:
    boundaries: dict[str, Tensor]
    axes: dict[str, Tensor]
    transported: dict[str, Tensor]
    feedback_axes: dict[str, Tensor]
    total_flow: Tensor
    conductor_feedback: Tensor
    conductor_energy: Tensor


def boundary_to_axis(boundary: Tensor) -> Tensor:
    n = face_normals(boundary.device, boundary.dtype)
    p = boundary.softmax(-1)
    axis = p @ n
    return axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

class RONCorticalBus:
    """RON-native cortical bus.

    The bus is not a dense connector. It gathers port boundaries from populations,
    passes them through rich RON synapses, then converts transported port-energy
    back into axes that can drive populations again.
    """
    def __init__(self, device: str|torch.device|None=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.synapses = RichRONSynapseBank(RichRONSynapseBank.default_edges(), device=self.device)

    def step(self, boundaries: dict[str, Tensor], axes: dict[str, Tensor], residual: float, release: float, dissonance: float) -> BusState:
        # Normalize keys to tensors on bus device.
        b = {k: v.to(self.device) for k, v in boundaries.items()}
        a = {k: v.to(self.device) for k, v in axes.items()}
        transported = self.synapses.transport(b, a)
        self.synapses.plasticity(b, transported, residual=residual, release=release, dissonance=dissonance)
        feedback_axes = {k: boundary_to_axis(v) for k, v in transported.items() if v.abs().sum() > 0}
        if transported:
            total_flow = torch.stack([v.abs().mean() for v in transported.values()]).sum()
        else:
            total_flow = torch.zeros((), device=self.device)
        if "conductor" in feedback_axes:
            conductor_feedback = feedback_axes["conductor"]
            conductor_energy = transported["conductor"].abs().mean(-1)
        else:
            B = next(iter(b.values())).shape[0]
            conductor_feedback = torch.zeros(B, 3, device=self.device)
            conductor_feedback[:, 2] = 1.0
            conductor_energy = torch.zeros(B, device=self.device)
        return BusState(b, a, transported, feedback_axes, total_flow, conductor_feedback, conductor_energy)

    def summary(self) -> dict:
        return self.synapses.summary()
