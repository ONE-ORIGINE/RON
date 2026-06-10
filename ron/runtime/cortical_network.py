from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.whole_core import RONWholeTriAxialCore
from ron.core.ports import face_normals
from .cortex import RONCortex
from .packet import RONPacket

def boundary_to_axis(boundary: Tensor) -> Tensor:
    """Convert a 6-face RON boundary back into a 3D axis.

    This is how one RON population speaks to another: not by a dense vector head,
    but by port-energy geometry.
    """
    n = face_normals(boundary.device, boundary.dtype)
    p = boundary.softmax(-1)
    axis = p @ n
    return axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def output_to_axes(out, mode: str) -> Tensor:
    if mode == "spatial":
        return torch.stack([
            boundary_to_axis(out.past),
            boundary_to_axis(out.present),
            boundary_to_axis(out.future),
        ], dim=1)
    if mode == "temporal":
        return torch.stack([
            boundary_to_axis(out.past + out.reverse),
            boundary_to_axis(out.diagonal + out.present),
            boundary_to_axis(out.future + out.consequence),
        ], dim=1)
    if mode == "causal":
        return torch.stack([
            boundary_to_axis(out.cause + out.past),
            boundary_to_axis(out.present + out.diagonal),
            boundary_to_axis(out.consequence + out.future),
        ], dim=1)
    if mode == "memory":
        return torch.stack([
            boundary_to_axis(out.past + 0.25*out.diagonal),
            boundary_to_axis(out.present),
            boundary_to_axis(out.future + 0.25*out.consequence),
        ], dim=1)
    # conductor
    return torch.stack([
        boundary_to_axis(out.past + out.cause),
        boundary_to_axis(out.main + out.diagonal),
        boundary_to_axis(out.future + out.consequence),
    ], dim=1)

@dataclass
class PopulationTrace:
    role: str
    release: float
    access: float
    residual: float
    energy: float
    port_entropy: float
    loop_q_delta: float

class RONCorticalNetwork:
    """Multi-population RON network.

    This is not a stack of dense layers. Each population is a RON core. Populations
    communicate by converting boundary/port energy back into oriented axes.
    """
    ROLES = ["spatial", "temporal", "causal", "memory", "conductor"]

    def __init__(self, device: str|None=None, **core_kwargs):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cores = {r: RONWholeTriAxialCore(**core_kwargs).to(self.device).eval() for r in self.ROLES}
        self.cortex = RONCortex(self.cores["conductor"], device=str(self.device))

    def parameters(self):
        for c in self.cores.values():
            yield from c.parameters()

    def infer_triplet(self, x:int, y:int, z:int, mask:int=0, adaptive_cycles:bool=True) -> tuple[RONPacket, list[PopulationTrace]]:
        axes = None
        triplet = torch.tensor([[x,y,z]], device=self.device, dtype=torch.long)
        mask_t = torch.tensor([mask], device=self.device, dtype=torch.long)
        traces: list[PopulationTrace] = []
        out = None
        with torch.no_grad():
            for role in self.ROLES:
                core = self.cores[role]
                if axes is None:
                    out = core(triplet=triplet, mask=mask_t, adaptive_cycles=adaptive_cycles)
                else:
                    out = core(axes=axes, mask=mask_t, adaptive_cycles=adaptive_cycles)
                d = out.diagnostics
                traces.append(PopulationTrace(
                    role=role,
                    release=float(d["release"].detach().cpu()),
                    access=float(d["branch_access"].detach().cpu()),
                    residual=float(d["residual_cost"].detach().cpu()),
                    energy=float(d["energy_mean"].detach().cpu()),
                    port_entropy=float(d["port_entropy"].detach().cpu()),
                    loop_q_delta=float(d.get("loop_q_delta", torch.tensor(0.)).detach().cpu()),
                ))
                axes = output_to_axes(out, role)
            fields = self.cortex.fields_from_output(out)
            packet = self.cortex.packet_from_output(out, fields)
        return packet, traces
