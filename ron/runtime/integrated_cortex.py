from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.runtime.deep_cortex import RONDeepCortex, boundary_to_axis
from ron.graph.cortical_bus import RONCorticalBus
from .packet import RONPacket

@dataclass
class IntegratedTrace:
    stage: str
    boundary_energy: float
    release: float
    residual: float
    flow: float
    changed_packet: bool

class RONIntegratedCortex:
    """V20 integrated cortex.

    This corrects the previous mistaken 50x instrumentation. The integrated cortex
    performs a useful RON-native step: deep population processing, cortical bus
    transport, then conductor feedback. It is adaptive and structural, not an
    artificial fixed macro-loop.
    """
    def __init__(self, device: str|None=None, lite_cells:int=10, **core_kwargs):
        self.deep = RONDeepCortex(device=device, lite_cells=lite_cells, **core_kwargs)
        self.device = self.deep.device
        self.bus = RONCorticalBus(device=self.device)

    def _trace_to_boundaries(self, trace) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        boundaries: dict[str, Tensor] = {}
        axes: dict[str, Tensor] = {}
        for t in trace:
            role = "conductor" if t.stage == "full_core_conductor" else t.stage
            b = torch.tensor([t.boundary], device=self.device, dtype=torch.float32)
            if b.shape[-1] == 6:
                boundaries[role] = b
                axes[role] = boundary_to_axis(b)
        return boundaries, axes

    def infer_triplet(self, x:int, y:int, z:int, mask:int=0, adaptive_cycles:bool=True) -> tuple[RONPacket, list[IntegratedTrace], dict]:
        packet0, trace0 = self.deep.infer_triplet(x, y, z, mask=mask, adaptive_cycles=adaptive_cycles)
        boundaries, axes = self._trace_to_boundaries(trace0)
        d0 = packet0.diagnostics
        bus_state = self.bus.step(
            boundaries,
            axes,
            residual=float(d0.get("residual_cost", packet0.residual_cost)),
            release=float(packet0.release),
            dissonance=float(d0.get("dissonance", 0.0)),
        )
        # Feedback is RON-native: conductor receives axes made from transported port-energy.
        fb = bus_state.conductor_feedback
        base = axes.get("conductor", fb)
        axes2 = torch.stack([
            -fb,
            (0.55*base + 0.45*fb) / (0.55*base + 0.45*fb).norm(dim=-1, keepdim=True).clamp_min(1e-6),
            torch.roll(fb, shifts=1, dims=-1),
        ], dim=1)
        with torch.no_grad():
            out = self.deep.core(axes=axes2, mask=torch.tensor([mask], device=self.device, dtype=torch.long), adaptive_cycles=adaptive_cycles)
            fields = self.deep.cortex.fields_from_output(out)
            packet1 = self.deep.cortex.packet_from_output(out, fields)
        changed = packet0.action_face != packet1.action_face or abs(packet0.release - packet1.release) > 1e-5
        trace = [
            IntegratedTrace("deep_pre_bus", float(sum([abs(v).mean().detach().cpu() for v in boundaries.values()])), float(packet0.release), float(packet0.residual_cost), 0.0, False),
            IntegratedTrace("cortical_bus", float(bus_state.total_flow.detach().cpu()), float(packet0.release), float(packet0.residual_cost), float(bus_state.total_flow.detach().cpu()), False),
            IntegratedTrace("conductor_feedback", float(bus_state.conductor_energy.mean().detach().cpu()), float(packet1.release), float(packet1.residual_cost), float(bus_state.total_flow.detach().cpu()), bool(changed)),
        ]
        summary = self.bus.summary()
        summary.update({"feedback_changed_packet": bool(changed), "bus_total_flow": float(bus_state.total_flow.detach().cpu())})
        return packet1, trace, summary
