from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.runtime.integrated_cortex import RONIntegratedCortex, IntegratedTrace
from ron.graph.role_governor import RONRoleGovernor
from ron.runtime.ron_invariants import validate_packet_and_bus, packet_coherence

@dataclass
class StableRONTrace:
    stage: str
    release: float
    residual: float
    flow: float
    coherence: float
    corrected: bool

class RONStableIntegratedCortex:
    """RON V20 stable integrated cortex.

    It keeps the integrated cortex intact and adds:
    - packet/bus invariant checks;
    - role-governed feedback;
    - one bounded corrective conductor pass only when coherence says it is useful.
    """
    def __init__(self, device: str|None=None, lite_cells:int=10, **core_kwargs):
        self.integrated = RONIntegratedCortex(device=device, lite_cells=lite_cells, **core_kwargs)
        self.device = self.integrated.device
        self.governor = RONRoleGovernor()

    def infer_triplet(self, x:int, y:int, z:int, mask:int=0, adaptive_cycles:bool=True):
        packet0, trace0, bus0 = self.integrated.infer_triplet(x,y,z,mask,adaptive_cycles)
        inv0 = validate_packet_and_bus(packet0, bus0)
        coh0 = packet_coherence(packet0)

        corrected = False
        packet = packet0
        bus = dict(bus0)
        trace = [
            StableRONTrace("integrated_core", float(packet0.release), float(packet0.residual_cost), float(bus0.get("bus_total_flow",0.0)), coh0["coherence_score"], False)
        ]

        # Use existing transported bus through role governor when coherence suggests correction.
        # This is not a loop; it is a bounded one-shot correction.
        if coh0["correction_pressure"] > 0.20:
            corrected = True
            # Reconstruct boundaries/axes from a fresh deep trace, then govern transported signal.
            packet_pre, deep_trace = self.integrated.deep.infer_triplet(x,y,z,mask=mask,adaptive_cycles=adaptive_cycles)
            boundaries, axes = self.integrated._trace_to_boundaries(deep_trace)
            d = packet_pre.diagnostics
            bus_state = self.integrated.bus.step(
                boundaries,
                axes,
                residual=float(d.get("residual_cost", packet_pre.residual_cost)),
                release=float(packet_pre.release),
                dissonance=float(d.get("dissonance", 0.0)),
            )
            governed = self.governor.apply(
                bus_state.transported,
                residual=float(packet_pre.residual_cost),
                release=float(packet_pre.release),
                dissonance=float(d.get("dissonance", 0.0)),
            )
            fb = governed.conductor_axis.to(self.device)
            base = axes.get("conductor", fb)
            mid = (0.50*base + 0.50*fb)
            mid = mid / mid.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            axes2 = torch.stack([-fb, mid, torch.roll(fb, shifts=1, dims=-1)], dim=1)
            with torch.no_grad():
                out = self.integrated.deep.core(
                    axes=axes2,
                    mask=torch.tensor([mask], device=self.device, dtype=torch.long),
                    adaptive_cycles=adaptive_cycles,
                )
                fields = self.integrated.deep.cortex.fields_from_output(out)
                packet = self.integrated.deep.cortex.packet_from_output(out, fields)
            bus.update({
                "governed_flow_before": governed.total_flow_before,
                "governed_flow_after": governed.total_flow_after,
                "role_gates": governed.role_gates,
                "stable_correction_applied": True,
            })
            coh1 = packet_coherence(packet)
            trace.append(
                StableRONTrace("role_governed_correction", float(packet.release), float(packet.residual_cost), governed.total_flow_after, coh1["coherence_score"], True)
            )
        else:
            bus["stable_correction_applied"] = False
            bus["role_gates"] = {}

        inv = validate_packet_and_bus(packet, bus)
        coh = packet_coherence(packet)
        bus.update({
            "stable_invariants_ok": inv.ok,
            "stable_invariant_notes": inv.notes,
            "coherence_score": coh["coherence_score"],
            "correction_pressure": coh["correction_pressure"],
            "release_consistent": coh["release_consistent"],
            "stable_corrected": corrected,
        })
        return packet, trace, bus
