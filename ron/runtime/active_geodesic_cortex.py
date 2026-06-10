from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.runtime.geodesic_symphony_cortex import RONGeodesicSymphonyCortex
from ron.core.geodesic_pressure import geodesic_pressure
from ron.memory.geodesic_memory import RONGeodesicMemoryBank

@dataclass
class ActiveGeodesicTrace:
    stage: str
    release: float
    residual: float
    geodesic_cost: float
    transport_cost: float
    phi_flux: float
    pressure_strength: float
    memory_confidence: float
    changed: bool

class RONActiveGeodesicCortex:
    """RON V20 active geodesic cortex.

    The geodesic/symphony layer adds diagnostics; the active cortex uses them directly:
    geodesic pressure + Phi flux + symphony consensus + small geodesic memory
    modify the conductor axes before producing the final packet.
    """
    def __init__(self, device: str|None=None, lite_cells:int=10, memory_slots:int=32, **core_kwargs):
        self.geo = RONGeodesicSymphonyCortex(device=device, lite_cells=lite_cells, **core_kwargs)
        self.device = self.geo.device
        self.memory = RONGeodesicMemoryBank(slots=memory_slots, device=self.device)

    def _triplet_axes(self, x:int, y:int, z:int):
        triplet = torch.tensor([[x,y,z]], device=self.device, dtype=torch.long)
        axes, energy = self.geo.stable.integrated.deep.core.token_axes(triplet)
        return axes, energy

    def infer_triplet(self, x:int, y:int, z:int, mask:int=0, adaptive_cycles:bool=True):
        packet0, stable_trace, geo_trace, summary0 = self.geo.infer_triplet(x,y,z,mask,adaptive_cycles)
        axes, energy = self._triplet_axes(x,y,z)
        pressure = geodesic_pressure(axes, energy, strength=0.24, cube_n=3)

        read = self.memory.read(axes[:,1])
        mem_axis = read.axis
        mem_conf = read.confidence.view(-1,1)
        corrected_axes = pressure.corrected_axes.clone()
        if read.found:
            corrected_axes[:,1] = corrected_axes[:,1] + 0.18 * mem_conf * mem_axis
            corrected_axes[:,1] = corrected_axes[:,1] / corrected_axes[:,1].norm(dim=-1, keepdim=True).clamp_min(1e-6)

        with torch.no_grad():
            out = self.geo.stable.integrated.deep.core(
                axes=corrected_axes,
                mask=torch.tensor([mask], device=self.device, dtype=torch.long),
                adaptive_cycles=adaptive_cycles,
            )
            fields = self.geo.stable.integrated.deep.cortex.fields_from_output(out)
            packet1 = self.geo.stable.integrated.deep.cortex.packet_from_output(out, fields)

        changed = (
            packet0.action_face != packet1.action_face
            or abs(packet0.release - packet1.release) > 1e-5
            or abs(packet0.residual_cost - packet1.residual_cost) > 1e-5
        )

        write_strength = torch.tensor([max(packet1.memory_write_strength, float(pressure.pressure_strength.detach().cpu()) * 0.35)], device=self.device)
        self.memory.write(axes[:,1], pressure.pressure_axis, write_strength)

        packet1.diagnostics = dict(packet1.diagnostics)
        packet1.diagnostics.update({
            "active_geodesic_pressure_strength": float(pressure.pressure_strength.detach().cpu()),
            "active_geodesic_cost": float(pressure.geodesic_cost.mean().detach().cpu()),
            "active_transport_cost": float(pressure.transport_cost.mean().detach().cpu()),
            "active_phi_flux": float(pressure.phi_flux.mean().detach().cpu()),
            "active_quaternion_order": float(pressure.quaternion_order.mean().detach().cpu()),
            "active_phase_order": float(pressure.phase_order.mean().detach().cpu()),
            "geodesic_memory_confidence": float(read.confidence.mean().detach().cpu()),
            "geodesic_memory_used_slots": self.memory.summary()["used_slots"],
            "active_changed_packet": bool(changed),
        })

        trace = [
            ActiveGeodesicTrace(
                "geodesic_symphony_diagnostic",
                float(packet0.release),
                float(packet0.residual_cost),
                float(summary0.get("geodesic_cost", 0.0)),
                float(summary0.get("transport_composition_cost", 0.0)),
                float(summary0.get("phi_flux", 0.0)),
                0.0,
                float(read.confidence.mean().detach().cpu()),
                False,
            ),
            ActiveGeodesicTrace(
                "active_geodesic_conductor",
                float(packet1.release),
                float(packet1.residual_cost),
                float(pressure.geodesic_cost.mean().detach().cpu()),
                float(pressure.transport_cost.mean().detach().cpu()),
                float(pressure.phi_flux.mean().detach().cpu()),
                float(pressure.pressure_strength.detach().cpu()),
                float(read.confidence.mean().detach().cpu()),
                bool(changed),
            ),
        ]
        summary = dict(summary0)
        summary.update({
            "active_changed_packet": bool(changed),
            "active_pressure_strength": float(pressure.pressure_strength.detach().cpu()),
            "active_geodesic_cost": float(pressure.geodesic_cost.mean().detach().cpu()),
            "active_transport_cost": float(pressure.transport_cost.mean().detach().cpu()),
            "active_phi_flux": float(pressure.phi_flux.mean().detach().cpu()),
            "active_quaternion_order": float(pressure.quaternion_order.mean().detach().cpu()),
            "memory_summary": self.memory.summary(),
            "memory_read_confidence": float(read.confidence.mean().detach().cpu()),
        })
        return packet1, trace, summary
