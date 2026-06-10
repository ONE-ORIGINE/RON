from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.runtime.ron_system import RONSystem
from ron.core.geodesic_pressure import geodesic_pressure

@dataclass
class RONPureDynamicsOutput:
    packet: object
    trace: list
    summary: dict
    pressure_summary: dict

class RONPureDynamicsNet:
    """Concrete RON pure network for X/Y/Z dynamics.

    Input is already RON-native axes/energy. No classical codec is used.
    """
    def __init__(self, device: str | None = None, **kwargs):
        self.system = RONSystem(
            mode="ron_pure",
            profile="active_geodesic",
            device=device,
            **kwargs,
        )
        self.device = self.system.engine.device if hasattr(self.system.engine, "device") else torch.device(device or "cpu")

    def infer_axes(self, axes: Tensor, energy: Tensor, mask: Tensor | None = None) -> RONPureDynamicsOutput:
        axes = axes.to(self.device)
        energy = energy.to(self.device)
        if mask is None:
            mask = torch.zeros(axes.shape[0], device=self.device, dtype=torch.long)
        pressure = geodesic_pressure(axes, energy, strength=0.24, cube_n=3)
        # Use core directly for batchable path; packet path remains single example for official output.
        out = self.system.engine.geo.stable.integrated.deep.core(
            axes=pressure.corrected_axes,
            mask=mask.to(self.device),
            adaptive_cycles=True,
        )
        fields = self.system.engine.geo.stable.integrated.deep.cortex.fields_from_output(out)
        packet = self.system.engine.geo.stable.integrated.deep.cortex.packet_from_output(out, fields)
        summary = {
            "profile": "active_geodesic",
            "mode": "ron_pure",
            "active_pressure_strength": float(pressure.pressure_strength.detach().cpu()),
            "geodesic_cost": float(pressure.geodesic_cost.mean().detach().cpu()),
            "transport_cost": float(pressure.transport_cost.mean().detach().cpu()),
            "phi_flux": float(pressure.phi_flux.mean().detach().cpu()),
            "quaternion_order": float(pressure.quaternion_order.mean().detach().cpu()),
        }
        return RONPureDynamicsOutput(packet, [], summary, summary)
