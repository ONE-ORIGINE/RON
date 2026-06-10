from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.runtime.stable_integrated_cortex import RONStableIntegratedCortex, StableRONTrace
from ron.core.geodesic_residual import axis_to_quat, geodesic_triaxial_report
from ron.core.cube_memory import write_oriented_cube
from ron.core.phi_flux import phi_flux
from ron.core.quaternion_symphony import quaternion_symphony_step
from ron.bridges.triaxial_lens import RONTriaxialLens

@dataclass
class GeodesicSymphonyTrace:
    stage: str
    geodesic_cost: float
    transport_cost: float
    phi_flux: float
    quaternion_order: float
    phase_order: float
    coherence_gap: float
    recommended_mode: str

class RONGeodesicSymphonyCortex:
    """RON V20 geodesic-symphony cortex.

    Adds the missing mathematical consolidation:
    - geodesic triaxial residual;
    - transport composition residual;
    - Phi flux from cube memory;
    - quaternion symphony consensus;
    - RON-native triaxial lens.
    """
    def __init__(self, device: str|None=None, lite_cells:int=10, **core_kwargs):
        self.stable = RONStableIntegratedCortex(device=device, lite_cells=lite_cells, **core_kwargs)
        self.device = self.stable.device
        self.lens = RONTriaxialLens()

    def _triplet_axes(self, x:int, y:int, z:int) -> tuple[Tensor, Tensor]:
        triplet = torch.tensor([[x,y,z]], device=self.device, dtype=torch.long)
        axes, energy = self.stable.integrated.deep.core.token_axes(triplet)
        return axes, energy

    def infer_triplet(self, x:int, y:int, z:int, mask:int=0, adaptive_cycles:bool=True):
        packet, stable_trace, bus = self.stable.infer_triplet(x,y,z,mask,adaptive_cycles)
        axes, energy = self._triplet_axes(x,y,z)
        q = axis_to_quat(axes)
        qx, qy, qz = q[:,0], q[:,1], q[:,2]
        omega_y = axes[:,1] - axes[:,0]

        geo = geodesic_triaxial_report(qx, qy, qz, omega_y=omega_y)
        cube, _ = write_oriented_cube(q, energy, n=3, sigma=0.42)
        pf = phi_flux(cube)
        sym = quaternion_symphony_step(q, omega=torch.stack([omega_y, omega_y, omega_y], dim=1), energy=energy)
        lens = self.lens(axes, omega_y=omega_y)

        # Enrich packet diagnostics without replacing RONPacket ontology.
        packet.diagnostics = dict(packet.diagnostics)
        packet.diagnostics.update({
            "geodesic_cost": float(geo.cost.mean().detach().cpu()),
            "transport_composition_cost": float(geo.transport_cost.mean().detach().cpu()),
            "phi_flux": float(pf.phi_flux.mean().detach().cpu()),
            "phi_sphere_flux": float(pf.sphere_flux.mean().detach().cpu()),
            "phi_cube_flux": float(pf.cube_flux.mean().detach().cpu()),
            "symphony_quaternion_order": float(sym.quaternion_order.mean().detach().cpu()),
            "symphony_phase_order": float(sym.order_parameter.mean().detach().cpu()),
            "symphony_coupling_energy": float(sym.coupling_energy.mean().detach().cpu()),
            "triaxial_lens_coherence_gap": lens.coherence_gap,
            "triaxial_lens_recommended_mode": lens.recommended_mode,
            "triaxial_lens_mode_costs": lens.mode_costs,
        })
        # Make uncertainty/coherence more honest with these new diagnostics.
        packet.uncertainty = float(max(0.0, min(1.0, packet.uncertainty + 0.10*lens.coherence_gap + 0.05*float(geo.cost.mean().detach().cpu()))))
        trace = [
            GeodesicSymphonyTrace(
                "geodesic_symphony",
                float(geo.cost.mean().detach().cpu()),
                float(geo.transport_cost.mean().detach().cpu()),
                float(pf.phi_flux.mean().detach().cpu()),
                float(sym.quaternion_order.mean().detach().cpu()),
                float(sym.order_parameter.mean().detach().cpu()),
                lens.coherence_gap,
                lens.recommended_mode,
            )
        ]
        summary = dict(bus)
        summary.update({
            "geodesic_cost": float(geo.cost.mean().detach().cpu()),
            "transport_composition_cost": float(geo.transport_cost.mean().detach().cpu()),
            "phi_flux": float(pf.phi_flux.mean().detach().cpu()),
            "symphony_quaternion_order": float(sym.quaternion_order.mean().detach().cpu()),
            "symphony_phase_order": float(sym.order_parameter.mean().detach().cpu()),
            "triaxial_lens_coherence_gap": lens.coherence_gap,
            "triaxial_lens_recommended_mode": lens.recommended_mode,
            "triaxial_lens_mode_count": len(lens.mode_costs),
        })
        return packet, stable_trace, trace, summary
