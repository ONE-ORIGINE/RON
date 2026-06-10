from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .geodesic_residual import geodesic_triaxial_report, axis_to_quat
from .phi_flux import phi_flux
from .quaternion_symphony import quaternion_symphony_step
from .cube_memory import write_oriented_cube

@dataclass
class GeodesicPressureReport:
    corrected_axes: Tensor
    pressure_axis: Tensor
    pressure_strength: Tensor
    geodesic_cost: Tensor
    transport_cost: Tensor
    phi_flux: Tensor
    quaternion_order: Tensor
    phase_order: Tensor

def _unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def geodesic_pressure(
    axes_xyz: Tensor,
    energy_xyz: Tensor,
    strength: float = 0.22,
    alpha: float = 0.35,
    cube_n: int = 3,
) -> GeodesicPressureReport:
    """Active RON pressure from geodesic residual, Phi-flux and symphony.

    axes_xyz: [B,3,3] for X,Y,Z.
    energy_xyz: [B,3].
    Output corrected axes [B,3,3] with a bounded adjustment mainly on Y/Z.
    """
    axes_xyz = _unit(axes_xyz)
    q = axis_to_quat(axes_xyz)
    qx, qy, qz = q[:,0], q[:,1], q[:,2]
    omega_y = axes_xyz[:,1] - axes_xyz[:,0]
    geo = geodesic_triaxial_report(qx, qy, qz, omega_y=omega_y, alpha=alpha)

    cube, _ = write_oriented_cube(q, energy_xyz.clamp_min(0.01), n=cube_n, sigma=0.42)
    pf = phi_flux(cube)

    sym = quaternion_symphony_step(q, omega=torch.stack([omega_y, omega_y, omega_y], dim=1), energy=energy_xyz)

    # Tangent pressure is a combination of geodesic mismatch and symphony consensus.
    # geodesic delta lives in R^3; Phi flux modulates magnitude; symphony pulls toward collective order.
    delta_axis = geo.delta
    if delta_axis.dim() == 1:
        delta_axis = delta_axis.unsqueeze(0)
    phi_gain = torch.tanh(pf.phi_flux).view(-1, 1)
    order_gain = (1.0 - sym.quaternion_order).view(-1, 1)
    pressure_axis = _unit(delta_axis + 0.35 * sym.omega_new[:,1] + 0.15 * phi_gain * axes_xyz[:,1])
    pressure_strength = torch.sigmoid(
        0.20 * geo.cost.mean().view(1)
        + 0.15 * geo.transport_cost.mean().view(1)
        + 0.35 * order_gain.mean().view(1)
        + 0.10 * pf.boundary_pressure.mean().view(1)
    ).view(1,1)

    s = strength * pressure_strength
    corrected = axes_xyz.clone()
    corrected[:,1] = _unit(axes_xyz[:,1] + s * pressure_axis)
    corrected[:,2] = _unit(axes_xyz[:,2] + 0.65 * s * pressure_axis + 0.25 * s * sym.omega_new[:,2])
    corrected[:,0] = _unit(axes_xyz[:,0] - 0.25 * s * pressure_axis)
    return GeodesicPressureReport(
        corrected_axes=corrected,
        pressure_axis=pressure_axis,
        pressure_strength=pressure_strength.squeeze(),
        geodesic_cost=geo.cost,
        transport_cost=geo.transport_cost,
        phi_flux=pf.phi_flux,
        quaternion_order=sym.quaternion_order,
        phase_order=sym.order_parameter,
    )
