from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .sphere_cube import phi_sphere_cube, cube_barrier, cube_barrier_per_ron, phi_pullback_proxy, cube_boundary_axis

@dataclass
class TriResidual:
    delta: Tensor
    residual: Tensor
    residual_axis: Tensor
    cost: Tensor
    metric: Tensor
    barrier: Tensor
    phi_pressure: Tensor
    boundary_axis: Tensor

def tri_residual(
    X: Tensor, Y: Tensor, Z: Tensor,
    ax: Tensor, ay: Tensor, az: Tensor,
    metric_weight: Tensor, impulse: Tensor,
    use_phi: bool=True, use_metric: bool=True, use_impulse: bool=True, bidirectional: bool=True
) -> TriResidual:
    PX=phi_sphere_cube(X, enabled=use_phi)
    PY=phi_sphere_cube(Y, enabled=use_phi)
    PZ=phi_sphere_cube(Z, enabled=use_phi)

    # Directional transports. Still local, not a dense matrix.
    sx=(ax*ay).sum(-1, keepdim=True)
    sz=(az*ay).sum(-1, keepdim=True)
    cross_x=torch.cross(ax, ay, dim=-1)
    cross_z=torch.cross(az, ay, dim=-1)
    if bidirectional:
        TX=PX*(1+0.18*sx) + 0.05*torch.roll(PX,1,dims=-1)
        TZ=PZ*(1+0.18*sz) - 0.05*torch.roll(PZ,-1,dims=-1)
        # add a small local torsional phase to the first channels
        k=min(3, PX.shape[-1])
        TX = TX.clone(); TZ = TZ.clone()
        TX[...,:k] = TX[...,:k] + 0.06*cross_x[...,:k]
        TZ[...,:k] = TZ[...,:k] + 0.06*cross_z[...,:k]
    else:
        TX=PY*0.5
        TZ=PY*0.5

    delta=2*PY - TX - TZ
    m=torch.nn.functional.softplus(metric_weight) if use_metric else torch.ones_like(metric_weight)
    m=m[:delta.shape[-1]].view(1,1,-1)
    imp=impulse[:delta.shape[-1]].view(1,1,-1) if use_impulse else torch.zeros_like(delta[:,:,:1]).expand_as(delta)
    raw=m*delta + imp

    if use_phi:
        pulled = phi_pullback_proxy(Y, raw)
        # pad back to residual length
        if pulled.shape[-1] < raw.shape[-1]:
            pulled = torch.nn.functional.pad(pulled, (0, raw.shape[-1]-pulled.shape[-1]))
        residual = 0.72*raw + 0.28*pulled
    else:
        residual=raw

    # active residual axis: first-moment residual + noncommutative roll twist + cube-face pressure
    r3=torch.nn.functional.pad(residual, (0,max(0,3-residual.shape[-1])))[...,:3]
    twist=torch.nn.functional.pad(torch.roll(residual,1,-1)-torch.roll(residual,-1,-1),(0,max(0,3-residual.shape[-1])))[...,:3]
    face=torch.nn.functional.pad(cube_boundary_axis(PY), (0,max(0,3-PY.shape[-1])))[...,:3]
    axis=torch.tanh(r3+0.25*twist+0.20*face)

    cost=(residual.pow(2).mean((-1,-2))).sqrt()
    phi_p=cube_barrier_per_ron(PY).mean(-1)
    return TriResidual(delta,residual,axis,cost,m.squeeze(),cube_barrier(PY),phi_p,face)
