from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .quaternion import qmul, qconj, qnorm, qexp

def qinv(q: Tensor) -> Tensor:
    """Inverse for unit quaternions."""
    return qconj(qnorm(q))

def qlog(q: Tensor) -> Tensor:
    """Quaternion logarithm to tangent vector.

    For q = [cos(theta/2), u sin(theta/2)], returns theta*u.
    This is the inverse of qexp used in this project.
    """
    q = qnorm(q)
    w = q[..., :1].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    v = q[..., 1:]
    vnorm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half_theta = torch.atan2(vnorm, w)
    return 2.0 * half_theta * v / vnorm

def axis_to_quat(axis: Tensor) -> Tensor:
    """Quaternion rotating +X to `axis`.

    This converts a RON axis representation into an orientation q.
    It is intentionally local and differentiable except at the exact antipode,
    where a stable fallback axis is used.
    """
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x = torch.zeros(axis.shape[:-1] + (3,), device=axis.device, dtype=axis.dtype)
    x[..., 0] = 1.0
    dot = (x * axis).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.cross(x, axis, dim=-1)
    # If axis ~= -X, choose a 180° rotation around Y.
    anti = dot.squeeze(-1) < -0.999
    q = torch.cat([1.0 + dot, cross], dim=-1)
    fallback = torch.zeros_like(q)
    fallback[..., 2] = 1.0  # [0,0,1,0] = pi around Y
    q = torch.where(anti.unsqueeze(-1), fallback, q)
    return qnorm(q)

@dataclass
class GeodesicResidualReport:
    q_pred: Tensor
    delta: Tensor
    cost: Tensor
    transport_delta: Tensor
    transport_cost: Tensor

def predict_future_orientation(qx: Tensor, qy: Tensor, omega_y: Tensor, dt: float = 1.0, alpha: float = 0.35) -> Tensor:
    """Predict qZ from qX, qY and angular velocity at Y.

    qZ_pred = qY ⊗ exp(dt*omegaY + alpha*log(qX^-1 ⊗ qY))
    """
    rel_xy = qmul(qinv(qx), qy)
    memory = qlog(rel_xy)
    tangent = dt * omega_y + alpha * memory
    return qnorm(qmul(qy, qexp(tangent)))

def orientation_residual(qx: Tensor, qy: Tensor, qz: Tensor, omega_y: Tensor | None = None,
                         dt: float = 1.0, alpha: float = 0.35) -> tuple[Tensor, Tensor, Tensor]:
    if omega_y is None:
        omega_y = qlog(qmul(qinv(qx), qy))
    q_pred = predict_future_orientation(qx, qy, omega_y, dt=dt, alpha=alpha)
    delta = qlog(qmul(qinv(q_pred), qz))
    cost = delta.pow(2).sum(-1)
    return q_pred, delta, cost

def transport_quat(qa: Tensor, qb: Tensor) -> Tensor:
    """Relative transport from orientation qa to qb."""
    return qnorm(qmul(qinv(qa), qb))

def transport_composition_residual(qx: Tensor, qy: Tensor, qz: Tensor) -> tuple[Tensor, Tensor]:
    """Group-correct transport composition residual.

    Direct transport X→Z should match composed transport X→Y→Z:
      ΔT = log( T_XZ^-1 ∘ T_YZ ∘ T_XY )
    """
    txy = transport_quat(qx, qy)
    tyz = transport_quat(qy, qz)
    txz = transport_quat(qx, qz)
    delta = qlog(qmul(qinv(txz), qmul(tyz, txy)))
    return delta, delta.pow(2).sum(-1)

def geodesic_triaxial_report(qx: Tensor, qy: Tensor, qz: Tensor, omega_y: Tensor | None = None,
                             dt: float = 1.0, alpha: float = 0.35) -> GeodesicResidualReport:
    q_pred, delta, cost = orientation_residual(qx, qy, qz, omega_y=omega_y, dt=dt, alpha=alpha)
    td, tc = transport_composition_residual(qx, qy, qz)
    return GeodesicResidualReport(q_pred, delta, cost, td, tc)
