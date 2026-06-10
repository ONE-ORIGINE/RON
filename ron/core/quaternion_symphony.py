from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .quaternion import qmul, qnorm
from .geodesic_residual import qinv, qlog

@dataclass
class SymphonyReport:
    omega_new: Tensor
    phase_new: Tensor
    order_parameter: Tensor
    quaternion_order: Tensor
    coupling_energy: Tensor

def quaternion_order(q: Tensor) -> Tensor:
    """Mean resultant length of quaternion orientations, sign-aligned."""
    q = qnorm(q)
    ref = q[..., :1, :]
    aligned = torch.where((q * ref).sum(-1, keepdim=True) < 0, -q, q)
    mean = aligned.mean(dim=-2)
    return mean.norm(dim=-1).clamp(0.0, 1.0)

def phase_order(theta: Tensor) -> Tensor:
    c = torch.cos(theta).mean(dim=-1)
    s = torch.sin(theta).mean(dim=-1)
    return torch.sqrt(c*c + s*s).clamp(0.0, 1.0)

def quaternion_symphony_step(q: Tensor, omega: Tensor, energy: Tensor | None = None,
                             phase: Tensor | None = None, coupling: float = 0.12,
                             phase_coupling: float = 0.08) -> SymphonyReport:
    """RON-native symphony step.

    Combines:
    - Kuramoto-like scalar release phase;
    - quaternion consensus in tangent space log(q_i^-1 ⊗ q_j).
    """
    q = qnorm(q)
    B, R, _ = q.shape
    device, dtype = q.device, q.dtype
    if energy is None:
        energy = torch.ones(B, R, device=device, dtype=dtype)
    if phase is None:
        phase = torch.atan2(omega.norm(dim=-1), energy.clamp_min(1e-6))

    qi = q.unsqueeze(2)  # [B,R,1,4]
    qj = q.unsqueeze(1)  # [B,1,R,4]
    rel = qmul(qinv(qi), qj)
    tangent = qlog(rel)  # [B,R,R,3]
    e = energy / energy.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    weights = (e.unsqueeze(1) * e.unsqueeze(2)).clamp(0.0, 3.0)
    eye = torch.eye(R, device=device, dtype=dtype).view(1,R,R)
    weights = weights * (1.0 - eye)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    consensus = (weights.unsqueeze(-1) * tangent).sum(dim=2) / denom

    omega_new = omega + coupling * consensus

    # Scalar release phase synchronization.
    diff = phase.unsqueeze(1) - phase.unsqueeze(2)
    phase_drive = (weights * torch.sin(diff)).sum(dim=-1) / denom.squeeze(-1)
    phase_new = phase + phase_coupling * phase_drive

    q_order = quaternion_order(q)
    p_order = phase_order(phase_new)
    coupling_energy = consensus.pow(2).sum(-1).mean(-1)
    return SymphonyReport(omega_new, phase_new, p_order, q_order, coupling_energy)
