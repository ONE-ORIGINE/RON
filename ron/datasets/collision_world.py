from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .synthetic_dynamics import DynamicsSample, _unit, _face_from_vec

@dataclass
class CollisionWorldConfig:
    batch_size: int = 32
    seed: int = 0
    radius: float = 0.22
    dt: float = 1.0

def make_collision_batch(cfg: CollisionWorldConfig, device=None) -> DynamicsSample:
    """Two-body collision-like synthetic world.

    This is a CPU-safe concrete proxy for CLEVRER/Physion style reasoning:
    object A and B move; if they approach within a radius, the future velocity reflects.
    """
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    B = cfg.batch_size
    a0 = torch.randn(B, 3, generator=g) * 0.6
    b0 = torch.randn(B, 3, generator=g) * 0.6
    va = torch.randn(B, 3, generator=g) * 0.22
    vb = torch.randn(B, 3, generator=g) * 0.22
    ay = a0 + va * cfg.dt
    by = b0 + vb * cfg.dt
    rel = ay - by
    dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    contact = (dist.squeeze(-1) < cfg.radius).float().view(B,1)
    normal = rel / dist
    # Elastic-ish reflection on contact.
    va_ref = va - 2.0 * (va * normal).sum(-1, keepdim=True) * normal
    va2 = contact * va_ref + (1.0-contact) * va
    az = ay + va2 * cfg.dt
    # RON axis represents object A with context from B.
    x_axis = _unit(a0 - 0.35*b0 + 0.1*va)
    y_axis = _unit(ay - 0.35*by + 0.1*va)
    z_axis = _unit(az - 0.35*(by + vb*cfg.dt) + 0.1*va2)
    axes = torch.stack([x_axis, y_axis, z_axis], dim=1)
    energy = torch.stack([
        0.35 + va.norm(dim=-1),
        0.45 + va.norm(dim=-1) + 0.65*contact.squeeze(-1),
        0.40 + va2.norm(dim=-1) + 0.35*contact.squeeze(-1),
    ], dim=1).clamp(0.05, 2.0)
    x_state = torch.cat([a0, b0, va], dim=-1)
    y_state = torch.cat([ay, by, va], dim=-1)
    z_state = torch.cat([az, by + vb*cfg.dt, va2], dim=-1)
    action_face = _face_from_vec(va2)
    cause_face = _face_from_vec(rel)
    consequence_face = _face_from_vec(az - ay)
    should_release = (1.0 - contact.squeeze(-1)).long()
    masks = torch.tensor([0,1,2,3,4,5,6], dtype=torch.long)
    mask = masks[torch.arange(B) % len(masks)]
    sample = DynamicsSample(x_state, y_state, z_state, axes, energy, mask, action_face, cause_face, consequence_face, should_release)
    if device is not None:
        sample = DynamicsSample(**{k: v.to(device) for k, v in sample.__dict__.items()})
    return sample
