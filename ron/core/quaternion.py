from __future__ import annotations
import torch
from torch import Tensor

def qnorm(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

def qmul(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)

def qconj(q: Tensor) -> Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def qexp(v: Tensor) -> Tensor:
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half = 0.5 * theta
    return qnorm(torch.cat([torch.cos(half), torch.sin(half) * v / theta], dim=-1))

def qrotate(q: Tensor, v: Tensor) -> Tensor:
    # rotate vector v by unit quaternion q, with broadcastable leading dimensions
    lead = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    q_b = q.expand(lead + (4,))
    v_b = v.expand(lead + (3,))
    qv = torch.cat([torch.zeros(lead + (1,), device=q.device, dtype=q.dtype), v_b.to(device=q.device, dtype=q.dtype)], dim=-1)
    return qmul(qmul(q_b, qv), qconj(q_b))[..., 1:]
