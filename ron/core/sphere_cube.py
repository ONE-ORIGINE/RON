from __future__ import annotations
import torch
from torch import Tensor

def soft_linf(x: Tensor, beta: float=16.0, dim: int=-1) -> Tensor:
    """Smooth L-infinity norm used as cube gauge."""
    ax = x.abs()
    return (torch.logsumexp(beta * ax, dim=dim) / beta).clamp_min(1e-6)

def phi_sphere_cube(x: Tensor, beta: float=16.0, enabled: bool=True) -> Tensor:
    """Sphere-cube gauge.

    It is not only a feature normalization: V14 uses its barrier and pullback proxy
    in residual dynamics, port gating, release, and energy economy.
    """
    if not enabled:
        return x
    l2 = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    li = soft_linf(x, beta=beta, dim=-1).unsqueeze(-1)
    return (l2 / li) * x

def cube_barrier(x: Tensor, limit: float=2.20, beta: float=10.0) -> Tensor:
    """Smooth penalty for leaving the usable cube region."""
    li = soft_linf(x, beta=beta, dim=-1)
    excess = torch.nn.functional.softplus(li - limit)
    return excess.pow(2).mean()

def cube_barrier_per_ron(x: Tensor, limit: float=2.20, beta: float=10.0) -> Tensor:
    """Per-sample/per-RON cube boundary pressure."""
    li = soft_linf(x, beta=beta, dim=-1)
    return torch.nn.functional.softplus(li - limit).pow(2)

def cube_boundary_axis(x: Tensor, beta: float=16.0) -> Tensor:
    """A differentiable proxy for the active cube face direction.

    This approximates the Clarke subgradient of ||x||_infty:
    high-magnitude coordinates receive higher responsibility.
    """
    weight = torch.softmax(beta * x.abs(), dim=-1) * x.sign()
    return weight

def phi_pullback_proxy(x: Tensor, residual: Tensor, beta: float=16.0) -> Tensor:
    """A practical D Phi^* proxy.

    It keeps the residual in the tangent direction of the sphere-cube gauge while
    adding a cube-boundary correction. This is intentionally explicit and local.
    """
    if x.shape[-1] != residual.shape[-1]:
        k = min(x.shape[-1], residual.shape[-1])
        x0 = x[..., :k]
        r0 = residual[..., :k]
    else:
        x0, r0 = x, residual
    l2 = x0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    li = soft_linf(x0, beta=beta, dim=-1).unsqueeze(-1)
    face = cube_boundary_axis(x0, beta=beta)
    radial = x0 / l2
    tangent = r0 - (r0 * radial).sum(-1, keepdim=True) * radial
    boundary = (r0 * face).sum(-1, keepdim=True) * face
    return (l2 / li) * tangent + 0.35 * boundary

def prox_cube(x: Tensor, limit: float=2.20) -> Tensor:
    return x.clamp(-limit, limit)
