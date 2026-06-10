from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class PhiFluxReport:
    sphere_flux: Tensor
    cube_flux: Tensor
    phi_flux: Tensor
    radial_pressure: Tensor
    boundary_pressure: Tensor

def _as_cube5(cube: Tensor) -> Tensor:
    """Return [B,R,N,N,N] from common cube layouts."""
    if cube.dim() == 4:
        return cube.unsqueeze(1)
    if cube.dim() != 5:
        raise ValueError("cube must be [B,N,N,N], [B,R,N,N,N] or [B,N,N,N,C]")
    # If last dim looks like channels and second dim looks spatial, merge channel into R.
    if cube.shape[-1] <= 8 and cube.shape[1] == cube.shape[2] == cube.shape[3]:
        return cube.permute(0, 4, 1, 2, 3).contiguous()
    return cube

def cube_gradient(cube: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    m = _as_cube5(cube)
    gx = torch.zeros_like(m); gy = torch.zeros_like(m); gz = torch.zeros_like(m)
    gx[..., :, :, 1:-1] = 0.5 * (m[..., :, :, 2:] - m[..., :, :, :-2])
    gx[..., :, :, 0] = m[..., :, :, 1] - m[..., :, :, 0]
    gx[..., :, :, -1] = m[..., :, :, -1] - m[..., :, :, -2]

    gy[..., :, 1:-1, :] = 0.5 * (m[..., :, 2:, :] - m[..., :, :-2, :])
    gy[..., :, 0, :] = m[..., :, 1, :] - m[..., :, 0, :]
    gy[..., :, -1, :] = m[..., :, -1, :] - m[..., :, -2, :]

    gz[..., 1:-1, :, :] = 0.5 * (m[..., 2:, :, :] - m[..., :-2, :, :])
    gz[..., 0, :, :] = m[..., 1, :, :] - m[..., 0, :, :]
    gz[..., -1, :, :] = m[..., -1, :, :] - m[..., -2, :, :]
    return gx, gy, gz

def phi_flux(cube: Tensor, sphere_radius: float = 0.72, shell_width: float = 0.34, cube_lambda: float = 0.65) -> PhiFluxReport:
    """Differentiable-ish finite-difference Phi-flux.

    Sphere flux approximates ∮ <∇M,n_s> dσ on a shell.
    Cube flux approximates ∮ <∇M,n_c> dσ on cube faces.
    Phi is the sphere/cube tension: sphere_flux - λ*cube_flux.
    """
    m = _as_cube5(cube)
    B = m.shape[0]
    device, dtype = m.device, m.dtype
    N = m.shape[-1]
    gx, gy, gz = cube_gradient(m)

    # Grid in [z,y,x] order matching cube memory.
    a = torch.linspace(-1, 1, N, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(a, a, a, indexing="ij")
    r = torch.sqrt(x*x + y*y + z*z).clamp_min(1e-6)
    nx, ny, nz = x/r, y/r, z/r
    shell = torch.exp(-((r - sphere_radius)**2) / (2 * shell_width * shell_width))
    shell = shell / shell.sum().clamp_min(1e-6)

    radial = gx * nx + gy * ny + gz * nz
    sphere_flux = (radial * shell.view(1,1,N,N,N)).sum(dim=(-1,-2,-3)).mean(1)

    # Cube boundary outward normal flux. Use absolute net pressure to avoid orientation cancellation.
    fx = 0.5 * (gx[..., :, :, -1].abs().mean(dim=(-1,-2)) + gx[..., :, :, 0].abs().mean(dim=(-1,-2)))
    fy = 0.5 * (gy[..., :, -1, :].abs().mean(dim=(-1,-2)) + gy[..., :, 0, :].abs().mean(dim=(-1,-2)))
    fz = 0.5 * (gz[..., -1, :, :].abs().mean(dim=(-1,-2)) + gz[..., 0, :, :].abs().mean(dim=(-1,-2)))
    cube_flux = (fx + fy + fz).mean(1) / 3.0

    radial_pressure = radial.abs().mean(dim=(-1,-2,-3)).mean(1)
    boundary_pressure = cube_flux
    phi = sphere_flux - cube_lambda * cube_flux
    return PhiFluxReport(sphere_flux, cube_flux, phi, radial_pressure, boundary_pressure)
