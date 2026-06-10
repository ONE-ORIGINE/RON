from __future__ import annotations
import torch
from torch import Tensor
from .quaternion import qrotate

def cube_grid(n: int, device=None, dtype=torch.float32) -> Tensor:
    a = torch.linspace(-1, 1, n, device=device, dtype=dtype)
    z,y,x = torch.meshgrid(a,a,a, indexing='ij')
    return torch.stack([x,y,z], dim=-1).reshape(-1,3)

def default_markers(device=None, dtype=torch.float32) -> Tensor:
    return torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]], device=device, dtype=dtype)

def write_oriented_cube(q: Tensor, energy: Tensor, n: int=5, sigma: float=0.38, decay: float=0.0) -> tuple[Tensor, Tensor]:
    # q: [B,R,4], energy: [B,R]
    B,R = energy.shape
    grid = cube_grid(n, q.device, q.dtype)
    markers = default_markers(q.device, q.dtype)
    rot = qrotate(q.unsqueeze(-2), markers.view(1,1,6,3)) # [B,R,6,3]
    diff = grid.view(1,1,1,-1,3) - rot.unsqueeze(-2)
    kern = torch.exp(-diff.pow(2).sum(-1)/(2*sigma*sigma))
    port_weight = torch.linspace(0.75,1.25,6,device=q.device,dtype=q.dtype).view(1,1,6,1)
    cube = (energy.view(B,R,1,1) * port_weight * kern).sum(2)
    return cube.view(B,R,n,n,n), grid
