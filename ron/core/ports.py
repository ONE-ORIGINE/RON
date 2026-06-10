from __future__ import annotations
import torch
from torch import Tensor

FACE_NORMALS = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])

def face_normals(device, dtype=torch.float32):
    return FACE_NORMALS.to(device=device, dtype=dtype)

def port_projection(axis: Tensor, energy: Tensor, torque: Tensor, signature_axis: Tensor, harmonic_axis: Tensor, residual_axis: Tensor, bias: Tensor|None=None) -> Tensor:
    # all axes [B,R,3], energy [B,R]
    n=face_normals(axis.device, axis.dtype)
    field = 0.80*axis + 0.65*torque + 0.55*signature_axis + 0.35*harmonic_axis + 1.15*residual_axis
    logits = torch.einsum('brd,fd->brf', field, n) + 0.18*energy.unsqueeze(-1)
    if bias is not None: logits = logits + bias.view(1,1,6)
    gates = torch.softmax(3.0*logits, dim=-1)
    return gates

def boundary_from_ports(gates: Tensor, energy: Tensor, grid_shape:int=2) -> Tensor:
    # A simple boundary: sum all RON face energy; for grid=2, all cells are boundary contributors.
    return (gates * energy.unsqueeze(-1)).sum(1)
