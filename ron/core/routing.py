from __future__ import annotations
import torch
from torch import Tensor

def neighbor_index(grid:int, device=None) -> Tensor:
    coords=[]
    for z in range(grid):
        for y in range(grid):
            for x in range(grid):
                coords.append((x,y,z))
    def idx(x,y,z):
        return z*grid*grid+y*grid+x
    dirs=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    n=[]
    for x,y,z in coords:
        row=[]
        for dx,dy,dz in dirs:
            xx,yy,zz=x+dx,y+dy,z+dz
            row.append(idx(xx,yy,zz) if 0<=xx<grid and 0<=yy<grid and 0<=zz<grid else -1)
        n.append(row)
    return torch.tensor(n,device=device,dtype=torch.long)

def route_energy(energy: Tensor, gates: Tensor, neigh: Tensor, gain: Tensor, absorb: float=0.25) -> tuple[Tensor, Tensor]:
    """Vectorized local routing.

    energy: [B,R], gates: [B,R,6], neigh: [R,6].
    Energy exiting outside the grid is accumulated as boundary output.
    """
    B, R = energy.shape
    F = gates.shape[-1]
    gain_abs = gain.abs()
    outgoing = energy.unsqueeze(-1) * gates * gain_abs
    transferable = absorb * outgoing
    spent = transferable.sum(-1).clamp_max(energy)
    new = (energy - spent).clamp_min(0.0)

    neigh_dev = neigh.to(device=energy.device)
    flat_dest = neigh_dev.reshape(-1)  # [R*F]
    flat_val = transferable.reshape(B, R*F)

    inside = flat_dest >= 0
    if bool(inside.any()):
        dest = flat_dest[inside].view(1, -1).expand(B, -1)
        vals = flat_val[:, inside]
        new = new.scatter_add(1, dest, vals)

    boundary = torch.zeros(B, F, device=energy.device, dtype=energy.dtype)
    outside = ~inside
    if bool(outside.any()):
        # outside contribution belongs to its face index
        face_ids = torch.arange(F, device=energy.device).view(1, F).expand(R, F).reshape(-1)
        faces = face_ids[outside].view(1, -1).expand(B, -1)
        vals = flat_val[:, outside]
        boundary = boundary.scatter_add(1, faces, vals)
    return new.clamp_min(0.0), boundary
