from __future__ import annotations
import torch
from torch import Tensor

def moment_exponents(D: int) -> list[tuple[int,int,int]]:
    exps=[]
    for d in range(1,D+1):
        for a in range(d+1):
            for b in range(d-a+1):
                c=d-a-b
                exps.append((a,b,c))
    return exps

def cube_moments(cube: Tensor, grid: Tensor, degree: int=3) -> tuple[Tensor, list[tuple[int,int,int]]]:
    # cube [B,R,N,N,N], grid [N^3,3]
    B,R = cube.shape[:2]
    w = cube.reshape(B,R,-1)
    denom = w.abs().sum(-1, keepdim=True).clamp_min(1e-6)
    mass = w / denom
    x,y,z = grid[:,0], grid[:,1], grid[:,2]
    vals=[]; exps=moment_exponents(degree)
    for a,b,c in exps:
        mono = (x**a)*(y**b)*(z**c)
        vals.append((mass*mono.view(1,1,-1)).sum(-1))
    return torch.stack(vals, dim=-1), exps

def active_moment_ops(m: Tensor, exps: list[tuple[int,int,int]]) -> dict[str, Tensor]:
    idx={e:i for i,e in enumerate(exps)}
    def g(e):
        return m[..., idx[e]] if e in idx else torch.zeros_like(m[...,0])
    x,y,z = g((1,0,0)),g((0,1,0)),g((0,0,1))
    xx,yy,zz = g((2,0,0)),g((0,2,0)),g((0,0,2))
    xy,xz,yz = g((1,1,0)),g((1,0,1)),g((0,1,1))
    xyz = g((1,1,1))
    center=torch.stack([x,y,z],-1)
    anis=torch.stack([xx-.5*(yy+zz), yy-.5*(xx+zz), zz-.5*(xx+yy)],-1)
    tors=torch.stack([yz,-xz,xy],-1)
    hand=torch.stack([xyz*y,-xyz*x,xyz*z],-1)
    torque=torch.tanh(0.55*center+0.35*tors+0.22*hand)
    curve=torch.tanh(torch.cross(center,tors,dim=-1)+0.20*anis+0.15*hand)
    stability=torch.exp(-(tors.pow(2).mean(-1)+0.35*anis.pow(2).mean(-1)))
    return {'center':center,'anisotropy':anis,'torsion':tors,'handed':hand,'torque':torque,'curvature':curve,'stability':stability}
