from __future__ import annotations
import torch
from torch import Tensor

def unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def real_harmonics(v: Tensor, degree:int=3) -> Tensor:
    u=unit(v); x,y,z=u.unbind(-1); out=[torch.ones_like(x)]
    if degree>=1: out += [x,y,z]
    if degree>=2: out += [x*y,y*z,z*x,x*x-y*y,2*z*z-x*x-y*y]
    if degree>=3: out += [x*y*z, x*(5*z*z-1), y*(5*z*z-1), z*(5*z*z-3), y*(3*x*x-y*y), x*(x*x-3*y*y)]
    return torch.stack(out,-1)

def harmonic_axis(h: Tensor) -> Tensor:
    a=torch.zeros(h.shape[:-1]+(3,), device=h.device, dtype=h.dtype)
    if h.shape[-1]>=4:
        a[...,0]+=h[...,1]; a[...,1]+=h[...,2]; a[...,2]+=h[...,3]
    if h.shape[-1]>=9:
        a[...,0]+=0.18*h[...,6]+0.08*h[...,7]
        a[...,1]+=0.18*h[...,4]-0.08*h[...,7]
        a[...,2]+=0.18*h[...,5]+0.08*h[...,8]
    if h.shape[-1]>=15:
        a[...,0]+=0.05*h[...,10]+0.04*h[...,13]
        a[...,1]+=0.05*h[...,11]+0.04*h[...,14]
        a[...,2]+=0.05*h[...,12]
    return torch.tanh(a)
