from __future__ import annotations
import torch
from torch import Tensor

def signature_size(degree: int) -> int:
    return sum(3**d for d in range(1,degree+1))

def update_signature(sig: Tensor, v: Tensor, degree: int=3, decay: float=0.965) -> Tensor:
    # sig [B,R,S], v [B,R,3]
    if degree <= 0: return sig
    parts=[]
    cur=v
    for d in range(1,degree+1):
        if d==1: cur=v
        else: cur=(parts[-1].unsqueeze(-1)*v.unsqueeze(-2)).reshape(v.shape[:-1]+(3**d,))
        parts.append(cur)
    inc=torch.cat(parts, dim=-1)
    if sig.shape[-1] != inc.shape[-1]:
        sig = torch.zeros(v.shape[:-1]+(inc.shape[-1],), device=v.device, dtype=v.dtype)
    return decay*sig + (1-decay)*inc

def signature_vector3(sig: Tensor, degree: int=3) -> Tensor:
    if sig.numel()==0 or sig.shape[-1] < 3:
        return torch.zeros(sig.shape[:-1]+(3,), device=sig.device, dtype=sig.dtype)
    v=sig[...,:3]
    if sig.shape[-1] >= 12:
        second=sig[...,3:12].reshape(sig.shape[:-1]+(3,3))
        skew=torch.stack([second[...,1,2]-second[...,2,1], second[...,2,0]-second[...,0,2], second[...,0,1]-second[...,1,0]], -1)
        v=v+0.35*skew
    return torch.tanh(v)
