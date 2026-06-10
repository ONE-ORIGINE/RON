from __future__ import annotations
import torch
from torch import Tensor

def conductor(main: Tensor, residual_cost: Tensor, energy: Tensor, branch_access: Tensor, port_entropy: Tensor, threshold: Tensor) -> dict[str, Tensor]:
    top=main.topk(2,dim=-1).values; margin=top[:,0]-top[:,1]
    harmony=torch.sigmoid(margin + 0.55*port_entropy + 0.35*branch_access.mean(-1) - 0.25*residual_cost)
    dissonance=torch.sigmoid(residual_cost + 0.3*(1-port_entropy) - margin)
    release=torch.sigmoid(1.6*margin + harmony - dissonance - threshold.abs())
    continue_pressure=torch.sigmoid(dissonance + 0.4*residual_cost - 0.6*harmony)
    return {'harmony':harmony,'dissonance':dissonance,'release':release,'continue_pressure':continue_pressure}

def entropy6(p: Tensor) -> Tensor:
    p=p/p.sum(-1,keepdim=True).clamp_min(1e-6)
    return -(p*p.clamp_min(1e-8).log()).sum(-1)/torch.log(torch.tensor(6.0,device=p.device,dtype=p.dtype))
