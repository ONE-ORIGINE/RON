from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class CausalTree:
    cost: Tensor
    access: Tensor
    consequence: Tensor
    filtered: Tensor
    pressure: Tensor

def causal_tree(boundary: Tensor, past: Tensor, future: Tensor, cause: Tensor, consequence: Tensor, energy_cost: Tensor, harmony: Tensor, dissonance: Tensor, gain: Tensor) -> CausalTree:
    conflict=(past.softmax(-1)-future.softmax(-1)).abs()+0.5*(cause.softmax(-1)-consequence.softmax(-1)).abs()
    cost=torch.nn.functional.softplus(gain).view(1,6)*conflict + 0.25*energy_cost.view(-1,1)+0.35*dissonance.view(-1,1)-0.22*harmony.view(-1,1)
    access=torch.sigmoid(boundary + consequence - cost)
    filtered=access*boundary + 0.45*access*consequence - 0.25*cost
    pressure=cost.mean(-1)
    return CausalTree(cost,access,consequence,filtered,pressure)
