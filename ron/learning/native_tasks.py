from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class NativeBatch:
    triplet: Tensor
    mask: Tensor
    future_target: Tensor
    past_target: Tensor
    present_target: Tensor
    cause_target: Tensor
    consequence_target: Tensor
    diagonal_target: Tensor

    def to(self, device):
        return NativeBatch(**{k: v.to(device) for k, v in self.__dict__.items()})

def _targets(triplet: Tensor) -> dict[str, Tensor]:
    x, y, z = triplet[:,0], triplet[:,1], triplet[:,2]
    return {
        "future_target": (x + y + z) % 6,
        "past_target": (2*x + y) % 6,
        "present_target": (x + 2*y + z) % 6,
        "cause_target": (z - x).remainder(6),
        "consequence_target": (x + z + 2*y).remainder(6),
        "diagonal_target": (x + y - z).remainder(6),
    }

def make_native_batch(batch_size:int=16, vocab:int=12, seed:int=0, device=None) -> NativeBatch:
    """Deterministic synthetic RON-native batch.

    The targets are port routes derived from X/Y/Z arithmetic. This is not a final
    benchmark; it is a minimal differentiable smoke task for RON-native training.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    triplet = torch.randint(0, vocab, (batch_size, 3), generator=g)
    # Mask codes follow existing RON convention: 0 none, 1 X, 2 Y, 3 Z, 4 X+Y, 5 Y+Z, 6 X+Z.
    masks = torch.tensor([0,1,2,3,4,5,6], dtype=torch.long)
    mask = masks[torch.arange(batch_size) % len(masks)]
    data = {"triplet": triplet, "mask": mask}
    data.update(_targets(triplet))
    batch = NativeBatch(**data)
    return batch.to(device) if device is not None else batch
