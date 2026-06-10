from __future__ import annotations
import torch
from torch import Tensor

def port_resonance(src_boundary: Tensor, dst_boundary: Tensor, src_axis: Tensor, dst_axis: Tensor) -> Tensor:
    """RON attention redefined as port-to-port resonance.

    High resonance means compatible orientation, compatible port distribution and
    low destructive conflict. This is not dot-product attention over tokens.
    """
    ps=src_boundary.softmax(-1)
    pd=dst_boundary.softmax(-1)
    port_match=(ps*pd).sum(-1)
    axis_match=0.5+0.5*(src_axis*dst_axis).sum(-1).clamp(-1,1)
    sharpness=1.0-(ps-pd).abs().mean(-1).clamp(0,1)
    return (0.45*port_match+0.40*axis_match+0.15*sharpness).clamp(0,1)
