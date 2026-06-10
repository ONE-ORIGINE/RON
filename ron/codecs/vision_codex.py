from __future__ import annotations
import torch
from torch import Tensor

def _unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

class RONVisionCodex:
    """Deterministic RON-native vision codex.

    It does not solve the task. It converts gradients, depth and apparent motion into
    three geometric axes: past, present, future.
    """
    def __init__(self, max_events:int=16):
        self.max_events=max_events

    def image_axes(self, image: Tensor, depth: Tensor|None=None, prev_image: Tensor|None=None) -> tuple[Tensor, Tensor]:
        if image.dim()==3:
            image=image.unsqueeze(1)
        img=image.mean(1,keepdim=True)
        gx=torch.zeros_like(img); gy=torch.zeros_like(img)
        gx[:,:,:,1:-1]=0.5*(img[:,:,:,2:]-img[:,:,:,:-2])
        gy[:,:,1:-1,:]=0.5*(img[:,:,2:,:]-img[:,:,:-2,:])
        lap=torch.zeros_like(img)
        lap[:,:,1:-1,1:-1]=img[:,:,1:-1,2:]+img[:,:,1:-1,:-2]+img[:,:,2:,1:-1]+img[:,:,:-2,1:-1]-4*img[:,:,1:-1,1:-1]
        edge=(gx.pow(2)+gy.pow(2)).sqrt()
        corner=(gx.abs()*gy.abs()).sqrt()

        if depth is None:
            dz=torch.zeros_like(img)
        else:
            dz=depth if depth.dim()==4 else depth.unsqueeze(1)
        if prev_image is None:
            mv=torch.zeros_like(img)
        else:
            ref=prev_image.mean(1,keepdim=True) if prev_image.dim()==4 and prev_image.shape[1]>1 else prev_image
            if ref.dim()==3:
                ref=ref.unsqueeze(1)
            mv=img-ref

        def weighted_axis(a,b,c,weight):
            W=weight.flatten(1).clamp_min(1e-6)
            v=torch.stack([(a.flatten(1)*W).sum(1),(b.flatten(1)*W).sum(1),(c.flatten(1)*W).sum(1)],-1)
            return _unit(torch.tanh(v/(W.sum(1,keepdim=True)+1e-6)))

        present=weighted_axis(gx, gy, 0.35*lap+dz, edge+0.35*corner+0.05)
        past=weighted_axis(-gx-0.25*mv, -gy-0.25*mv, 0.50*mv.abs()-0.15*dz, edge+mv.abs()+0.05)
        future=weighted_axis(gx+0.45*mv, gy+0.45*mv, dz+0.30*mv+0.20*lap, edge+dz.abs()+mv.abs()+0.05)
        axes=torch.stack([past,present,future],1)

        energy=torch.stack([
            edge.mean((1,2,3))+mv.abs().mean((1,2,3))+0.2,
            img.abs().mean((1,2,3))+edge.mean((1,2,3))+0.2,
            (edge+dz.abs()+mv.abs()+corner).mean((1,2,3))+0.2
        ],1).clamp(0.05, 2.5)
        return axes, energy
