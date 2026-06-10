from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.geodesic_residual import axis_to_quat, qlog, qmul, qinv

@dataclass
class DynamicsSample:
    x_state: Tensor
    y_state: Tensor
    z_state: Tensor
    axes: Tensor
    energy: Tensor
    mask: Tensor
    action_face: Tensor
    cause_face: Tensor
    consequence_face: Tensor
    should_release: Tensor

def _unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def _face_from_vec(v: Tensor) -> Tensor:
    """Map vector direction to six RON faces: +X,-X,+Y,-Y,+Z,-Z."""
    abs_v = v.abs()
    axis = abs_v.argmax(dim=-1)
    sign = v.gather(-1, axis.unsqueeze(-1)).squeeze(-1) >= 0
    # axis 0 -> +X 0, -X 1; axis 1 -> +Y 2, -Y 3; axis 2 -> +Z 4, -Z 5.
    return axis * 2 + (~sign).long()

def make_dynamics_batch(batch_size:int=32, seed:int=0, device=None) -> DynamicsSample:
    """Synthetic RON dynamics batch.

    It creates concrete moving 3D states with X=past, Y=present, Z=future.
    RON sees axes/energy and predicts port-like consequences.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos_x = torch.randn(batch_size, 3, generator=g) * 0.7
    vel = torch.randn(batch_size, 3, generator=g) * 0.25
    accel = torch.randn(batch_size, 3, generator=g) * 0.08
    pos_y = pos_x + vel
    pos_z = pos_y + vel + accel
    # States include position, velocity-like delta, and acceleration proxy.
    x_state = torch.cat([pos_x, vel, accel], dim=-1)
    y_state = torch.cat([pos_y, pos_y-pos_x, accel], dim=-1)
    z_state = torch.cat([pos_z, pos_z-pos_y, accel], dim=-1)

    axes = torch.stack([_unit(pos_x + 0.1), _unit(pos_y + 0.1), _unit(pos_z + 0.1)], dim=1)
    speed = vel.norm(dim=-1)
    curvature = accel.norm(dim=-1)
    energy = torch.stack([
        0.35 + speed,
        0.45 + speed + 0.5*curvature,
        0.40 + speed + curvature,
    ], dim=1).clamp(0.05, 2.0)

    q = axis_to_quat(axes)
    dq = qlog(qmul(qinv(q[:,1]), q[:,2]))
    action_face = _face_from_vec(dq + accel)
    cause_face = _face_from_vec(pos_y - pos_x)
    consequence_face = _face_from_vec(pos_z - pos_y)
    should_release = ((dq.norm(dim=-1) + curvature) < 0.85).long()
    masks = torch.tensor([0,1,2,3,4,5,6], dtype=torch.long)
    mask = masks[torch.arange(batch_size) % len(masks)]

    sample = DynamicsSample(x_state, y_state, z_state, axes, energy, mask, action_face, cause_face, consequence_face, should_release)
    if device is not None:
        sample = DynamicsSample(**{k: v.to(device) for k, v in sample.__dict__.items()})
    return sample

class SyntheticDynamicsDataset(torch.utils.data.Dataset):
    def __init__(self, size:int=1024, seed:int=0):
        self.size = size
        self.seed = seed
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        b = make_dynamics_batch(1, seed=self.seed + idx)
        return {k: v.squeeze(0) for k, v in b.__dict__.items()}
