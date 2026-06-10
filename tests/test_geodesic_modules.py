from __future__ import annotations
import torch
from ron.core.geodesic_residual import axis_to_quat, predict_future_orientation, orientation_residual
from ron.core.geodesic_pressure import geodesic_pressure

def test_geodesic_self_consistency():
    axes = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.7, 0.7, 0.0],
        [0.2, 0.9, 0.3],
    ]], dtype=torch.float32)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    q = axis_to_quat(axes)
    omega = axes[:, 1] - axes[:, 0]
    qpred = predict_future_orientation(q[:,0], q[:,1], omega)
    _, _, cost = orientation_residual(q[:,0], q[:,1], qpred, omega)
    assert float(cost.mean()) < 1e-8

def test_geodesic_pressure_changes_axes():
    axes = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.7, 0.7, 0.0],
        [0.2, 0.9, 0.3],
    ]], dtype=torch.float32)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    energy = torch.ones(1, 3)
    p = geodesic_pressure(axes, energy)
    assert torch.isfinite(p.corrected_axes).all()
    assert float((p.corrected_axes - axes).abs().mean()) > 1e-7
