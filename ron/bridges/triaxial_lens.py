from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.geodesic_residual import axis_to_quat, geodesic_triaxial_report, transport_composition_residual

MODES = ("XY_PRED_Z","YZ_PRED_X","XZ_PRED_Y","X_PRED_YZ","Z_PRED_XY","Y_PRED_XZ")

@dataclass
class TriaxialLensReport:
    mode_costs: dict[str, float]
    coherence_gap: float
    recommended_mode: str
    missing_axis_pressure: dict[str, float]
    geodesic_cost: float
    transport_cost: float

class RONTriaxialLens:
    """RON-native triaxial lens.

    This absorbs the useful idea of triaxial masking without importing classical
    GRU/attention modules. It measures X/Y/Z coherence from RON axes/quaternions.
    """
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha

    def __call__(self, axes_xyz: Tensor, omega_y: Tensor | None = None) -> TriaxialLensReport:
        """axes_xyz: [B,3,3] for X,Y,Z axes."""
        if axes_xyz.dim() != 3 or axes_xyz.shape[1] != 3 or axes_xyz.shape[-1] != 3:
            raise ValueError("axes_xyz must be [B,3,3]")
        q = axis_to_quat(axes_xyz)
        qx, qy, qz = q[:,0], q[:,1], q[:,2]
        if omega_y is None:
            omega_y = axes_xyz[:,1] - axes_xyz[:,0]
        geo = geodesic_triaxial_report(qx, qy, qz, omega_y=omega_y, alpha=self.alpha)
        td, tc = transport_composition_residual(qx, qy, qz)

        # Axis-specific pressures from simple leave-one-axis tensions.
        x_gap = (axes_xyz[:,0] - (2*axes_xyz[:,1] - axes_xyz[:,2])).norm(dim=-1)
        y_gap = (axes_xyz[:,1] - 0.5*(axes_xyz[:,0] + axes_xyz[:,2])).norm(dim=-1)
        z_gap = (axes_xyz[:,2] - (2*axes_xyz[:,1] - axes_xyz[:,0])).norm(dim=-1)

        geoc = geo.cost.mean()
        transc = tc.mean()
        mode_costs_t = {
            "XY_PRED_Z": z_gap.mean() + geoc,
            "YZ_PRED_X": x_gap.mean() + geoc,
            "XZ_PRED_Y": y_gap.mean() + transc,
            "X_PRED_YZ": (y_gap + z_gap).mean() + geoc,
            "Z_PRED_XY": (x_gap + y_gap).mean() + geoc,
            "Y_PRED_XZ": (x_gap + z_gap).mean() + transc,
        }
        mode_costs = {k: float(v.detach().cpu()) for k,v in mode_costs_t.items()}
        recommended = min(mode_costs, key=mode_costs.get)
        missing = {
            "X": float(x_gap.mean().detach().cpu()),
            "Y": float(y_gap.mean().detach().cpu()),
            "Z": float(z_gap.mean().detach().cpu()),
        }
        return TriaxialLensReport(
            mode_costs=mode_costs,
            coherence_gap=float((x_gap+y_gap+z_gap).mean().detach().cpu()/3.0),
            recommended_mode=recommended,
            missing_axis_pressure=missing,
            geodesic_cost=float(geoc.detach().cpu()),
            transport_cost=float(transc.detach().cpu()),
        )
