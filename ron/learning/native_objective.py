from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class NativeLossParts:
    route: float
    residual: float
    phi: float
    dissonance: float
    entropy: float
    release_balance: float
    geodesic: float
    total: float

def _ce(logits, target):
    return F.cross_entropy(logits, target.long())

def _diag(out, key, default=0.0):
    v = out.diagnostics.get(key, None)
    if v is None:
        return torch.as_tensor(default, device=out.main.device, dtype=out.main.dtype)
    return v if torch.is_tensor(v) else torch.as_tensor(v, device=out.main.device, dtype=out.main.dtype)

def native_training_loss(out, batch, geodesic_report=None):
    """RON-native training objective.

    Uses all RON outputs as port fields. There is no dense classifier head.
    """
    route = (
        0.25*_ce(out.future, batch.future_target)
        + 0.18*_ce(out.past, batch.past_target)
        + 0.18*_ce(out.present, batch.present_target)
        + 0.12*_ce(out.cause, batch.cause_target)
        + 0.16*_ce(out.consequence, batch.consequence_target)
        + 0.11*_ce(out.diagonal, batch.diagonal_target)
    )
    residual = _diag(out, "residual_cost").mean()
    phi = _diag(out, "phi_pressure").abs().mean()
    diss = _diag(out, "dissonance").mean()
    entropy = 1.0 - _diag(out, "port_entropy", 1.0).mean()
    release = _diag(out, "release").mean()
    # Prefer neither total silence nor premature release for the native task.
    release_balance = (release - 0.50).pow(2)
    geo = torch.zeros((), device=out.main.device, dtype=out.main.dtype)
    if geodesic_report is not None:
        geo = torch.log1p(geodesic_report.geodesic_cost).mean() + 0.25*torch.log1p(geodesic_report.transport_cost).mean()
    total = route + 0.08*residual + 0.04*phi + 0.05*diss + 0.04*entropy + 0.05*release_balance + 0.03*geo
    return total, NativeLossParts(
        route=float(route.detach().cpu()),
        residual=float(residual.detach().cpu()),
        phi=float(phi.detach().cpu()),
        dissonance=float(diss.detach().cpu()),
        entropy=float(entropy.detach().cpu()),
        release_balance=float(release_balance.detach().cpu()),
        geodesic=float(geo.detach().cpu()),
        total=float(total.detach().cpu()),
    )
