from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.ports import face_normals

ROLE_INDEX = {
    "sensory": 0,
    "spatial": 1,
    "temporal": 2,
    "causal": 3,
    "memory": 4,
    "value": 5,
    "conductor": 6,
}

@dataclass
class RichSynapseState:
    conductance: Tensor
    trust: Tensor
    trace: Tensor
    delay: Tensor
    homeostasis: Tensor
    causal_sign: Tensor

class RichRONSynapseBank:
    """RON-native multi-channel synapses.

    A synapse is not a scalar weight. It transports oriented port-energy with:
    conductance, trust, trace, delay, homeostasis, resonance and causal sign.
    """
    def __init__(self, edges: list[tuple[str,str,int,int,float]], device=None, dtype=torch.float32):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.edges = edges
        self.E = len(edges)
        self.src_role = [e[0] for e in edges]
        self.dst_role = [e[1] for e in edges]
        self.src_port = torch.tensor([e[2] for e in edges], device=self.device, dtype=torch.long)
        self.dst_port = torch.tensor([e[3] for e in edges], device=self.device, dtype=torch.long)
        base = torch.tensor([e[4] for e in edges], device=self.device, dtype=dtype).clamp_min(0.01)
        sign = torch.ones(self.E, device=self.device, dtype=dtype)
        # A few long range inhibitory/corrective connections are allowed, but encoded explicitly.
        for i, (s,t,sp,tp,g) in enumerate(edges):
            if s == "conductor" and t in ("sensory","spatial","temporal","causal"):
                sign[i] = -1.0 if (sp + tp) % 2 else 1.0
        self.state = RichSynapseState(
            conductance=base,
            trust=torch.full((self.E,), 0.55, device=self.device, dtype=dtype),
            trace=torch.zeros(self.E, device=self.device, dtype=dtype),
            delay=torch.zeros(self.E, device=self.device, dtype=dtype),
            homeostasis=torch.ones(self.E, device=self.device, dtype=dtype),
            causal_sign=sign,
        )

    @staticmethod
    def default_edges() -> list[tuple[str,str,int,int,float]]:
        roles = ["sensory","spatial","temporal","causal","memory","value","conductor"]
        edges=[]
        # local forward backbone
        for i in range(len(roles)-1):
            for p in range(6):
                edges.append((roles[i], roles[i+1], p, p, 0.12 + 0.015*p))
        # recurrent corrections from conductor
        for role in roles[:-1]:
            for p in range(6):
                edges.append(("conductor", role, p, 5-p, 0.055))
        # memory and causal shortcuts
        for p in range(6):
            edges.append(("memory", "causal", p, (p+2)%6, 0.070))
            edges.append(("causal", "value", p, (p+3)%6, 0.075))
            edges.append(("spatial", "memory", p, (p+1)%6, 0.060))
            edges.append(("temporal", "memory", p, (p+4)%6, 0.060))
        return edges

    def transport(self, boundaries: dict[str, Tensor], axes: dict[str, Tensor]|None=None) -> dict[str, Tensor]:
        """Transport port-energy from source boundaries to target roles."""
        if not boundaries:
            return {}
        B = next(iter(boundaries.values())).shape[0]
        out: dict[str, Tensor] = {}
        for role in set(self.dst_role):
            out[role] = torch.zeros(B, 6, device=self.device, dtype=self.dtype)

        n = face_normals(self.device, self.dtype)
        for i, (s,t,sp,tp,_) in enumerate(self.edges):
            if s not in boundaries:
                continue
            flow = boundaries[s][:, sp]
            if axes is not None and s in axes and t in axes:
                # Orientation resonance is cosine transported through face normals.
                rs = (axes[s] * n[sp]).sum(-1).abs()
                rt = (axes[t] * n[tp]).sum(-1).abs()
                resonance = (0.5 + 0.5*rs*rt).clamp(0.0, 1.5)
            else:
                resonance = 1.0
            gate = torch.sigmoid(self.state.trust[i] - self.state.delay[i])
            signed = 1.0 + 0.35*self.state.causal_sign[i]
            amount = flow * self.state.conductance[i] * gate * self.state.homeostasis[i] * resonance * signed
            out[t][:, tp] = out[t][:, tp] + amount.clamp_min(0.0)
        return out

    def plasticity(self, boundaries: dict[str, Tensor], transported: dict[str, Tensor], residual: float, release: float, dissonance: float):
        """RON-native synaptic plasticity.

        Strengthen paths that carry energy while residual/dissonance drop; weaken
        routes that carry energy into high dissonance or premature release.
        """
        with torch.no_grad():
            for i, (s,t,sp,tp,_) in enumerate(self.edges):
                if s not in boundaries or t not in transported:
                    continue
                src = boundaries[s][:, sp].detach().mean()
                dst = transported[t][:, tp].detach().mean()
                coflow = (src * dst).clamp_min(0.0)
                usefulness = coflow * (1.0 + release) / (1.0 + residual + dissonance)
                penalty = coflow * (residual + 0.5*dissonance) / (1.0 + release)
                delta = 0.010*usefulness - 0.006*penalty
                self.state.trace[i].mul_(0.92).add_(coflow)
                self.state.conductance[i].add_(delta).clamp_(0.005, 1.25)
                self.state.trust[i].add_(0.006*(usefulness-penalty)).clamp_(0.05, 1.50)
                # Delay rises for noisy, low-trust paths and falls when useful.
                self.state.delay[i].add_(0.003*(penalty-usefulness)).clamp_(0.0, 1.0)
                # Homeostasis prevents one edge from monopolizing all flow.
                self.state.homeostasis[i] = (1.0 / (1.0 + 0.04*self.state.trace[i])).clamp(0.25, 1.0)

    def summary(self) -> dict:
        return {
            "edge_count": self.E,
            "conductance_mean": float(self.state.conductance.mean().detach().cpu()),
            "trust_mean": float(self.state.trust.mean().detach().cpu()),
            "trace_mean": float(self.state.trace.mean().detach().cpu()),
            "delay_mean": float(self.state.delay.mean().detach().cpu()),
            "homeostasis_mean": float(self.state.homeostasis.mean().detach().cpu()),
            "has_inhibitory_corrections": bool((self.state.causal_sign < 0).any().detach().cpu()),
        }
