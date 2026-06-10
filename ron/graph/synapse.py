from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from ron.core.ports import face_normals
from .topology import RONTopology, build_default_topology
from .resonance import port_resonance

@dataclass
class SynapseDiagnostics:
    total_flow: Tensor
    mean_resonance: Tensor
    mean_trust: Tensor
    plastic_delta: Tensor
    active_edges: Tensor

class RONSynapseBank(nn.Module):
    """Port-to-port synapse bank for a RON cortex.

    A synapse transports oriented energy, not a scalar activation. It has:
    conductance, trust, trace, source port, target port and resonance gating.
    """
    def __init__(self, topology: RONTopology|None=None, device=None):
        super().__init__()
        topo = topology or build_default_topology(device=device)
        self.roles = topo.roles
        self.register_buffer('src', topo.src.clone(), persistent=False)
        self.register_buffer('dst', topo.dst.clone(), persistent=False)
        self.register_buffer('src_port', topo.src_port.clone(), persistent=False)
        self.register_buffer('dst_port', topo.dst_port.clone(), persistent=False)
        E = topo.src.numel()
        self.conductance = nn.Parameter(torch.ones(E)*0.42)
        self.trust = nn.Parameter(torch.ones(E)*0.55)
        self.delay_gate = nn.Parameter(torch.zeros(E))
        self.register_buffer('trace', torch.zeros(E), persistent=False)
        self.kind = topo.kind

    @property
    def n_roles(self) -> int:
        return len(self.roles)

    def forward(self, boundaries: Tensor, axes: Tensor, energy: Tensor) -> tuple[Tensor, Tensor, Tensor, SynapseDiagnostics]:
        """Transport between RON populations.

        boundaries: [B,N,6], axes: [B,N,3], energy: [B,N]
        returns incoming_axis [B,N,3], incoming_energy [B,N], edge_flow [B,E]
        """
        B,N,_ = boundaries.shape
        dev,dtype = boundaries.device,boundaries.dtype
        normals = face_normals(dev,dtype)
        src_b = boundaries[:, self.src, :]       # [B,E,6]
        dst_b = boundaries[:, self.dst, :]
        src_a = axes[:, self.src, :]
        dst_a = axes[:, self.dst, :]
        src_e = energy[:, self.src]
        src_port_power = src_b.gather(-1, self.src_port.view(1,-1,1).expand(B,-1,1)).squeeze(-1)
        dst_normal = normals[self.dst_port].view(1,-1,3)
        src_normal = normals[self.src_port].view(1,-1,3)
        resonance = port_resonance(src_b, dst_b, src_a, dst_a)
        gate = torch.sigmoid(self.conductance).view(1,-1) * torch.sigmoid(self.trust).view(1,-1) * torch.sigmoid(1.0+self.delay_gate).view(1,-1)
        flow = (src_port_power.abs() + 0.10*src_e) * resonance * gate
        transported_axis = src_a + 0.38*dst_normal + 0.12*src_normal
        transported_axis = transported_axis / transported_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        incoming_axis_num = torch.zeros(B,N,3,device=dev,dtype=dtype)
        incoming_energy = torch.zeros(B,N,device=dev,dtype=dtype)
        dst_index = self.dst.view(1,-1,1).expand(B,-1,3)
        incoming_axis_num = incoming_axis_num.scatter_add(1, dst_index, transported_axis * flow.unsqueeze(-1))
        incoming_energy = incoming_energy.scatter_add(1, self.dst.view(1,-1).expand(B,-1), flow)
        incoming_axis = incoming_axis_num / incoming_axis_num.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # For isolated roles, preserve current axis rather than NaN/zero.
        incoming_axis = torch.where(incoming_energy.unsqueeze(-1)>1e-6, incoming_axis, axes)
        diag = SynapseDiagnostics(
            total_flow=flow.mean(),
            mean_resonance=resonance.mean(),
            mean_trust=torch.sigmoid(self.trust).mean(),
            plastic_delta=torch.zeros((),device=dev,dtype=dtype),
            active_edges=(flow>1e-5).float().mean(),
        )
        return incoming_axis, incoming_energy, flow, diag

    def plasticity_update_(self, flow: Tensor, residual_before: Tensor|float, residual_after: Tensor|float, usefulness: Tensor|float):
        """Local synaptic plasticity.

        Reinforce edges when transport reduces residual or improves usefulness;
        damp edges that carry energy without improving the state.
        """
        with torch.no_grad():
            rb = residual_before if torch.is_tensor(residual_before) else torch.tensor(float(residual_before),device=self.conductance.device)
            ra = residual_after if torch.is_tensor(residual_after) else torch.tensor(float(residual_after),device=self.conductance.device)
            u = usefulness if torch.is_tensor(usefulness) else torch.tensor(float(usefulness),device=self.conductance.device)
            improvement = torch.tanh(rb - ra + 0.25*u).detach()
            edge_flow = flow.detach().mean(0)
            delta = 0.012 * improvement * edge_flow / edge_flow.mean().clamp_min(1e-6)
            self.conductance.add_(delta).clamp_(-4.0,4.0)
            self.trust.add_(0.006*delta.sign()).clamp_(-4.0,4.0)
            self.trace.mul_(0.96).add_(edge_flow.clamp(0,10).to(self.trace.device)*0.04)
            return delta.abs().mean()
