from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .quaternion import qexp, qmul, qnorm, qrotate
from .ports import face_normals
from .symphony import entropy6
@dataclass
class RONLiteState:
    q: Tensor; omega: Tensor; energy: Tensor; trace: Tensor; ports: Tensor; axis: Tensor; release: Tensor; residual: Tensor
def _unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
class RONLiteBank(nn.Module):
    """Reduced RON population: q, omega, energy, trace, ports, release. No dense head."""
    def __init__(self, cells:int=16, trace_dim:int=8, seed:int=123):
        super().__init__(); self.cells=cells; self.trace_dim=trace_dim
        self.drive_gain=nn.Parameter(torch.tensor(0.64)); self.memory_gain=nn.Parameter(torch.tensor(0.35))
        self.route_gain=nn.Parameter(torch.tensor(0.70)); self.release_threshold=nn.Parameter(torch.tensor(0.30)); self.energy_decay=nn.Parameter(torch.tensor(0.08))
        g=torch.Generator().manual_seed(seed); pref=torch.randn(cells,3,generator=g); pref=pref/pref.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        self.register_buffer('preferred_axis',pref,persistent=False)
    def initial_state(self,batch:int,device=None,dtype=None):
        device=device or self.preferred_axis.device; dtype=dtype or self.preferred_axis.dtype
        q=torch.zeros(batch,self.cells,4,device=device,dtype=dtype); q[...,0]=1.0
        omega=torch.zeros(batch,self.cells,3,device=device,dtype=dtype); energy=torch.full((batch,self.cells),0.20,device=device,dtype=dtype)
        trace=torch.zeros(batch,self.cells,self.trace_dim,device=device,dtype=dtype); ports=torch.ones(batch,self.cells,6,device=device,dtype=dtype)/6.0
        axis=self.preferred_axis.to(device=device,dtype=dtype).unsqueeze(0).expand(batch,-1,-1)
        release=torch.zeros(batch,self.cells,device=device,dtype=dtype); residual=torch.zeros(batch,self.cells,device=device,dtype=dtype)
        return RONLiteState(q,omega,energy,trace,ports,axis,release,residual)
    def step(self,input_axis:Tensor,input_energy:Tensor|None=None,state:RONLiteState|None=None,steps:int=2):
        if input_axis.dim()==2:
            B=input_axis.shape[0]; axis_in=input_axis[:,None,:].expand(B,self.cells,3)
        else:
            B=input_axis.shape[0]; axis_in=input_axis
            if axis_in.shape[1]!=self.cells: axis_in=axis_in.mean(1,keepdim=True).expand(B,self.cells,3)
        axis_in=_unit(axis_in)
        if input_energy is None: e_in=torch.ones(B,self.cells,device=axis_in.device,dtype=axis_in.dtype)*0.5
        elif input_energy.dim()==1: e_in=input_energy[:,None].expand(B,self.cells)
        else:
            e_in=input_energy
            if e_in.shape[1]!=self.cells: e_in=e_in.mean(1,keepdim=True).expand(B,self.cells)
        st=self.initial_state(B,axis_in.device,axis_in.dtype) if state is None else state
        pref=self.preferred_axis.to(axis_in.device,axis_in.dtype).unsqueeze(0); q,omega,energy,trace=st.q,st.omega,st.energy,st.trace
        for _ in range(max(1,steps)):
            current=qrotate(q, torch.tensor([1.,0.,0.],device=axis_in.device,dtype=axis_in.dtype)); residual_axis=axis_in-current; residual=residual_axis.norm(dim=-1)
            drive=self.drive_gain.abs()*residual_axis+0.22*torch.cross(current,axis_in,dim=-1)+0.12*pref
            omega=(0.82*omega+drive).clamp(-2.0,2.0); q=qnorm(qmul(q,qexp(omega*(0.50+0.25*e_in).unsqueeze(-1))))
            energy=(1.0-self.energy_decay.sigmoid()*0.18)*energy+0.25*e_in+0.10*residual
            axis=qrotate(q, torch.tensor([1.,0.,0.],device=axis_in.device,dtype=axis_in.dtype))
            tri=torch.cat([axis,omega,energy.unsqueeze(-1),residual.unsqueeze(-1)],dim=-1)
            tri=tri[...,:self.trace_dim] if tri.shape[-1]>=self.trace_dim else torch.nn.functional.pad(tri,(0,self.trace_dim-tri.shape[-1]))
            trace=0.86*torch.roll(trace,shifts=1,dims=-1)+self.memory_gain.abs()*tri
        n=face_normals(axis_in.device,axis_in.dtype); scores=self.route_gain.abs()*torch.einsum('bcd,fd->bcf',axis,n)+0.15*energy.unsqueeze(-1)
        ports=scores.softmax(-1); portH=entropy6(ports.mean(1)).view(B,1); release=torch.sigmoid(energy-self.release_threshold.abs()-0.25*residual+0.15*portH)
        return RONLiteState(q,omega,energy.clamp_min(0.0),trace,ports,axis,release,residual)
    def boundary(self,state:RONLiteState)->Tensor:
        return (state.ports*state.energy.unsqueeze(-1)*state.release.unsqueeze(-1)).sum(1)
    def axis(self,state:RONLiteState)->Tensor:
        n=face_normals(state.axis.device,state.axis.dtype); p=self.boundary(state).softmax(-1); a=p@n
        return a/a.norm(dim=-1,keepdim=True).clamp_min(1e-6)
