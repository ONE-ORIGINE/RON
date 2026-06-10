from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.ron_lite import RONLiteBank
from ron.core.whole_core import RONWholeTriAxialCore
from ron.core.ports import face_normals
from .cortex import RONCortex
from .packet import RONPacket
def boundary_to_axis(boundary:Tensor)->Tensor:
    n=face_normals(boundary.device,boundary.dtype); p=boundary.softmax(-1); a=p@n
    return a/a.norm(dim=-1,keepdim=True).clamp_min(1e-6)
@dataclass
class DeepRONTrace:
    stage:str; mean_energy:float; mean_release:float; mean_residual:float; boundary:list[float]
class RONDeepCortex:
    """Deep RON cortex: RON-Lite stages feed a full RON conductor through port geometry."""
    STAGES=['sensory','spatial','temporal','causal','memory','value']
    def __init__(self,device:str|None=None,lite_cells:int=12,trace_dim:int=8,**core_kwargs):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lite={role:RONLiteBank(cells=lite_cells,trace_dim=trace_dim,seed=100+i).to(self.device).eval() for i,role in enumerate(self.STAGES)}
        self.core=RONWholeTriAxialCore(**core_kwargs).to(self.device).eval(); self.cortex=RONCortex(self.core,device=str(self.device))
    def triplet_to_axis(self,triplet:Tensor):
        axes,energies=self.core.token_axes(triplet); return axes.mean(1),energies.mean(1)
    def infer_triplet(self,x:int,y:int,z:int,mask:int=0,adaptive_cycles:bool=True):
        triplet=torch.tensor([[x,y,z]],device=self.device,dtype=torch.long); mask_t=torch.tensor([mask],device=self.device,dtype=torch.long)
        axis,energy=self.triplet_to_axis(triplet); traces=[]
        with torch.no_grad():
            for role in self.STAGES:
                st=self.lite[role].step(axis,energy,steps=2); boundary=self.lite[role].boundary(st); axis=self.lite[role].axis(st); energy=boundary.abs().mean(-1)+st.energy.mean(-1)*0.35
                traces.append(DeepRONTrace(role,float(st.energy.mean().cpu()),float(st.release.mean().cpu()),float(st.residual.mean().cpu()),[float(v) for v in boundary[0].cpu().tolist()]))
            axes=torch.stack([-axis,axis,torch.roll(axis,shifts=1,dims=-1)],dim=1); out=self.core(axes=axes,mask=mask_t,adaptive_cycles=adaptive_cycles); fields=self.cortex.fields_from_output(out); packet=self.cortex.packet_from_output(out,fields)
            d=out.diagnostics; traces.append(DeepRONTrace('full_core_conductor',float(d['energy_mean'].cpu()),float(d['release'].cpu()),float(d['residual_cost'].cpu()),[float(v) for v in out.main[0].softmax(-1).cpu().tolist()]))
        return packet,traces
