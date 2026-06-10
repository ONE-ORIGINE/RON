from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.ron_lite import RONLiteBank
from ron.core.whole_core import RONWholeTriAxialCore
from ron.core.ports import face_normals
from ron.graph import RONSynapseBank, build_default_topology
from .cortex import RONCortex
from .packet import RONPacket

@dataclass
class RecurrentLoopTrace:
    loop: int
    total_flow: float
    resonance: float
    trust: float
    conductor_release: float
    conductor_residual: float
    conductor_access: float
    active_edges: float
    plastic_delta: float

class RONRecurrentCortex:
    """RON V20 recurrent RON cortex.

    The cortex does five RON-native refinement loops:
    improve -> infer -> optimize -> correct -> repeat.
    Communication is through RON synapses: port, energy, orientation, resonance,
    trust and plastic trace. No dense bridge is introduced.
    """
    ROLES = ['sensory','spatial','temporal','causal','memory','value','conductor']
    LITE_ROLES = ['sensory','spatial','temporal','causal','memory','value']

    def __init__(self, device: str|None=None, lite_cells:int=12, trace_dim:int=8, loops:int=5, **core_kwargs):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.loops=loops
        self.lite={r: RONLiteBank(cells=lite_cells, trace_dim=trace_dim, seed=900+i).to(self.device).eval() for i,r in enumerate(self.LITE_ROLES)}
        self.core=RONWholeTriAxialCore(**core_kwargs).to(self.device).eval()
        self.cortex=RONCortex(self.core,device=str(self.device))
        topo=build_default_topology(device=self.device)
        self.synapses=RONSynapseBank(topo,device=self.device).to(self.device).eval()

    def _boundary_to_axis(self,boundary: Tensor) -> Tensor:
        n=face_normals(boundary.device,boundary.dtype); p=boundary.softmax(-1); a=p@n
        return a/a.norm(dim=-1,keepdim=True).clamp_min(1e-6)

    def _init_state(self, triplet: Tensor):
        axes,energies=self.core.token_axes(triplet)
        base_axis=axes.mean(1); base_energy=energies.mean(1)
        B=triplet.shape[0]; N=len(self.ROLES); dev=triplet.device; dtype=torch.float32
        role_axes=torch.zeros(B,N,3,device=dev,dtype=dtype)
        role_energy=torch.zeros(B,N,device=dev,dtype=dtype)
        role_boundaries=torch.ones(B,N,6,device=dev,dtype=dtype)/6.0
        # Seed roles with different interpretations of the same triplet.
        role_axes[:,0]=axes[:,1]                # sensory/present
        role_axes[:,1]=axes[:,1] + 0.25*axes[:,0]
        role_axes[:,2]=axes[:,2] - 0.15*axes[:,0]
        role_axes[:,3]=axes[:,2] - axes[:,0]
        role_axes[:,4]=axes[:,0]
        role_axes[:,5]=axes[:,2]
        role_axes[:,6]=base_axis
        role_axes=role_axes/role_axes.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        role_energy[:,:6]=base_energy[:,None]*(0.75+0.05*torch.arange(6,device=dev,dtype=dtype).view(1,6))
        role_energy[:,6]=base_energy
        return role_axes,role_energy,role_boundaries

    def infer_triplet(self,x:int,y:int,z:int,mask:int=0,loops:int|None=None,adaptive_cycles:bool=True):
        L=int(loops or self.loops)
        triplet=torch.tensor([[x,y,z]],device=self.device,dtype=torch.long)
        mask_t=torch.tensor([mask],device=self.device,dtype=torch.long)
        axes,energy,boundaries=self._init_state(triplet)
        traces=[]; out=None; packet=None
        states={}
        with torch.no_grad():
            incoming_axis=torch.zeros_like(axes); incoming_energy=torch.zeros_like(energy)
            incoming_axis[:] = axes
            for t in range(1,L+1):
                prev_residual = 1.0 if out is None else float(out.diagnostics['residual_cost'].detach().cpu())
                # Update lite populations from synaptic input plus their own state.
                for i,role in enumerate(self.LITE_ROLES):
                    bank=self.lite[role]
                    drive_axis=(0.58*incoming_axis[:,i]+0.42*axes[:,i])
                    drive_axis=drive_axis/drive_axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
                    drive_energy=(0.60*incoming_energy[:,i]+0.40*energy[:,i]).clamp_min(0.01)
                    st=bank.step(drive_axis,drive_energy,state=states.get(role),steps=2)
                    states[role]=st
                    axes[:,i]=bank.axis(st)
                    energy[:,i]=bank.boundary(st).abs().mean(-1)+0.20*st.energy.mean(-1)
                    boundaries[:,i]=bank.boundary(st)
                # Full conductor integrates spatial/temporal/causal-memory-value axes.
                conductor_axes=torch.stack([
                    axes[:,4],
                    (axes[:,1]+axes[:,2]+axes[:,5])/3.0,
                    (axes[:,3]+axes[:,5])/2.0,
                ],dim=1)
                conductor_axes=conductor_axes/conductor_axes.norm(dim=-1,keepdim=True).clamp_min(1e-6)
                out=self.core(axes=conductor_axes,mask=mask_t,adaptive_cycles=adaptive_cycles)
                fields=self.cortex.fields_from_output(out)
                packet=self.cortex.packet_from_output(out,fields)
                boundaries[:,6]=out.main.softmax(-1)
                axes[:,6]=self._boundary_to_axis(boundaries[:,6])
                energy[:,6]=float(out.diagnostics['energy_mean'].detach().cpu())
                # Synaptic transport for next loop.
                incoming_axis,incoming_energy,flow,diag=self.synapses(boundaries,axes,energy)
                usefulness=float(out.diagnostics['branch_access'].detach().cpu()) + float(out.diagnostics['release'].detach().cpu()) - float(out.diagnostics['dissonance'].detach().cpu())
                plastic_delta=self.synapses.plasticity_update_(flow,prev_residual,out.diagnostics['residual_cost'],usefulness)
                traces.append(RecurrentLoopTrace(
                    loop=t,
                    total_flow=float(diag.total_flow.detach().cpu()),
                    resonance=float(diag.mean_resonance.detach().cpu()),
                    trust=float(diag.mean_trust.detach().cpu()),
                    conductor_release=float(out.diagnostics['release'].detach().cpu()),
                    conductor_residual=float(out.diagnostics['residual_cost'].detach().cpu()),
                    conductor_access=float(out.diagnostics['branch_access'].detach().cpu()),
                    active_edges=float(diag.active_edges.detach().cpu()),
                    plastic_delta=float(plastic_delta.detach().cpu()) if hasattr(plastic_delta,'detach') else float(plastic_delta),
                ))
            return packet,traces
