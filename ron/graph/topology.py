from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class RONTopology:
    roles: list[str]
    src: torch.Tensor
    dst: torch.Tensor
    src_port: torch.Tensor
    dst_port: torch.Tensor
    kind: list[str]

_FACE = {'+X':0,'-X':1,'+Y':2,'-Y':3,'+Z':4,'-Z':5}

def build_default_topology(device=None) -> RONTopology:
    """Default recurrent RON cortex topology.

    It contains forward flows, feedback flows, memory shortcuts, causal shortcuts,
    and local self-stabilizing loops. Edges are port-to-port, not dense matrices.
    """
    roles = ['sensory','spatial','temporal','causal','memory','value','conductor']
    idx = {r:i for i,r in enumerate(roles)}
    edges = []
    def add(s,d,sp,dp,k): edges.append((idx[s],idx[d],_FACE[sp],_FACE[dp],k))
    # forward perception-action stream
    add('sensory','spatial','+X','-X','forward')
    add('spatial','temporal','+Y','-Y','forward')
    add('temporal','causal','+Z','-Z','forward')
    add('causal','memory','+X','-X','forward')
    add('memory','value','+Y','-Y','forward')
    add('value','conductor','+Z','-Z','forward')
    # shortcuts: RON-native attention as port resonance
    add('sensory','memory','+Z','-Z','memory_shortcut')
    add('spatial','causal','+X','-X','causal_shortcut')
    add('temporal','value','+Y','-Y','temporal_value')
    add('causal','conductor','+Z','-Z','causal_conductor')
    add('memory','conductor','+X','-X','memory_conductor')
    # conductor feedback, used to correct premature release and request more processing
    for r,p in [('sensory','-Z'),('spatial','-X'),('temporal','-Y'),('causal','-Z'),('memory','-X'),('value','-Y')]:
        add('conductor',r,'-Z',p,'feedback')
    # local recurrent stabilizers
    for r in roles:
        add(r,r,'+X','-X','local')
        add(r,r,'+Y','-Y','local')
    src=torch.tensor([e[0] for e in edges],device=device,dtype=torch.long)
    dst=torch.tensor([e[1] for e in edges],device=device,dtype=torch.long)
    sp=torch.tensor([e[2] for e in edges],device=device,dtype=torch.long)
    dp=torch.tensor([e[3] for e in edges],device=device,dtype=torch.long)
    kind=[e[4] for e in edges]
    return RONTopology(roles,src,dst,sp,dp,kind)
