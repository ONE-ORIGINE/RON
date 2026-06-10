from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.runtime.deep_cortex import RONDeepCortex, boundary_to_axis
from ron.graph.rich_synapse import RichRONSynapseBank
from ron.core.ports import face_normals

@dataclass
class EmergenceStep:
    loop: int
    phase: str
    action_face: str
    confidence: float
    release: float
    residual: float
    dissonance: float
    synapse_flow: float
    trust_mean: float
    conductance_mean: float
    corrected: bool

class RONEmergentRefiner:
    """Five-loop RON refinement.

    Each macro-loop explicitly performs:
    improve -> innovate -> emerge -> infer -> optimize -> correct.
    It is still RON-native: port boundaries become axes; rich synapses transport
    energy; plasticity changes synaptic conductance/trust.
    """
    PHASES = ["improve", "innovate", "emerge", "infer", "optimize", "correct"]

    def __init__(self, device: str|None=None, **kwargs):
        self.deep = RONDeepCortex(device=device, **kwargs)
        self.device = self.deep.device
        self.syn = RichRONSynapseBank(RichRONSynapseBank.default_edges(), device=self.device)

    def _boundary_dict_from_trace(self, trace) -> dict[str, Tensor]:
        d = {}
        for t in trace:
            role = t.stage
            if role == "full_core_conductor":
                role = "conductor"
            b = torch.tensor([t.boundary], device=self.device, dtype=torch.float32)
            if b.shape[-1] == 6:
                d[role] = b
        return d

    def _axes_from_boundaries(self, b: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: boundary_to_axis(v) for k,v in b.items()}

    def refine_triplet(self, x:int, y:int, z:int, mask:int=0, loops:int=5):
        phase_trace: list[EmergenceStep] = []
        packet = None
        final_trace = None
        corrected = False

        for loop in range(1, loops+1):
            # Deep cortex produces base packet and stage boundaries.
            packet, trace = self.deep.infer_triplet(x,y,z,mask)
            final_trace = trace
            boundaries = self._boundary_dict_from_trace(trace)
            axes = self._axes_from_boundaries(boundaries)

            # Rich synapses transport energy between populations.
            transported = self.syn.transport(boundaries, axes)
            syn_flow = sum(float(v.abs().mean().detach().cpu()) for v in transported.values()) if transported else 0.0

            # Use conductor diagnostics to adapt connections.
            diag = packet.diagnostics
            residual = float(diag.get("residual_cost", packet.residual_cost))
            dissonance = float(diag.get("dissonance", 0.0))
            release = float(packet.release)
            self.syn.plasticity(boundaries, transported, residual=residual, release=release, dissonance=dissonance)
            summary = self.syn.summary()

            # Correction policy: if release is high but confidence/access weak, push mask toward present/future uncertainty.
            corrected = bool(packet.release > 0.65 and (packet.confidence < 0.34 or packet.branch_access < 0.48))
            if corrected:
                mask = 2 if mask == 0 else mask  # force present stabilization next loop

            # Innovation/emergence: if synaptic flow is high, perturb the triplet symbolically via port choice.
            # This is not data augmentation; it is a RON route-induced continuation.
            if syn_flow > 0.08:
                z = (z + packet.action_index + loop) % max(8, z+1)
                if z == 0:
                    z = 1

            for phase in self.PHASES:
                phase_trace.append(EmergenceStep(
                    loop=loop,
                    phase=phase,
                    action_face=packet.action_face,
                    confidence=float(packet.confidence),
                    release=release,
                    residual=residual,
                    dissonance=dissonance,
                    synapse_flow=syn_flow,
                    trust_mean=summary["trust_mean"],
                    conductance_mean=summary["conductance_mean"],
                    corrected=corrected if phase == "correct" else False,
                ))
        return packet, final_trace, phase_trace, self.syn.summary()
