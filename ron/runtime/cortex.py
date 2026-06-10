from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ron.core.whole_core import RONWholeTriAxialCore
from .packet import RONPacket, FACE_NAMES

def _softmax_conf(logits: Tensor) -> tuple[int, float, list[float]]:
    p = logits.softmax(-1)
    idx = int(p.argmax(-1).detach().cpu())
    return idx, float(p[idx].detach().cpu()), [float(v) for v in p.detach().cpu().tolist()]

def _face(logits: Tensor) -> str:
    return FACE_NAMES[int(logits.argmax(-1).detach().cpu())]

@dataclass
class CortexFields:
    spatial: Tensor
    temporal: Tensor
    causal: Tensor
    memory: Tensor
    value: Tensor
    conductor: Tensor

class RONCortex:
    """Small operational cortex around the Whole RON core.

    The cortex does not stack arbitrary heads. It interprets the rich RON output
    through fields that correspond to functions of a brain-like artificial system:
    spatial stabilization, temporal completion, causal explanation, memory update,
    value/consequence, and conductor/release.
    """
    def __init__(self, model: RONWholeTriAxialCore|None=None, device: str|None=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model=(model or RONWholeTriAxialCore()).to(self.device).eval()

    def fields_from_output(self, out) -> CortexFields:
        # Each field is derived from RON-native boundaries and diagnostics.
        spatial = 0.45*out.present + 0.35*out.diagonal + 0.20*out.main
        temporal = 0.45*out.future + 0.35*out.past + 0.20*out.reverse
        causal = 0.42*out.cause + 0.42*out.consequence + 0.16*out.diagonal
        # Memory field is a write-pressure field, not a dense memory head.
        residual = out.diagnostics['residual_cost'].view(-1,1)
        phi = out.diagnostics.get('phi_pressure', torch.zeros_like(residual.squeeze(-1))).view(-1,1)
        energy = out.diagnostics['energy_mean'].view(-1,1)
        memory = torch.tanh(0.30*out.present + 0.30*out.past + 0.25*out.diagonal + 0.15*(residual+phi-energy))
        # Value field estimates useful accessible consequence.
        access = out.diagnostics['branch_access'].view(-1,1)
        dis = out.diagnostics['dissonance'].view(-1,1)
        value = torch.tanh(out.consequence + 0.35*access - 0.30*dis - 0.15*energy)
        # Conductor field decides release/action by integrating all fields.
        conductor = torch.tanh(0.28*spatial + 0.28*temporal + 0.28*causal + 0.18*value + 0.10*memory)
        return CortexFields(spatial, temporal, causal, memory, value, conductor)

    def packet_from_output(self, out, fields: CortexFields) -> RONPacket:
        logits = fields.conductor[0]
        action_idx, conf, route = _softmax_conf(logits)
        diag={k:float(v.detach().cpu()) if torch.is_tensor(v) and v.numel()==1 else str(v) for k,v in out.diagnostics.items()}
        residual=diag.get('residual_cost',0.0)
        phi=diag.get('phi_pressure',0.0)
        dis=diag.get('dissonance',0.0)
        access=diag.get('branch_access',0.0)
        release=diag.get('release',0.0)
        energy=diag.get('energy_mean',0.0)
        uncertainty=float(max(0.0, min(1.0, 0.45*residual + 0.35*dis + 0.20*(1.0-conf))))
        should_release=bool(release>0.50 and uncertainty<0.65 and access>0.35)
        should_continue=not should_release
        memory_write_strength=float(max(0.0, min(1.0, 0.45*residual + 0.25*phi + 0.20*(1.0-access) + 0.10*energy)))
        if should_release:
            mode='release_packet'
        elif uncertainty>0.70:
            mode='request_more_context'
        elif memory_write_strength>0.45:
            mode='write_memory_then_continue'
        else:
            mode='continue_internal_processing'
        explanation=(
            f"RON selected port {FACE_NAMES[action_idx]} with confidence {conf:.3f}; "
            f"release={release:.3f}, access={access:.3f}, residual={residual:.3f}, "
            f"dissonance={dis:.3f}. Mode: {mode}."
        )
        return RONPacket(
            action_face=FACE_NAMES[action_idx],
            action_index=action_idx,
            confidence=conf,
            release=release,
            branch_access=access,
            energy=energy,
            residual_cost=residual,
            phi_pressure=phi,
            prediction_face=_face(out.future[0]),
            reconstruction_face=_face(out.past[0]),
            stabilization_face=_face(out.present[0]),
            cause_face=_face(out.cause[0]),
            consequence_face=_face(out.consequence[0]),
            diagonal_face=_face(out.diagonal[0]),
            route_vector=route,
            uncertainty=uncertainty,
            should_continue=should_continue,
            should_release=should_release,
            memory_write_strength=memory_write_strength,
            control_mode=mode,
            explanation=explanation,
            diagnostics=diag,
        )

    def infer_triplet(self, x:int, y:int, z:int, mask:int=0, adaptive_cycles:bool=True) -> RONPacket:
        triplet=torch.tensor([[x,y,z]],device=self.device,dtype=torch.long)
        mask_t=torch.tensor([mask],device=self.device,dtype=torch.long)
        with torch.no_grad():
            out=self.model(triplet=triplet, mask=mask_t, adaptive_cycles=adaptive_cycles)
            fields=self.fields_from_output(out)
            return self.packet_from_output(out, fields)

    def infer_axes(self, axes: Tensor, mask: Tensor|None=None, adaptive_cycles:bool=True) -> RONPacket:
        axes=axes.to(self.device)
        mask=mask.to(self.device) if mask is not None else None
        with torch.no_grad():
            out=self.model(axes=axes, mask=mask, adaptive_cycles=adaptive_cycles)
            fields=self.fields_from_output(out)
            return self.packet_from_output(out, fields)
