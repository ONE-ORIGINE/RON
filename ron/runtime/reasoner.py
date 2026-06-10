from __future__ import annotations
from ron.core.whole_core import RONWholeTriAxialCore
from .cortex import RONCortex
from .packet import RONPacket

class RONReasoner:
    """Operational wrapper around the RON Cortex.

    A RON is useful as an inference-routing unit: it completes missing axes,
    estimates cause and consequence, evaluates access and release, and emits a
    packet that can be used by another RON, a controller, or a decoder.
    """
    def __init__(self, model: RONWholeTriAxialCore|None=None, device: str|None=None):
        self.cortex=RONCortex(model=model, device=device)

    def from_triplet(self, x:int, y:int, z:int, mask:int=0, adaptive_cycles:bool=True) -> RONPacket:
        return self.cortex.infer_triplet(x,y,z,mask,adaptive_cycles)

    def as_dict(self, packet: RONPacket) -> dict:
        return packet.as_dict()
