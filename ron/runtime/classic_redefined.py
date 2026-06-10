from __future__ import annotations
from dataclasses import dataclass
from .packet import RONPacket

@dataclass
class RONClassicalView:
    """Classical-looking outputs redefined through RON packets.

    These are not dense heads. They are interpretations of the RON packet for
    conventional tasks.
    """
    classification: str
    classification_confidence: float
    reconstruction: str
    prediction: str
    anomaly_score: float
    action: str
    memory_write: float
    explanation: str

def packet_to_classical_view(packet: RONPacket) -> RONClassicalView:
    anomaly = max(0.0, min(1.0, 0.45*packet.residual_cost + 0.35*packet.uncertainty + 0.20*(1.0-packet.branch_access)))
    return RONClassicalView(
        classification=packet.action_face,
        classification_confidence=packet.confidence,
        reconstruction=packet.reconstruction_face,
        prediction=packet.prediction_face,
        anomaly_score=anomaly,
        action=packet.control_mode,
        memory_write=packet.memory_write_strength,
        explanation=packet.explanation
    )
