from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

FACE_NAMES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

@dataclass
class RONPacket:
    """Operational output packet of a RON field.

    This is what a RON can emit to another RON, to a controller, or to a decoder.
    It is deliberately richer than a class label.
    """
    action_face: str
    action_index: int
    confidence: float
    release: float
    branch_access: float
    energy: float
    residual_cost: float
    phi_pressure: float
    prediction_face: str
    reconstruction_face: str
    stabilization_face: str
    cause_face: str
    consequence_face: str
    diagonal_face: str
    route_vector: list[float]
    uncertainty: float
    should_continue: bool
    should_release: bool
    memory_write_strength: float
    control_mode: str
    explanation: str
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
