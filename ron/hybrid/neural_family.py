from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

class NeuralFamily(str, Enum):
    RON = "ron"
    TRIAXIAL_BRUTE = "triaxial_brute"
    CLASSICAL = "classical"

@dataclass
class FamilySignal:
    family: NeuralFamily
    role: str
    payload: dict[str, Any]
    confidence: float = 0.0
    may_decide: bool = False

def family_policy() -> dict[str, dict[str, bool | str]]:
    """Authoritative V20 hybrid policy.

    RON decides. Other families can enrich or diagnose unless explicitly enabled
    at the boundary as codecs.
    """
    return {
        "ron": {
            "allowed_in_core": True,
            "may_decide": True,
            "default_role": "movement_causal_conductor",
        },
        "triaxial_brute": {
            "allowed_in_core": False,
            "may_decide": False,
            "default_role": "xyz_diagnostic_teacher_optional",
        },
        "classical": {
            "allowed_in_core": False,
            "may_decide": False,
            "default_role": "boundary_codec_optional",
        },
    }
