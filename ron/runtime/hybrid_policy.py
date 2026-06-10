from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

class RONMode(str, Enum):
    RON_PURE = "ron_pure"
    RON_TRIAXIAL = "ron_triaxial"
    RON_HYBRID = "ron.hybrid"

@dataclass
class HybridPolicy:
    """Explicit policy for what is allowed around the RON core.

    RON_PURE:
        only RON-native modules.

    RON_TRIAXIAL:
        allows the RON-native TriaxialLens bridge, not the classical triaxial GRU/attention model.

    RON_HYBRID:
        reserves the right to attach external codecs/adapters at the boundary, but the conductor
        and packet decision must remain RON-native.
    """
    mode: RONMode = RONMode.RON_PURE
    allow_external_codec: bool = False
    allow_triaxial_lens: bool = True
    allow_classical_core: bool = False

    @classmethod
    def from_string(cls, mode: str) -> "HybridPolicy":
        m = RONMode(mode)
        if m == RONMode.RON_PURE:
            return cls(m, allow_external_codec=False, allow_triaxial_lens=False, allow_classical_core=False)
        if m == RONMode.RON_TRIAXIAL:
            return cls(m, allow_external_codec=False, allow_triaxial_lens=True, allow_classical_core=False)
        return cls(m, allow_external_codec=True, allow_triaxial_lens=True, allow_classical_core=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "allow_external_codec": self.allow_external_codec,
            "allow_triaxial_lens": self.allow_triaxial_lens,
            "allow_classical_core": self.allow_classical_core,
        }
