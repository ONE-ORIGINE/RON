from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ron.hybrid.neural_family import FamilySignal, NeuralFamily, family_policy

@dataclass
class HybridRouteResult:
    accepted: list[FamilySignal]
    rejected: list[FamilySignal]
    reason: dict[str, str]

class HybridRouter:
    """Safe hybridation router.

    It allows hybrid signals in the repository without letting them replace the
    RON core. RON remains the only default decision family.
    """
    def __init__(self, allow_boundary_codecs: bool = True, allow_triaxial_teacher: bool = True):
        self.allow_boundary_codecs = allow_boundary_codecs
        self.allow_triaxial_teacher = allow_triaxial_teacher
        self.policy = family_policy()

    def route(self, signals: list[FamilySignal]) -> HybridRouteResult:
        accepted: list[FamilySignal] = []
        rejected: list[FamilySignal] = []
        reason: dict[str, str] = {}
        for sig in signals:
            fam = sig.family.value
            pol = self.policy[fam]
            if sig.family == NeuralFamily.RON:
                accepted.append(sig)
                reason[fam + ":" + sig.role] = "ron_core_signal"
            elif sig.family == NeuralFamily.TRIAXIAL_BRUTE and self.allow_triaxial_teacher and not sig.may_decide:
                accepted.append(sig)
                reason[fam + ":" + sig.role] = "accepted_as_teacher_or_diagnostic"
            elif sig.family == NeuralFamily.CLASSICAL and self.allow_boundary_codecs and not sig.may_decide:
                accepted.append(sig)
                reason[fam + ":" + sig.role] = "accepted_as_boundary_codec"
            else:
                rejected.append(sig)
                reason[fam + ":" + sig.role] = "rejected_to_preserve_ron_core"
        return HybridRouteResult(accepted=accepted, rejected=rejected, reason=reason)
