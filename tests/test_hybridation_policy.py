from __future__ import annotations
from ron.hybrid import HybridRouter, FamilySignal, NeuralFamily

def test_hybrid_router_keeps_ron_decision():
    router = HybridRouter()
    signals = [
        FamilySignal(NeuralFamily.RON, "conductor", {}, confidence=0.8, may_decide=True),
        FamilySignal(NeuralFamily.TRIAXIAL_BRUTE, "teacher", {}, confidence=0.6, may_decide=False),
        FamilySignal(NeuralFamily.CLASSICAL, "codec", {}, confidence=0.4, may_decide=False),
        FamilySignal(NeuralFamily.CLASSICAL, "hidden_decider", {}, confidence=1.0, may_decide=True),
    ]
    result = router.route(signals)
    assert len(result.accepted) == 3
    assert len(result.rejected) == 1
    assert result.rejected[0].role == "hidden_decider"
