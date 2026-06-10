from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ron.runtime.hybrid_policy import HybridPolicy
from ron.runtime.active_geodesic_cortex import RONActiveGeodesicCortex
from ron.runtime.geodesic_symphony_cortex import RONGeodesicSymphonyCortex
from ron.runtime.stable_integrated_cortex import RONStableIntegratedCortex

@dataclass
class RONSystemOutput:
    packet: Any
    trace: list[Any]
    summary: dict[str, Any]
    policy: dict[str, Any]

class RONSystem:
    """Clean public runtime entry point for RON V20.

    It gives one interface while keeping the distinction between:
    - stable integrated cortex;
    - geodesic/symphony diagnostics;
    - active geodesic correction.
    """
    def __init__(self, mode: str = "ron_pure", profile: str = "active_geodesic", device: str | None = None, **kwargs):
        self.policy = HybridPolicy.from_string(mode)
        self.profile = profile
        common = dict(kwargs)
        if profile == "stable":
            self.engine = RONStableIntegratedCortex(device=device, **common)
        elif profile == "geodesic_symphony":
            self.engine = RONGeodesicSymphonyCortex(device=device, **common)
        elif profile == "active_geodesic":
            self.engine = RONActiveGeodesicCortex(device=device, **common)
        else:
            raise ValueError(f"Unknown RON profile: {profile}")

    def infer_triplet(self, x: int, y: int, z: int, mask: int = 0) -> RONSystemOutput:
        out = self.engine.infer_triplet(x, y, z, mask=mask)
        if self.profile == "stable":
            packet, trace, summary = out
            trace_out = trace
        elif self.profile == "geodesic_symphony":
            packet, stable_trace, geo_trace, summary = out
            trace_out = list(stable_trace) + list(geo_trace)
        else:
            packet, trace, summary = out
            trace_out = trace
        summary = dict(summary)
        summary["profile"] = self.profile
        summary["policy"] = self.policy.as_dict()
        return RONSystemOutput(packet=packet, trace=trace_out, summary=summary, policy=self.policy.as_dict())
