from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math

@dataclass
class RONInvariantReport:
    packet_numeric: bool
    route_normalized: bool
    release_bounded: bool
    confidence_bounded: bool
    energy_nonnegative: bool
    residual_nonnegative: bool
    bus_flow_nonnegative: bool
    synapse_trace_nonnegative: bool
    ok: bool
    notes: list[str]

def _finite(x: float) -> bool:
    return isinstance(x, (float, int)) and math.isfinite(float(x))

def validate_packet_and_bus(packet: Any, bus: dict) -> RONInvariantReport:
    notes: list[str] = []
    values = [
        packet.confidence, packet.release, packet.branch_access, packet.energy,
        packet.residual_cost, packet.phi_pressure, packet.uncertainty,
        packet.memory_write_strength
    ]
    packet_numeric = all(_finite(v) for v in values) and all(_finite(v) for v in packet.route_vector)
    if not packet_numeric:
        notes.append("non_finite_packet_value")

    route_sum = sum(float(v) for v in packet.route_vector) if packet.route_vector else 0.0
    route_normalized = abs(route_sum - 1.0) < 1e-3
    if not route_normalized:
        notes.append(f"route_sum={route_sum:.6f}")

    release_bounded = 0.0 <= float(packet.release) <= 1.0
    confidence_bounded = 0.0 <= float(packet.confidence) <= 1.0
    energy_nonnegative = float(packet.energy) >= 0.0
    residual_nonnegative = float(packet.residual_cost) >= 0.0
    bus_flow_nonnegative = float(bus.get("bus_total_flow", 0.0)) >= 0.0
    synapse_trace_nonnegative = float(bus.get("trace_mean", 0.0)) >= 0.0

    for name, ok in [
        ("release_bounded", release_bounded),
        ("confidence_bounded", confidence_bounded),
        ("energy_nonnegative", energy_nonnegative),
        ("residual_nonnegative", residual_nonnegative),
        ("bus_flow_nonnegative", bus_flow_nonnegative),
        ("synapse_trace_nonnegative", synapse_trace_nonnegative),
    ]:
        if not ok:
            notes.append(name)

    ok = all([
        packet_numeric, route_normalized, release_bounded, confidence_bounded,
        energy_nonnegative, residual_nonnegative, bus_flow_nonnegative,
        synapse_trace_nonnegative,
    ])
    return RONInvariantReport(
        packet_numeric=packet_numeric,
        route_normalized=route_normalized,
        release_bounded=release_bounded,
        confidence_bounded=confidence_bounded,
        energy_nonnegative=energy_nonnegative,
        residual_nonnegative=residual_nonnegative,
        bus_flow_nonnegative=bus_flow_nonnegative,
        synapse_trace_nonnegative=synapse_trace_nonnegative,
        ok=ok,
        notes=notes,
    )

def packet_coherence(packet: Any) -> dict:
    """RON-native coherence, not classical accuracy."""
    confidence = float(packet.confidence)
    release = float(packet.release)
    access = float(packet.branch_access)
    residual = float(packet.residual_cost)
    uncertainty = float(packet.uncertainty)
    memory = float(packet.memory_write_strength)
    # Coherence favors accessible, confident low-residual packets.
    score = max(0.0, min(1.0, 0.28*confidence + 0.25*access + 0.18*release + 0.16*(1.0-residual) + 0.13*(1.0-uncertainty)))
    correction_pressure = max(0.0, min(1.0, 0.40*residual + 0.25*uncertainty + 0.20*(1.0-access) + 0.15*memory))
    return {
        "coherence_score": score,
        "correction_pressure": correction_pressure,
        "release_consistent": bool(release > 0.5 and correction_pressure < 0.65),
        "needs_memory": bool(memory > 0.45),
        "needs_more_context": bool(correction_pressure > 0.70),
    }
