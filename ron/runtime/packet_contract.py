from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ron.runtime.packet import RONPacket, FACE_NAMES

REQUIRED_PACKET_FIELDS = {
    "action_face": str,
    "action_index": int,
    "confidence": float,
    "release": float,
    "branch_access": float,
    "energy": float,
    "residual_cost": float,
    "phi_pressure": float,
    "prediction_face": str,
    "reconstruction_face": str,
    "stabilization_face": str,
    "cause_face": str,
    "consequence_face": str,
    "diagonal_face": str,
    "route_vector": list,
    "uncertainty": float,
    "should_continue": bool,
    "should_release": bool,
    "memory_write_strength": float,
    "control_mode": str,
    "explanation": str,
    "diagnostics": dict,
}

@dataclass
class PacketValidation:
    ok: bool
    errors: list[str]
    warnings: list[str]

def validate_packet(packet: RONPacket | dict[str, Any]) -> PacketValidation:
    d = packet.as_dict() if hasattr(packet, "as_dict") else dict(packet)
    errors: list[str] = []
    warnings: list[str] = []
    for field, typ in REQUIRED_PACKET_FIELDS.items():
        if field not in d:
            errors.append(f"missing:{field}")
        elif not isinstance(d[field], typ):
            # bool is subclass of int; forbid this for integer fields except booleans.
            if typ is int and isinstance(d[field], bool):
                errors.append(f"type:{field}:bool_not_int")
            else:
                errors.append(f"type:{field}:{type(d[field]).__name__}")
    if "action_face" in d and d["action_face"] not in FACE_NAMES:
        errors.append("action_face:not_known_face")
    if "route_vector" in d:
        rv = d["route_vector"]
        if len(rv) != 6:
            errors.append("route_vector:length_not_6")
        else:
            s = sum(float(v) for v in rv)
            if abs(s - 1.0) > 1e-3:
                warnings.append(f"route_vector:sum={s:.6f}")
            if any(float(v) < -1e-6 for v in rv):
                errors.append("route_vector:negative_probability")
    for k in ("confidence", "release", "branch_access", "uncertainty", "memory_write_strength"):
        if k in d and isinstance(d[k], (int, float)):
            if not (0.0 <= float(d[k]) <= 1.0):
                errors.append(f"range:{k}")
    if d.get("should_continue") and d.get("should_release"):
        warnings.append("continue_and_release_both_true")
    return PacketValidation(ok=len(errors) == 0, errors=errors, warnings=warnings)

def packet_schema() -> dict[str, str]:
    return {k: v.__name__ for k, v in REQUIRED_PACKET_FIELDS.items()}
