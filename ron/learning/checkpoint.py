from __future__ import annotations
from pathlib import Path
import json
import torch

def save_checkpoint(path: str | Path, model, optimizer=None, metadata: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path

def load_checkpoint(path: str | Path, model, optimizer=None):
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("metadata", {})
