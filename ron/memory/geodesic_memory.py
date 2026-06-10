from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class MemoryRead:
    found: bool
    axis: Tensor
    confidence: Tensor
    slot_index: Tensor
    similarity: Tensor

class RONGeodesicMemoryBank:
    """Small RON-native geodesic memory.

    Stores axis keys and correction axes. This is not an embedding table; it is
    a bounded runtime memory for port/orientation traces.
    """
    def __init__(self, slots:int=32, device=None, dtype=torch.float32):
        self.slots = slots
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.keys = torch.zeros(slots, 3, device=self.device, dtype=dtype)
        self.values = torch.zeros(slots, 3, device=self.device, dtype=dtype)
        self.energy = torch.zeros(slots, device=self.device, dtype=dtype)
        self.age = torch.zeros(slots, device=self.device, dtype=dtype)
        self.used = torch.zeros(slots, device=self.device, dtype=torch.bool)

    def _unit(self, x: Tensor) -> Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def read(self, query_axis: Tensor) -> MemoryRead:
        q = self._unit(query_axis.to(self.device, self.dtype))
        if not bool(self.used.any()):
            idx = torch.zeros(q.shape[0], dtype=torch.long, device=self.device)
            return MemoryRead(False, torch.zeros_like(q), torch.zeros(q.shape[0],device=self.device,dtype=self.dtype), idx, torch.zeros(q.shape[0],device=self.device,dtype=self.dtype))
        keys = self._unit(self.keys)
        sim = q @ keys.T
        sim = sim.masked_fill(~self.used.view(1,-1), -1e9)
        best_sim, idx = sim.max(dim=-1)
        conf = torch.sigmoid(4.0 * (best_sim - 0.35)) * self.energy[idx].clamp(0, 1)
        value = self.values[idx]
        return MemoryRead(True, value, conf, idx, best_sim)

    def write(self, key_axis: Tensor, value_axis: Tensor, strength: Tensor | float):
        k = self._unit(key_axis.to(self.device, self.dtype))
        v = self._unit(value_axis.to(self.device, self.dtype))
        if isinstance(strength, float):
            s = torch.full((k.shape[0],), strength, device=self.device, dtype=self.dtype)
        else:
            s = strength.to(self.device, self.dtype).view(-1).clamp(0,1)
        with torch.no_grad():
            for i in range(k.shape[0]):
                if bool(self.used.any()):
                    sim = (self._unit(self.keys) @ k[i]).masked_fill(~self.used, -1e9)
                    best_sim, best_idx = sim.max(dim=0)
                else:
                    best_sim = torch.tensor(-1.0, device=self.device)
                    best_idx = torch.tensor(0, device=self.device)
                if float(best_sim) > 0.82:
                    idx = int(best_idx)
                elif bool((~self.used).any()):
                    idx = int((~self.used).nonzero()[0])
                else:
                    idx = int((self.energy - 0.003*self.age).argmin())
                self.used[idx] = True
                rate = float(s[i].clamp(0.02, 0.75))
                self.keys[idx] = self._unit((1-rate)*self.keys[idx] + rate*k[i])
                self.values[idx] = self._unit((1-rate)*self.values[idx] + rate*v[i])
                self.energy[idx] = ((1-rate)*self.energy[idx] + rate*s[i]).clamp(0,1)
                self.age[idx] = 0.0
            self.age.add_(1.0)

    def summary(self) -> dict:
        return {
            "slots": self.slots,
            "used_slots": int(self.used.sum().detach().cpu()),
            "energy_mean": float(self.energy[self.used].mean().detach().cpu()) if bool(self.used.any()) else 0.0,
            "age_mean": float(self.age[self.used].mean().detach().cpu()) if bool(self.used.any()) else 0.0,
        }
