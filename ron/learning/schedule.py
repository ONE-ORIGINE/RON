from __future__ import annotations
import math

def cosine_lr(step:int, total_steps:int, base_lr:float, min_lr:float=1e-5) -> float:
    if total_steps <= 1:
        return base_lr
    t = max(0.0, min(1.0, step / float(total_steps - 1)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
