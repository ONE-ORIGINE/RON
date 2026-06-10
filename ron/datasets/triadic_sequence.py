from __future__ import annotations
import torch
from ron.learning.native_tasks import make_native_batch

def make_triadic_sequence_batch(batch_size:int=32, vocab:int=12, seed:int=0, device=None):
    """Alias to the native triadic completion batch used by RON learning."""
    return make_native_batch(batch_size=batch_size, vocab=vocab, seed=seed, device=device)
