# RON Native Learning

RON V20 includes a minimal native learning path.

It is intentionally small and CPU-safe:

```text
native synthetic task
→ RONWholeTriAxialCore
→ RON-native objective
→ gradient-assisted update
→ RON projection
→ checkpoint
→ reload validation
```

Run:

```bash
python scripts/train.py --steps 12 --batch-size 16 --cpu
```

The native objective uses RON outputs directly:

```text
future
past
present
cause
consequence
diagonal
residual
Phi pressure
dissonance
port entropy
release balance
geodesic pressure
```

No dense classifier head is added.
