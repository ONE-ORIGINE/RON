# Dataset adapters

These adapters are intentionally light. They define the expected conversion target:

```text
external video / robot / physics data
→ X,Y,Z states
→ axes [B,3,3]
→ energy [B,3]
→ mask
→ action/cause/consequence labels when available
```

Suggested real datasets:

```text
CLEVRER
Physion / Physion++
Kubric
robomimic
```

The V20.1 artifact includes synthetic RON-native data first, so the core can be trained without external downloads.
