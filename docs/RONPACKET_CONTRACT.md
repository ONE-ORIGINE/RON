# RONPacket Contract

The official RON runtime output is `RONPacket`.

## Required fields

```text
action_face
action_index
confidence
release
branch_access
energy
residual_cost
phi_pressure
prediction_face
reconstruction_face
stabilization_face
cause_face
consequence_face
diagonal_face
route_vector
uncertainty
should_continue
should_release
memory_write_strength
control_mode
explanation
diagnostics
```

## Meaning

```text
action_face            selected output port
confidence             route confidence
release                release gate value
branch_access          causal accessibility
energy                 active mean energy
residual_cost          internal motion inconsistency
phi_pressure           sphere/cube pressure
prediction_face        future-facing route
reconstruction_face    past-facing route
stabilization_face     present-stabilizing route
cause_face             causal-origin route
consequence_face       consequence route
diagonal_face          cross-axis route
route_vector           six-face probability vector
uncertainty            uncertainty estimate
should_continue        internal continuation flag
should_release         packet emission flag
memory_write_strength  memory write pressure
control_mode           release/continue label
explanation            human-readable packet summary
diagnostics            detailed RON-native metrics
```
