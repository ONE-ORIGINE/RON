# RON V20

**RON** is a new artificial-neuron architecture built around movement, orientation, causal routing, geodesic memory, and packet-based decision output.

RON starts from a different primitive than a scalar neuron.  
Its central unit is an oriented causal movement operator that carries:

```text
orientation
motion
energy
memory
ports
causal branch access
release / continue control
```

The official output is:

```text
RONPacket
```

The default runtime profile is:

```text
active_geodesic
```

---

## Core idea

RON asks:

```text
What movement is happening?
What caused it?
What follows?
Should the system release or continue?
```

The resulting decision is not only an index.  
It is a structured packet containing action, confidence, energy, residuals, branch access, route vector, memory pressure, and diagnostics.

---

## Repository structure

```text
ron/
  core/          quaternion, cube memory, Phi pressure, geodesic residual
  graph/         ports, synapses, cortical bus
  runtime/       RONSystem, RONPacket, active_geodesic cortex
  memory/        geodesic memory
  learning/      native learning path
  datasets/      synthetic RON-native data
  networks/      RON pure, RON+triaxial, hybrid tri-family networks
  experiments/   reproducible scientific protocols
  hybrid/        optional hybridation policy
  bridges/       RON-native diagnostic bridges
  codecs/        boundary codecs

scripts/
examples/
tests/
docs/
visuals/
```

---

## Install

```bash
pip install -e .
```

---

## Minimal demo

```bash
python scripts/infer.py --x 2 --y 5 --z 7 --mask 3 --device auto
```

or:

```python
from ron import RONSystem, validate_packet

ron = RONSystem(
    mode="ron_pure",
    profile="active_geodesic",
)

out = ron.infer_triplet(2, 5, 7, mask=3)

assert validate_packet(out.packet).ok
print(out.packet.as_dict())
```

---

## RONPacket

A `RONPacket` contains:

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

---

## Run validation

```bash
python scripts/validate.py
```

---

## Run native training

```bash
python scripts/train.py --steps 12 --batch-size 16 --device auto
```

---

## Run comparative protocol

```bash
python scripts/run_v20_3_scientific_protocol.py \
  --steps 200 \
  --batch-size 64 \
  --seeds 10 \
  --device auto
```

The protocol writes reproducible CSV and JSON outputs under `runs/`.

---

## Generate scientific graphics

```bash
python visuals/generate_v20_3_publication_graphics.py
```

---

## Current status

RON V20 is a working research architecture for a new artificial-neuron family.

It includes:

```text
runnable core
packet contract
native learning path
controlled hybridation
synthetic datasets
comparative protocols
scientific graphics
tests
documentation
```

The next research phase is to strengthen the evidence through harder tasks, stronger protocols, larger runs, and external review.
