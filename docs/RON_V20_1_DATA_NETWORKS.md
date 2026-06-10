# RON V20.1 — Data, Networks, Interfaces

V20.1 starts the applied layer around RON V20.

## Added datasets

```text
ron/datasets/synthetic_dynamics.py
ron/datasets/collision_world.py
ron/datasets/triadic_sequence.py
ron/datasets/adapters/
```

The goal is to feed RON with concrete X/Y/Z contexts:

```text
X = past state
Y = present state
Z = future state
```

including axes, energy, mask, action/cause/consequence labels.

## Added networks

```text
ron/networks/ron_pure_dynamics.py
ron/networks/ron_triaxial_teacher.py
ron/networks/ron_hybrid_trifamily.py
```

### RON pure

```text
axes + energy → RON active_geodesic → RONPacket
```

### RON + triaxial

```text
axes + energy → TriaxialLens diagnostic → RON active_geodesic → RONPacket
```

### Hybrid tri-family

```text
classical deterministic codec
→ triaxial diagnostic teacher
→ RON active_geodesic conductor
→ RONPacket
```

The classical family is a boundary codec. It does not decide.

## Optional brute triaxial

```text
ron/hybrid/triaxial_brute.py
```

This wrapper can load the external `triaxial_neuron.py` file. It is not imported into the RON core.

## Interfaces

```text
interfaces/ron_core_explorer.html
interfaces/ron_hybrid_viewer.html
```

## Visuals

```text
visuals/generate_v20_1_visuals.py
visuals/output/ron_axes_xyz.png
visuals/output/ron_packet_ports.png
visuals/output/ron_active_diagnostics.png
visuals/output/ron_hybrid_trifamily_flow.png
```

## Validate

```bash
python scripts/validate_v20_1.py
```

## Use networks

```bash
python scripts/use_networks_v20_1.py --network pure --dataset dynamics --cpu
python scripts/use_networks_v20_1.py --network triaxial --dataset dynamics --cpu
python scripts/use_networks_v20_1.py --network hybrid --dataset collision --cpu
```
