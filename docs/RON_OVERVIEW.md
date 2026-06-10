# RON Overview

RON is an artificial-neuron architecture based on movement.

Instead of treating the neuron as a scalar activation unit, RON treats it as an oriented causal operator.  
A RON unit carries an orientation, energy, motion trace, memory pressure, and port-routing state.

## Main components

```text
Quaternion orientation
Geodesic residual
Phi sphere/cube pressure
Cube memory
Port routing
Causal branch access
Quaternion symphony
Release/continue gate
RONPacket output
```

## Default runtime

```text
RONSystem(mode="ron_pure", profile="active_geodesic")
```

## Why active_geodesic

`active_geodesic` is the main operating mode because it connects the core ideas:

```text
stable cortex
+ geodesic residual
+ Phi pressure
+ quaternion symphony
+ geodesic memory
+ active bounded conductor correction
```

## Official output

The system emits a `RONPacket`.

A RONPacket is the public contract of the runtime and is designed to be readable, inspectable, and comparable across experiments.
