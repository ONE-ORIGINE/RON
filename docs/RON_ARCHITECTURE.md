# RON Architecture

## Package map

```text
ron/core
    mathematical operators and RON-native core computation

ron/graph
    ports, synaptic routing, cortical bus and role governance

ron/runtime
    public runtime, packet contract, active_geodesic execution

ron/memory
    geodesic memory bank

ron/learning
    minimal RON-native learning path

ron/datasets
    RON-native synthetic data generation

ron/networks
    concrete network compositions around RON

ron/experiments
    reproducible comparison protocols and statistics

ron/hybrid
    optional hybridation policy

ron/bridges
    diagnostic bridges

ron/codecs
    boundary codecs
```

## Runtime flow

```text
input X/Y/Z
→ axes and energy
→ RON core
→ geodesic residual
→ Phi pressure
→ port routing
→ cortical bus
→ geodesic memory
→ active conductor
→ RONPacket
```

## Three network forms

```text
RON pure
    axes + energy → RON → RONPacket

RON + triaxial diagnostic
    axes + energy → triaxial lens → RON → RONPacket

RON hybrid tri-family
    boundary codec → triaxial diagnostic → RON → RONPacket
```

In every form, RON remains the packet-producing decision system.
