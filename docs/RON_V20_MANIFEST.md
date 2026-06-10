# RON V20 Manifest

RON V20 is the final clean repository.

## Package structure

```text
ron/
  core/
  graph/
  runtime/
  memory/
  learning/
  tasks/
  codecs/
  bridges/
  hybrid/

scripts/
examples/
tests/
docs/
README.md
pyproject.toml
```

## Default runtime

```text
RONSystem(mode="ron_pure", profile="active_geodesic")
```

## Official output

```text
RONPacket
```

## Hybridation

```text
ron/hybrid/
```

Included in the main repository, but restricted by policy:

```text
RON = core and decision
triaxial brute = optional diagnostic / teacher
classical = optional boundary codec
```

## Validation targets

```text
runtime_ok = true
packet_contract_ok = true
hybrid_folder_present = true
hybrid_rejects_hidden_classical_decider = true
active_pressure_strength_positive = true
active_memory_writes = true
no_classical_disguise = true
```
