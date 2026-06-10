# RON V20.3 — Scientific Protocol

V20.3 adds the layer needed for a serious, falsifiable, reproducible experimental proof.

## What V20.3 can prove

It can provide strong prototype-level evidence:

```text
multi-seed comparison
variant comparison
ablation-style metrics
packet analysis
bootstrap confidence intervals
effect sizes
reproducible CSV/JSON outputs
publication graphics
```

## What V20.3 cannot claim alone

It is not a universal mathematical proof and not a final peer-reviewed benchmark.

For that, RON needs:

```text
large real datasets
external replication
longer runs
independent baselines
peer review
hardware diversity
```

## Six-hour protocol

A 6-hour run can be a strong local scientific proof if it uses:

```bash
python scripts/run_v20_3_scientific_protocol.py --steps 100 --batch-size 32 --seeds 10 --cpu
```

If the machine is strong enough:

```bash
python scripts/run_v20_3_scientific_protocol.py --steps 200 --batch-size 64 --seeds 10 --cpu
```

The important part is not only duration. It is:

```text
several seeds
same budget per variant
saved CSV
saved JSON
confidence intervals
packet analysis
graphics
```

## Validate quickly

```bash
python scripts/validate_v20_3.py
```

## Generate publication graphics

```bash
python visuals/generate_v20_3_publication_graphics.py
```
