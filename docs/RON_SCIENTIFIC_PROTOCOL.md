# RON Scientific Protocol

RON includes a reproducible protocol layer for comparing:

```text
pure
triaxial
hybrid
```

The protocol records:

```text
all_steps_metrics.csv
final_metrics_by_seed.csv
scientific_protocol_summary.json
```

Metrics include:

```text
total_loss
action_acc
cause_acc
consequence_acc
residual
release
geodesic_loss
```

Statistics include:

```text
mean
standard deviation
bootstrap confidence interval
paired differences
effect size
```

## Important output rule

Each protocol run should use a unique `--out-dir`.

Example:

```bash
python scripts/run_v20_3_scientific_protocol.py \
  --steps 200 \
  --batch-size 64 \
  --seeds 10 \
  --device auto \
  --out-dir runs/ron_protocol_200x64_10seeds_run001
```

## Six-hour oriented run

Use the dedicated script:

```bash
python scripts/run_v20_3_six_hour_protocol.py --device auto --profile balanced
```

Profiles:

```text
safe
balanced
strong
```

The goal is not only duration.  
The important points are:

```text
several seeds
same budget per variant
saved CSV/JSON
confidence intervals
effect comparison
packet analysis
graphics
```
