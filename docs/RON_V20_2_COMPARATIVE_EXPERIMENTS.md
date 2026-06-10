# RON V20.2 — Comparative Experiments

V20.2 removes the HTML interfaces and focuses on concrete comparative experiments and stronger visuals.

## Removed

```text
interfaces/
```

## Added

```text
ron/experiments/collision_compare.py
scripts/run_v20_2_collision_compare.py
scripts/validate_v20_2.py
visuals/generate_v20_2_graphics.py
```

## Compared variants

```text
pure
    RON core on CollisionWorld axes.

triaxial
    RON core + geodesic/triaxial regularization.

hybrid
    deterministic boundary codec + triaxial/geodesic regularization + RON core.
```

The classical part is still a boundary codec. It does not decide.

## Validate

```bash
python scripts/validate_v20_2.py
```

## Run comparison

```bash
python scripts/run_v20_2_collision_compare.py --steps 6 --batch-size 10 --cpu
```

## Generate graphics

```bash
python visuals/generate_v20_2_graphics.py
```
