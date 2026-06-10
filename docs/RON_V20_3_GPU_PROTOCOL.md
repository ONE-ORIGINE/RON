# RON V20.3.1 — GPU Protocol Patch

This patch fixes the protocol device behavior.

Previous behavior:

```text
scripts/run_v20_3_scientific_protocol.py always used device="cpu"
```

New behavior:

```text
--device auto   uses CUDA if available, otherwise CPU
--device cuda   forces CUDA and fails if unavailable
--device cpu    forces CPU
--cpu           legacy alias, forces CPU
```

## Check device

```bash
python scripts/check_device.py
```

## Run on GPU

```bash
python scripts/run_v20_3_scientific_protocol.py --steps 200 --batch-size 64 --seeds 10 --device cuda
```

## Six-hour oriented run

Safe GPU:

```bash
python scripts/run_v20_3_six_hour_protocol.py --device cuda --profile safe
```

Balanced GPU:

```bash
python scripts/run_v20_3_six_hour_protocol.py --device cuda --profile balanced
```

Strong GPU:

```bash
python scripts/run_v20_3_six_hour_protocol.py --device cuda --profile strong
```

## Important

Do not add `--cpu` if you want GPU.

If `torch.cuda.is_available()` is false, install a CUDA-enabled PyTorch build compatible with your GPU and NVIDIA driver.
