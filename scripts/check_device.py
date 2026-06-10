from __future__ import annotations
import json
import torch

def main():
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "selected_auto_device": "cuda" if torch.cuda.is_available() else "cpu",
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 3),
                "capability": f"{props.major}.{props.minor}",
            })
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
