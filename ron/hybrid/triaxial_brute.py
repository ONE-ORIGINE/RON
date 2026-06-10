from __future__ import annotations
from pathlib import Path
import importlib.util

class OptionalTriaxialBrute:
    """Optional wrapper for the user-provided brute triaxial neuron.

    The brute triaxial implementation is not part of the RON core. It can be
    loaded from an external file such as `/mnt/data/triaxial_neuron.py`.
    """
    def __init__(self, source_path: str | None = None):
        self.source_path = source_path
        self.module = None

    def available(self) -> bool:
        return self.source_path is not None and Path(self.source_path).exists()

    def load(self):
        if not self.available():
            raise FileNotFoundError("No triaxial brute source file was provided.")
        spec = importlib.util.spec_from_file_location("external_triaxial_brute", self.source_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.module = module
        return module

    def summary(self) -> dict:
        if self.module is None:
            return {"loaded": False, "source_path": self.source_path}
        return {
            "loaded": True,
            "source_path": self.source_path,
            "has_TriaxialNeuron": hasattr(self.module, "TriaxialNeuron"),
            "has_TriaxialNetwork": hasattr(self.module, "TriaxialNetwork"),
            "has_TriaxialTrainer": hasattr(self.module, "TriaxialTrainer"),
            "has_AxisMask": hasattr(self.module, "AxisMask"),
        }
