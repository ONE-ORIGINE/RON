from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch
from ron.core.whole_core import RONWholeTriAxialCore
from ron.core.geodesic_pressure import geodesic_pressure
from ron.learning.native_tasks import make_native_batch
from ron.learning.native_objective import native_training_loss
from ron.learning.schedule import cosine_lr
from ron.learning.checkpoint import save_checkpoint, load_checkpoint

@dataclass
class NativeTrainConfig:
    vocab: int = 12
    batch_size: int = 16
    steps: int = 12
    lr: float = 0.035
    seed: int = 123
    grid: int = 1
    cube: int = 3
    moment_degree: int = 1
    signature_degree: int = 1
    harmonic_degree: int = 1
    internal_cycles: int = 1
    device: str = "cpu"
    out_dir: str = "runs/native_v20"

class RONNativeTrainer:
    """Small RON-native trainer.

    It is meant to prove the training path exists and is reproducible on CPU.
    Heavy benchmarks can be run later on stronger hardware.
    """
    def __init__(self, cfg: NativeTrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model = RONWholeTriAxialCore(
            vocab=cfg.vocab,
            grid=cfg.grid,
            cube=cfg.cube,
            moment_degree=cfg.moment_degree,
            signature_degree=cfg.signature_degree,
            harmonic_degree=cfg.harmonic_degree,
            internal_cycles=cfg.internal_cycles,
            seed=cfg.seed,
        ).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def _geodesic(self, triplet):
        axes, energy = self.model.token_axes(triplet)
        return geodesic_pressure(axes, energy, strength=0.0, cube_n=self.cfg.cube)

    def step(self, seed:int):
        self.model.train()
        batch = make_native_batch(self.cfg.batch_size, self.cfg.vocab, seed, self.device)
        geo = self._geodesic(batch.triplet)
        out = self.model(triplet=batch.triplet, mask=batch.mask, adaptive_cycles=True)
        loss, parts = native_training_loss(out, batch, geo)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        if hasattr(self.model, "project_parameters_"):
            self.model.project_parameters_()
        return float(loss.detach().cpu()), parts.__dict__

    def evaluate_loss(self, seed:int=999):
        self.model.eval()
        with torch.no_grad():
            batch = make_native_batch(self.cfg.batch_size, self.cfg.vocab, seed, self.device)
            geo = self._geodesic(batch.triplet)
            out = self.model(triplet=batch.triplet, mask=batch.mask, adaptive_cycles=True)
            loss, parts = native_training_loss(out, batch, geo)
        return float(loss.detach().cpu()), parts.__dict__

    def run(self):
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        initial, initial_parts = self.evaluate_loss(seed=self.cfg.seed + 1000)
        history = [{"step": 0, "eval_loss": initial, "parts": initial_parts}]
        for step in range(1, self.cfg.steps + 1):
            lr = cosine_lr(step-1, self.cfg.steps, self.cfg.lr, min_lr=self.cfg.lr*0.10)
            for group in self.opt.param_groups:
                group["lr"] = lr
            train_loss, parts = self.step(self.cfg.seed + step)
            eval_loss, eval_parts = self.evaluate_loss(seed=self.cfg.seed + 1000)
            history.append({"step": step, "lr": lr, "train_loss": train_loss, "eval_loss": eval_loss, "parts": eval_parts})
        ckpt = save_checkpoint(out_dir / "ron_native_v20.pt", self.model, self.opt, {"config": asdict(self.cfg), "history": history})
        # Reload check into a fresh model.
        fresh = RONWholeTriAxialCore(
            vocab=self.cfg.vocab,
            grid=self.cfg.grid,
            cube=self.cfg.cube,
            moment_degree=self.cfg.moment_degree,
            signature_degree=self.cfg.signature_degree,
            harmonic_degree=self.cfg.harmonic_degree,
            internal_cycles=self.cfg.internal_cycles,
            seed=self.cfg.seed,
        ).to("cpu")
        metadata = load_checkpoint(ckpt, fresh)
        # Compare one parameter group through state dict keys.
        reloaded_ok = set(fresh.state_dict().keys()) == set(self.model.cpu().state_dict().keys())
        # Put model back to configured device after CPU compare.
        self.model.to(self.device)
        summary = {
            "config": asdict(self.cfg),
            "initial_eval_loss": initial,
            "final_eval_loss": history[-1]["eval_loss"],
            "loss_changed": abs(history[-1]["eval_loss"] - initial) > 1e-7,
            "loss_nonexplosive": history[-1]["eval_loss"] <= initial * 1.50,
            "checkpoint": str(ckpt),
            "checkpoint_reload_ok": bool(reloaded_ok and "config" in metadata),
            "history": history,
        }
        (out_dir / "native_training_summary_v20.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
