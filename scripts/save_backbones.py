from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import torch
import tyro
import lightning as L
import yaml

from src.image import model


class Ensemble(L.LightningModule):
    def __init__(self, modules: List[torch.nn.Module]):
        super().__init__()
        self.module_list = torch.nn.ModuleList([module.backbone for module in modules])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat([module(x) for module in self.module_list], -1)


def get_loss(checkpoint_path: Path) -> float:
    return float(checkpoint_path.stem.split("=")[-1])


def load_from_checkpoints_dir(checkpoints_dir: Path) -> Tuple[List[model.LightningModule], int]:
    modules = []
    for checkpoint_path in sorted(checkpoints_dir.rglob("*.ckpt")):
        print("<----", checkpoint_path)
        config_path = checkpoint_path.parent.parent / "config.yaml"
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        model_kwargs = config["model"]["init_args"]
        module = model.LightningModule.load_from_checkpoint(checkpoint_path, **model_kwargs).cpu()
        modules.append(module)
    return modules, config


def save(checkpoints_dir: Path, out_dir: Path):
    modules, config = load_from_checkpoints_dir(checkpoints_dir)
    ens = Ensemble(modules)

    model_path = out_dir / "model.pt"
    config_path = out_dir / "config.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)

    ens.to_torchscript(model_path)

    yaml.dump(config, open(config_path, "w"))


if __name__ == "__main__":
    tyro.cli(save)
