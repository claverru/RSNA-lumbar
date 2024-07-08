from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import torch
import tyro
import lightning as L
import numpy as np
import yaml

from src.sequence import model, data_loading, constants


class Ensemble(L.LightningModule):
    def __init__(self, modules: List[torch.nn.Module]):
        super().__init__()
        self.module_list = torch.nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        result_list: List[Dict[str, torch.Tensor]] = [module(x, desc) for module in self.module_list]
        result_dict: Dict[str, torch.Tensor] = {}
        for k in result_list[0]:
            init = torch.zeros_like(result_list[0][k])
            result_dict[k] = sum((result[k] for result in result_list), init) / len(result_list)
        return result_dict


def get_loss(checkpoint_path: Path) -> float:
    return float(checkpoint_path.stem.split("=")[-1])

def load_from_checkpoints_dir(checkpoints_dir: Path) -> Tuple[List[model.LightningModule], int]:
    modules = []
    losses = []
    for checkpoint_path in checkpoints_dir.rglob("*.ckpt"):
        print("<----", checkpoint_path)
        losses.append(get_loss(checkpoint_path))
        config_path = checkpoint_path.parent.parent / "config.yaml"
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        model_kwargs = config["model"]["init_args"]
        module = model.LightningModule.load_from_checkpoint(checkpoint_path, **model_kwargs).cpu()
        modules.append(module)
    return modules, config, losses


def save(checkpoints_dir: Path, out_dir: Path):
    modules, config, losses = load_from_checkpoints_dir(checkpoints_dir)
    ens = Ensemble(modules)

    model_path = out_dir / "model.pt"
    config_path = out_dir / "config.yaml"
    metrics_path = out_dir / "metrics.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(2, 3, constants.INPUT_SIZE - constants.D)
    desc = torch.randint(0, 4, (2, 3))
    example_input = (x, desc)
    ts = ens.to_torchscript(model_path, "trace", example_inputs=example_input, strict=False)

    print({k: v.shape for k, v in ts(*example_input).items()})

    x = torch.randn(4, 5, constants.INPUT_SIZE - constants.D)
    desc = torch.randint(0, 4, (4, 5))
    example_input = (x, desc)
    print({k: v.shape for k, v in ts(*example_input).items()})

    yaml.dump(config, open(config_path, "w"))
    metrics = {"losses": losses, "mean": float(np.mean(losses)), "std": float(np.std(losses))}
    yaml.dump(metrics, open(metrics_path, "w"))
    print(metrics)


if __name__ == "__main__":
    tyro.cli(save)
