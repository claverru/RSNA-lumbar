from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import torch
import tyro
import lightning as L
import yaml

from src.study import model, constants as study_constants


class Ensemble(L.LightningModule):
    def __init__(self, modules: List[torch.nn.Module]):
        super().__init__()
        self.module_list = torch.nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result_list: List[Dict[str, torch.Tensor]] = [module(x) for module in self.module_list]
        result_dict: Dict[str, torch.Tensor] = {}
        for k in result_list[0]:
            init = torch.zeros_like(result_list[0][k])
            result_dict[k] = sum((result[k] for result in result_list), init) / len(result_list)
        return result_dict


def load_from_checkpoints_dir(checkpoints_dir: Path) -> Tuple[List[model.LightningModule], int]:
    modules = []
    for checkpoint_path in checkpoints_dir.rglob("*.ckpt"):
        print("<----", checkpoint_path)
        hparams_path = checkpoint_path.parent.parent / "config.yaml"
        hparams = yaml.load(open(hparams_path), Loader=yaml.FullLoader)
        model_kwargs = hparams["model"]["init_args"]
        img_size = hparams["data"]["init_args"]["img_size"]
        module = model.LightningModule.load_from_checkpoint(checkpoint_path, **model_kwargs).cpu()
        modules.append(module)
    return modules, img_size


def save(checkpoints_dir: Path, out_path: Path):
    modules, img_size = load_from_checkpoints_dir(checkpoints_dir)
    print("---->", out_path)
    ens = Ensemble(modules)
    x = torch.randn(1, 3, study_constants.IMGS_PER_DESC, img_size, img_size)
    ens.to_torchscript(out_path, "trace", x, strict=False)


if __name__ == "__main__":
    tyro.cli(save)
