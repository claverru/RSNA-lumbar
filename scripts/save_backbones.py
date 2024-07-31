from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Tuple
import torch
import tyro
import lightning as L
import yaml

from src.keypoint import model as keypoint_model
from src.levels import model as levels_model


MODEL = {
    "keypoint": keypoint_model,
    "levels": levels_model
}



class Ensemble(L.LightningModule):
    def __init__(self, modules: List[torch.nn.Module]):
        super().__init__()
        self.module_list = torch.nn.ModuleList([module for module in modules])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        preds = [module(x) for module in self.module_list]
        outs = []
        for i in range(len(preds[0])):
            out = torch.concat([p[i] for p in preds])
            outs.append(out)
        return outs


def get_loss(ckpt_path: Path) -> float:
    return float(ckpt_path.stem.split("=")[-1])


def load_from_ckpts_dir(ckpts_dir: Path, model_type: str) -> Tuple[List[L.LightningModule], int]:
    modules = []
    for ckpt_path in sorted(ckpts_dir.rglob("*.ckpt")):
        print("<----", ckpt_path)
        config_path = ckpt_path.parent.parent / "config.yaml"
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        model_kwargs = config["model"]["init_args"]
        module = MODEL[model_type].LightningModule.load_from_checkpoint(ckpt_path, **model_kwargs).cpu()
        modules.append(module)
    return modules, config


def save(ckpts_dir: Path, out_dir: Path, model_type: Literal["keypoint", "levels"]):
    modules, config = load_from_ckpts_dir(ckpts_dir, model_type)
    ens = Ensemble(modules)

    img_size = config["data"]["init_args"]["img_size"]
    model_path = out_dir / "model.pt"
    config_path = out_dir / "config.yaml"

    example = torch.randn(2, 1, img_size, img_size)
    print([i.shape for i in ens(example)])


    out_dir.mkdir(parents=True, exist_ok=True)

    ts = ens.to_torchscript(model_path)

    print([i.shape for i in ts(example)])

    yaml.dump(config, open(config_path, "w"))


if __name__ == "__main__":
    tyro.cli(save)
