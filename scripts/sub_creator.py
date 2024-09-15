import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import yaml

MODEL2MAINMODEL = {
    "keypoints": "keypoints",
    "levels": "levels",
    "patch": "patch",
    "sequence": "sequence",
    "finetune_sequence": "sequence",
}


@dataclass
class Model:
    model_key: str
    checkpoint_path: str

    @property
    def model_name(self):
        return "_".join(self.model_key.split("_")[:-1])

    @property
    def main_model(self):
        return MODEL2MAINMODEL[self.model_name]

    @property
    def fold_id(self):
        return int(self.model_key.split("_")[-1])

    @property
    def score(self):
        last_part = self.checkpoint_path.split("=")[-1]
        score = ".".join(last_part.split(".")[:-1])
        return float(score)

    @property
    def checkpoint_dir(self):
        return Path(self.checkpoint_path).parent.parent

    @property
    def ckpt_name(self):
        return Path(self.checkpoint_path).name

    @property
    def config_path(self):
        return self.checkpoint_dir / "config.yaml"

    def __repr__(self):
        return f"{self.main_model=}, {self.fold_id=}, {self.score=}"


class ModelScorer:
    def __init__(self, models: List[Model]):
        self.models = models
        self.name = self.get_name()

    def get_name(self):
        names = set([model.main_model for model in self.models])
        if len(names) != 1:
            raise ValueError("All models must be of the same type")
        return names.pop()

    @property
    def best_models(self) -> Dict[str, Model]:
        max_fold = max([model.fold_id for model in self.models])
        best_models = {fold_id: None for fold_id in range(max_fold + 1)}
        for model in self.models:
            if best_models[model.fold_id] is None or model.score < best_models[model.fold_id].score:
                best_models[model.fold_id] = model
        return best_models

    @property
    def average_score(self):
        best_models = self.best_models
        return sum([model.score for model in best_models.values()]) / len(best_models)

    def __repr__(self):
        return f"{self.name=}, {self.average_score=}"

    def copy_to_sub_folder(self, sub_folder: Path):
        best_models = self.best_models
        directory = sub_folder / self.name
        directory.mkdir(exist_ok=True)

        # copy config
        new_config = merge_configs([model.config_path for model in best_models.values()])
        yaml.dump(new_config, open(directory / "config.yaml", "w"))

        # copy the checkpoints
        new_checkpoint = merge_checkpoints([model.checkpoint_path for model in best_models.values()])
        torch.save(new_checkpoint, directory / f"{self.average_score:.5f}.ckpt")


def pop_things(d: dict):
    if "lr_scheduler" in d:
        d.pop("lr_scheduler")
    if "pretrained" in d:
        d["pretrained"] = False
    if "ckpt_path" in d:
        d.pop("ckpt_path")


def clean_config(config):
    pop_things(config["model"]["init_args"])
    if "backbone" in config["model"]["init_args"]:
        pop_things(config["model"]["init_args"]["backbone"]["init_args"])
    config["trainer"]["callbacks"] = None
    config["trainer"]["logger"] = None
    return config


def merge_configs(config_paths):
    configs = [yaml.load(open(config_path), Loader=yaml.FullLoader) for config_path in config_paths]
    configs = [clean_config(config) for config in configs]
    config = configs[0]
    config["model"] = {
        "class_path": "src.model.Ensemble",
        "init_args": {"models": [config["model"] for config in configs]},
    }
    return config


def merge_checkpoints(checkpoints_paths):
    checkpoints = [torch.load(ckpt_path) for ckpt_path in checkpoints_paths]
    new_checkpoint = {k: v for k, v in checkpoints[0].items() if k != "state_dict"}
    new_checkpoint["state_dict"] = {}
    for i, checkpoint in enumerate(checkpoints):
        for k, state_dict in checkpoint["state_dict"].items():
            new_checkpoint["state_dict"][f"models.{i}.{k}"] = state_dict
    return new_checkpoint


def create_scorers(models):
    m = defaultdict(list)
    for model in models:
        m[model.main_model].append(model)
    return [ModelScorer(models) for models in m.values()]


def clean_and_save_checkpoints(checkpoints: Dict[str, str], sub_folder: Path):
    checkpoints = {k: v.split("/")[-1] for k, v in checkpoints.items()}
    yaml.dump(checkpoints, open(sub_folder / "checkpoints.yaml", "w"))


def reset_sub_folder():
    sub_folder = Path("checkpoints")
    shutil.rmtree(str(sub_folder), ignore_errors=True)
    sub_folder.mkdir(exist_ok=True)
    return sub_folder


if __name__ == "__main__":
    checkpoints = json.load(open("e2e_progress.json"))["ckpt_paths"]

    sub_folder = reset_sub_folder()
    clean_and_save_checkpoints(checkpoints, sub_folder)

    models = [Model(model_key=model, checkpoint_path=checkpoint_path) for model, checkpoint_path in checkpoints.items()]
    scorers = create_scorers(models)
    for scorer in scorers:
        print(scorer)
        if scorer.name == "patch":
            print(f"Skipping {scorer.name}")
            continue
        scorer.copy_to_sub_folder(sub_folder)
