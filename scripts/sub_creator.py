import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

e2e_progress = json.load(open("e2e_progress.json"))
checkpoints = e2e_progress["ckpt_paths"]

sub_folder = Path("checkpoints")
shutil.rmtree(str(sub_folder), ignore_errors=True)
sub_folder.mkdir(exist_ok=True)


@dataclass
class Model:
    model_key: str
    checkpoint: str

    @property
    def model_name(self):
        return "_".join(self.model_key.split("_")[:-1])

    @property
    def main_model(self):
        model_name = self.model_name
        if model_name == "keypoints":
            return "keypoints"
        elif model_name == "levels":
            return "levels"
        elif model_name == "patch":
            return "patch"
        elif model_name == "sequence":
            return "sequence"
        elif model_name == "finetune_sequence":
            return "sequence"

    @property
    def fold_id(self):
        return int(self.model_key.split("_")[-1])

    @property
    def score(self):
        last_part = self.checkpoint.split("=")[-1]
        score = ".".join(last_part.split(".")[:-1])
        return float(score)

    @property
    def checkpoint_dir(self):
        return Path(self.checkpoint).parent.parent

    @property
    def ckpt_name(self):
        return Path(self.checkpoint).name

    def __repr__(self):
        return f"{self.main_model=}, {self.fold_id=}, {self.score=}"


class ModelScorer:
    def __init__(self, models):
        self.models = models
        self.name = self.get_name()

    def get_name(self):
        names = set([model.main_model for model in self.models])
        if len(names) != 1:
            raise ValueError("All models must be of the same type")
        return names.pop()

    @property
    def best_models(self):
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

    def to_submission(self, sub_folder):
        best_models = self.best_models
        first_fold = best_models[0].checkpoint_dir
        # copy all the file from the first fold folder (but not the checkpoints folder)
        directory = sub_folder / self.name
        directory.mkdir(exist_ok=True)
        for file in first_fold.iterdir():
            if file.name != "checkpoints":
                shutil.copy(file, directory / file.name)

        checkpoints_dir = directory / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        # copy the checkpoints
        for _, model in best_models.items():
            shutil.copy(model.checkpoint, checkpoints_dir / model.ckpt_name)


def create_scorers(models):
    m = defaultdict(list)
    for model in models:
        m[model.main_model].append(model)
    return [ModelScorer(models) for models in m.values()]


models = [Model(model_key=model, checkpoint=ckpt_path) for model, ckpt_path in checkpoints.items()]
scorers = create_scorers(models)
for scorer in scorers:
    print(scorer)
    if scorer.name == "patch":
        print(f"Skipping {scorer.name}")
        continue
    scorer.to_submission(sub_folder)
