from pathlib import Path
from typing import Dict, List, Tuple


import lightning as L
import numpy as np
import pandas as pd
import torch
import tqdm
import tyro
import yaml

from src.image import data_loading, model
from src import constants, utils



def parse_preds(preds_dict: Dict[str, torch.Tensor], index2id: Dict[int, Tuple[int, int, int]]) -> pd.DataFrame:
    columns = sum([(p + "-Normal/Mild", p + "-Moderate", p + "-Severe") for p in preds_dict], ())
    data = np.concatenate(list(preds_dict.values()), 1)
    index = pd.MultiIndex.from_tuples(list(index2id.values()), names=["study_id", "series_id", "instance_number"])
    preds_df = pd.DataFrame(data=data, index=index, columns=columns)
    df = preds_df.melt(ignore_index=False)
    df[["condition_level", "severity"]] = df.pop("variable").str.split("-", expand=True)

    new_index = ["study_id", "series_id", "instance_number", "condition_level", "severity"]
    df = df.reset_index().set_index(new_index).unstack()["value"]
    return df


def main(ckpt_dir: Path):
    config_path = ckpt_dir / "config.yaml"
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    L.pytorch.seed_everything(config["seed_everything"])

    ckpt_path = next(ckpt_dir.rglob("*.ckpt"))
    model_kwargs = config["model"]["init_args"]
    L_module = model.LightningModule.load_from_checkpoint(ckpt_path, **model_kwargs).cuda()

    datamodule_kwargs = config["data"]["init_args"]
    L_datamodule = data_loading.DataModule(**datamodule_kwargs)
    L_datamodule.setup("predict")

    preds_list = []
    target_list = []
    for x, y in tqdm.tqdm(L_datamodule.predict_dataloader(), desc="Predicting"):
        with torch.no_grad():
            x = x.to(L_module.device)
            preds = L_module(x)
        preds_list.append(preds)
        target_list.append(y)

    preds_dict = utils.cat_preds(preds_list)
    preds_dict = {k: v.cpu().numpy() for k, v in preds_dict.items()}

    preds_df = parse_preds(preds_dict, L_datamodule.predict_ds.index2id)
    preds_df.to_csv(ckpt_dir / constants.VAL_PREDS_NAME)


if __name__ == "__main__":
    tyro.cli(main)
