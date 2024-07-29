from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightning as L
import torch

from src import constants, utils
from src.sequence.constants import *


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: pd.DataFrame,
        meta: pd.DataFrame,
        feats: pd.DataFrame,
        aug: float = 0.2
    ):
        self.train = train
        self.train_index = train.index
        self.meta = meta
        self.feats = feats
        self.feats_index = set(feats.index)
        self.n_feats = len(feats.columns)
        self.n_meta = len(meta.columns)
        self.aug = aug
        self.levels = list(set([c.rsplit("_", 1)[0] for c in feats.columns]))

    def __len__(self):
        return len(self.train_index)

    def get_target(self, idx):
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def load_meta(self, study_id):
        return torch.tensor(self.meta.loc[study_id].values, dtype=torch.float)

    def load_feats(self, study_id, level):
        cols = self.feats.columns.str.contains(level)
        return torch.tensor(self.feats.loc[study_id].values[:, cols], dtype=torch.float)

    def __getitem__(self, index):
        idx = self.train_index[index]
        study_id, level = idx
        target = self.get_target(idx)
        X = self.load_feats(study_id, level)
        meta = self.load_meta(study_id)
        mask = torch.tensor([False] * len(X))
        return X, meta, mask, target


def load_feats(path: Path = constants.FEATS_PATH) -> pd.DataFrame:
    print(f"Loading feats DataFrame: {path}")
    df = pd.read_parquet("lightning_logs/version_1/preds.parquet")
    df = df.sort_values(["study_id", "series_id", "instance_number"])
    df = df.drop(columns=["series_id", "instance_number"])
    df = df.set_index(["study_id"])
    emb_dim = len(df.columns) // len(constants.LEVELS)
    new_cols = [f"{l}_{i}" for l in constants.LEVELS for i in range(emb_dim)]
    df.columns = new_cols
    print(df.head())
    return df


def load_meta(
    meta_path: Path = constants.META_PATH,
    desc_path: Path = constants.DESC_PATH
):
    meta = utils.load_meta(meta_path)
    desc = pd.read_csv(desc_path)

    df = pd.merge(desc, meta, on=["study_id", "series_id"])
    df = df.sort_values(["study_id", "series_id", "instance_number"])
    df = df.drop(columns=["instance_number", "series_id", "series_description"]).set_index(["study_id"]).sort_index()
    print(df.head())
    return df


def load_levels(levels_path):
    df = pd.read_parquet(levels_path)
    df = df.set_index(["study_id", "series_id", "instance_number"]).sort_index()
    df.index = df.index.droplevel([1, 2])
    return df


def load_this_train(train_path: Path = constants.TRAIN_PATH):
    pat = r"^(.*)_(l\d+_[l|s]\d+)$"
    df = utils.load_train(train_path)
    df[["condition", "level"]] = df.pop("condition_level").str.extractall(pat).reset_index()[[0, 1]]
    df["severity"] = df["severity"].map(lambda x: constants.SEVERITY2LABEL.get(x, -1))
    df = df.set_index(["study_id", "level", "condition"]).unstack(fill_value=-1)["severity"]
    return df


def collate_fn(data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
    X, meta, mask, targets = zip(*data)
    X = utils.pad_sequences(X, 0)
    meta = utils.pad_sequences(meta, 0)
    mask = utils.pad_sequences(mask, True)
    targets = utils.cat_dict_tensor(targets, utils.stack)
    return X, meta, mask, targets


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            feats_path: Path,
            meta_path: Path = constants.META_PATH,
            desc_path: Path = constants.DESC_PATH,
            train_path: Path = constants.TRAIN_PATH,
            aug: float = 0.2,
            n_splits: int = 5,
            this_split: int = 0,
            batch_size: int = 64,
            num_workers: int = 8,
        ):
        super().__init__()
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.aug = aug
        self.num_workers = num_workers
        self.train = load_this_train(train_path)
        self.meta = load_meta(meta_path, desc_path)
        self.feats = load_feats(feats_path)
        self.train_path = train_path

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split, self.train_path)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.meta, self.feats, aug=self.aug)
            self.val_ds = Dataset(val_df, self.meta, self.feats, aug=0.0)

        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/sequence.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, meta, mask, targets in dm.train_dataloader():
        print("---------------------------------------")
        print("X", (X.shape, X.dtype))
        print("meta", (meta.shape, meta.dtype))
        print("mask", (mask.shape, mask.dtype))
        print("targets", {k: (v.shape, v.dtype) for k, v in targets.items()})
        print("---------------------------------------")
