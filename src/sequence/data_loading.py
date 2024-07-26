from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightning as L
from sklearn.model_selection import StratifiedKFold
import torch

from src import constants, utils
from src.sequence.constants import *


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        study_ids: List[int],
        train: pd.DataFrame,
        meta: pd.DataFrame,
        feats: pd.DataFrame,
        aug: float = 0.2
    ):
        self.study_ids = study_ids
        self.train = train
        self.meta = meta
        self.feats = feats
        self.feats_index = set(feats.index)
        self.n_feats = len(feats.columns)
        self.n_meta = len(meta.columns)
        self.aug = aug
        self.levels = list(set([c.rsplit("_", 1)[0] for c in feats.columns]))

    def __len__(self):
        return len(self.study_ids)

    def get_target(self, study_id):
        return self.train.loc[study_id].apply(torch.tensor).to_dict()

    # def do_aug(self, values: np.ndarray) -> np.ndarray:
    #     mask = np.random.rand(values.shape[0], N_IMG_ENSEMBLES, 1) > self.aug
    #     mask = mask.repeat(len(self.fids) // N_IMG_ENSEMBLES, -1).reshape(mask.shape[0], -1)
    #     values[:, self.fids] *= mask
    #     return values

    def get_feats(self, study_id, description):
        index = (study_id, description)
        levels = ["right", "left"] if "Axial" in description else [c for c in self.levels if c not in ["right", "left"]]
        n = len(levels)
        cols = self.feats.columns.str.contains("|".join(levels))

        if index in self.feats_index:
            values = self.feats.loc[index].values.astype(np.float32)
        else:
            values = np.zeros((1, self.n_feats), dtype=np.float32)

        values = values[:, cols].reshape(values.shape[0], n, -1)

        return values

    def load_meta(self, study_id, description):
        index = (study_id, description)
        if index in self.feats_index:
            values = self.meta.loc[index].values.astype(np.float32)
        else:
            values = np.zeros((1, self.n_meta), dtype=np.float32)
        return values

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        target = self.get_target(study_id)
        X = {}
        meta = {}
        mask = {}
        for description in constants.DESCRIPTIONS:
            X[description] = torch.tensor(self.get_feats(study_id, description))
            mask[description] = torch.tensor([False] * len(X[description]))
            meta[description] = torch.tensor(self.load_meta(study_id, description))

        return X, meta, mask, target


def load_feats(path: Path = constants.FEATS_PATH, desc_path: Path = constants.DESC_PATH) -> pd.DataFrame:
    print(f"Loading feats DataFrame: {path}")
    df = pd.read_parquet(path)
    desc = utils.load_desc(desc_path)
    df = df.merge(desc, how="inner", on=["study_id", "series_id"])
    df = df.sort_values(["study_id", "series_id", "instance_number"])
    df = df.drop(columns=["instance_number", "series_id"]).set_index(["study_id", "series_description"]).sort_index()
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
    df = df.drop(columns=["instance_number", "series_id"]).set_index(["study_id", "series_description"]).sort_index()
    print(df.head())
    return df


def collate_fn(data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
    X, meta, mask, targets = zip(*data)
    X = utils.cat_dict_tensor(X, lambda x: utils.pad_sequences(x, 0))
    meta = utils.cat_dict_tensor(meta, lambda x: utils.pad_sequences(x, 0))
    mask = utils.cat_dict_tensor(mask, lambda x: utils.pad_sequences(x, True))
    targets = utils.cat_dict_tensor(targets, utils.stack)
    return X, meta, mask, targets


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            feats_path: Path = constants.FEATS_PATH,
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
        self.train = utils.load_train_flatten(train_path)
        self.meta = load_meta(meta_path, desc_path)
        self.feats = load_feats(feats_path, desc_path)

    def prepare_data(self):
        pass

    def split(self) -> Tuple[List[int], List[int]]:
        strats = self.train.astype(str).sum(1)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats)):
            if i == self.this_split:
                break
        return list(self.train.index[train_ids]), list(self.train.index[val_ids])

    def setup(self, stage: str):
        if stage == "fit":
            train_study_ids, val_study_ids = self.split()
            self.train_ds = Dataset(train_study_ids, self.train, self.meta, self.feats, aug=self.aug)
            self.val_ds = Dataset(val_study_ids, self.train, self.meta, self.feats, aug=0.0)

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
        print("X", {k: (v.shape, v.dtype) for k, v in X.items()})
        print("meta", {k: (v.shape, v.dtype) for k, v in meta.items()})
        print("mask", {k: (v.shape, v.dtype) for k, v in mask.items()})
        print("targets", {k: (v.shape, v.dtype) for k, v in targets.items()})
        print("---------------------------------------")
