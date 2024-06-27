from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightning as L
from sklearn.model_selection import StratifiedGroupKFold
import torch

from src import constants, utils


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        study_ids: List[int],
        train_df: pd.DataFrame,
        df: pd.DataFrame,
        aug: float = 0.2
    ):
        self.study_ids = study_ids
        self.train_df = train_df
        self.df = df
        self.feats_index = set(df.index)
        self.n_feats = len(df.columns)
        self.fids = [i for i, c in enumerate(df.columns) if c.startswith("f") and "_v" in c]
        self.aug = aug

    def __len__(self):
        return len(self.study_ids)

    def get_labels(self, study_id):
        d: dict = self.train_df.loc[study_id].set_index("condition_level")["severity"].to_dict()
        return {k: torch.tensor(constants.SEVERITY2LABEL.get(d.get(k), -1)) for k in constants.CONDITION_LEVEL}

    def do_aug(self, values: np.ndarray) -> np.ndarray:
        mask = np.random.rand(values.shape[0], 5, 1) > self.aug
        mask = mask.repeat(len(self.fids) // 5, -1).reshape(mask.shape[0], -1)
        values[:, self.fids] *= mask
        return values

    def get_feats(self, study_id, description): # (L, 1793, 5)
        index = (study_id, description)
        if index in self.feats_index:
            values = self.df.loc[index].values.astype(np.float32)
            values = self.do_aug(values)
        else:
            values = np.zeros((1, self.n_feats), dtype=np.float32)
        return values

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        target = self.get_labels(study_id)
        X = {}
        for description in constants.DESCRIPTIONS:
            X[description] = torch.tensor(self.get_feats(study_id, description))
        return X, target


def load_feats(path: Path = constants.FEATS_PATH) -> pd.DataFrame:
    print("Loading feats DataFrame")
    df = pd.read_parquet(path).reset_index()
    return df


def merge_dfs(
    feats_path: Path = constants.FEATS_PATH,
    meta_path: Path = constants.META_PATH,
    desc_path: Path = constants.DESC_PATH
):
    feats = load_feats(feats_path)
    meta = utils.load_meta(meta_path)
    desc = pd.read_csv(desc_path)

    df = pd.merge(desc, meta, on=["study_id", "series_id"])
    df = df.sort_values(["study_id", "series_id", "instance_number"])
    df = pd.merge(df, feats, on=["series_id", "instance_number"])
    df = df.drop(columns=["instance_number", "series_id"]).set_index(["study_id", "series_description"]).sort_index()
    return df


def collate_fn(data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
    Xs, targets = zip(*data)
    Xs = utils.cat_dict_tensor(Xs, utils.pad_sequences)
    targets = utils.cat_dict_tensor(targets, utils.stack)
    return Xs, targets


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            feats_path: Path = constants.FEATS_PATH,
            meta_path: Path = constants.META_PATH,
            desc_path: Path = constants.DESC_PATH,
            train_path: Path = constants.TRAIN_PATH,
            n_splits: int = 5,
            this_split: int = 0,
            batch_size: int = 64,
            num_workers: int = 8,
        ):
        super().__init__()
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df = merge_dfs(feats_path, meta_path, desc_path)
        self.train_df = utils.load_train(train_path).set_index("study_id").sort_index()

    def prepare_data(self):
        pass

    def split(self) -> Tuple[List[int], List[int]]:
        strats = self.train_df["condition_level"] + "_" + self.train_df["severity"]
        groups = self.train_df.reset_index()["study_id"]
        skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
            if i == self.this_split:
                break
        return list(set(groups[train_ids])), list(set(groups[val_ids]))

    def setup(self, stage: str):
        if stage == "fit":
            train_study_ids, val_study_ids = self.split()
            self.train_ds = Dataset(train_study_ids, self.train_df, self.df, aug=0.2)
            self.val_ds = Dataset(val_study_ids, self.train_df, self.df, aug=0)

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
    dm = DataModule(num_workers=1)
    dm.setup("fit")
    for x, y in dm.train_dataloader():
        print("---------------------------------------")
        print({k: (v.shape, v.dtype) for k, v in x.items()})
        print({k: (v.shape, v.dtype) for k, v in y.items()})
