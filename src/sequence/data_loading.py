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
        feats_df: pd.DataFrame,
        aug: float = 0.2
    ):
        self.study_ids = study_ids
        self.train_df = train_df
        self.feats_df = feats_df
        self.feats_index = set(feats_df.index)
        self.fcols = [c for c in self.feats_df.columns if c.startswith("f")]
        self.aug = aug

    def __len__(self):
        return len(self.study_ids)

    def get_labels(self, study_id):
        d: dict = self.train_df.loc[study_id].set_index("condition_level")["severity"].to_dict()
        return {k: torch.tensor([constants.SEVERITY2LABEL.get(d.get(k), -1)]) for k in constants.CONDITION_LEVEL}

    def get_feats(self, study_id, description): # (L, 1793, 5)
        index = (study_id, description)
        if index in self.feats_index:
            chunk: pd.DataFrame = self.feats_df.loc[index]
            unique_si = chunk[["series_id", "instance_number"]].drop_duplicates().to_numpy()
            unique_siv = [tuple(u) + (i, ) for u in unique_si for i in range(5)]

            i_cols = ["series_id", "instance_number", "version"]
            chunk = chunk.set_index(i_cols).reindex(unique_siv, fill_value=0.0)
            chunk = chunk.reset_index()
            chunk["instance_number"] = chunk.groupby("series_id")["instance_number"].apply(lambda x: x/x.max()).reset_index(drop=True)
            values = chunk[["instance_number"] + self.fcols].values.astype(np.float32)
            values = values.reshape(-1, 5, values.shape[-1]).transpose(0, 2, 1)
            return values

        else:
            return np.zeros((1, len(self.fcols) + 1, 5), dtype=np.float32)

    def do_aug(self, X):
        for k, v in X.items():
            keep_mask = torch.rand(v.shape[0], 1, v.shape[-1]) > self.aug
            X[k] = X[k] * keep_mask
        return X

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        target = self.get_labels(study_id)
        X = {}
        for description in constants.DESCRIPTIONS:
            X[description] = torch.tensor(self.get_feats(study_id, description))

        if self.aug > 0:
            X = self.do_aug(X)

        return X, target


def load_feats(path: Path = constants.FEATS_PATH) -> pd.DataFrame:
    print("Loading feats DataFrame")
    df = pd.read_parquet(path)
    index_cols = ["study_id", "series_description"]
    sort_cols = ["series_id", "instance_number", "version"]
    df = df.set_index(index_cols).sort_values(sort_cols).sort_index()
    return df


def pad_sequences(sequences):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def collate_fn(data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
    Xs, targets = zip(*data)
    Xs = utils.cat_preds(Xs, pad_sequences)
    targets = utils.cat_preds(targets)
    return Xs, targets


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            feats_path: Path = constants.FEATS_PATH,
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
        self.feats_df = load_feats(feats_path)
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
            self.train_ds = Dataset(train_study_ids, self.train_df, self.feats_df, aug=0.2)
            self.val_ds = Dataset(val_study_ids, self.train_df, self.feats_df, aug=0)

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
    dm = DataModule()
    dm.setup("fit")
    for x, y in dm.train_dataloader():
        print("---------------------------------------")
        print({k: (v.shape, v.dtype) for k, v in x.items()})
        print({k: (v.shape, v.dtype) for k, v in y.items()})
