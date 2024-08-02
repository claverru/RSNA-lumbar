from typing import List, Optional, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightning as L

from src import constants, utils


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: Optional[pd.DataFrame],
        df: pd.DataFrame,
    ):
        self.train = train
        self.df = df
        self.index = train.index if train is not None else df.index

    def __len__(self):
        return len(self.index)

    def get_target(self, study_id):
        if self.train is None:
            return None
        return self.train.loc[study_id].apply(torch.tensor).to_dict()

    def get_x(self, study_id):
        return torch.tensor(self.df.loc[study_id].values, dtype=torch.float)

    def __getitem__(self, index):
        study_id = self.index[index]
        x = self.get_x(study_id)
        target = self.get_target(study_id)
        if target is not None:
            return x, target
        else:
            return x


def load_df(probas_path: Path = constants.PROBAS_PATH):
    df = pd.read_parquet(probas_path)
    df = df.melt(id_vars=["study_id", "level"])
    df["level_condition_severity"] = df.pop("level") + "_" + df.pop("variable")
    df = df.pivot(index="study_id", columns="level_condition_severity")["value"]
    return df


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: Path = constants.TRAIN_PATH,
        probas_path: Path = constants.PROBAS_PATH,
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
        self.df = load_df(Path(probas_path))
        self.train = utils.load_train(train_path)
        self.train_path = train_path

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.df)
            self.val_ds = Dataset(val_df, self.df)

        if stage == "test":
            pass

        if stage == "predict":
            self.predict_ds = Dataset(None, self.df)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/probas.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, target in dm.train_dataloader():
        print()
        print(X.shape, X.dtype)
        print({k: (v.shape, v.dtype) for k, v in target.items()})
        print()
        input()
