from pathlib import Path
import random
from typing import Dict, Optional, Tuple

from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
import torchvision
import torchvision.transforms.functional

from src import constants, utils


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        train: bool = False,
        transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.train = train
        self.transforms = transforms
        self.index2id = {i: idx for i, idx in enumerate(self.df.index.unique())}

    def __len__(self):
        return len(self.index2id)

    @staticmethod
    def hflip(img: torch.Tensor, d: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, str]]:
        img = torchvision.transforms.functional.hflip(img)
        for k in list(d):
            if "left" in k:
                d[k.replace("left", "right")] = d.pop(k)
            if "right" in k:
                d[k.replace("right", "left")] = d.pop(k)
        return img, d

    def __getitem__(self, index):
        idx = self.index2id[index]
        chunk: pd.DataFrame = self.df.loc[idx]
        img_path = (self.img_dir / str(idx[0]) / str(idx[1]) / str(idx[2])).with_suffix(".dcm")
        img = utils.load_dcm_img(img_path)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        d = chunk.set_index("condition_level")["severity"].to_dict()

        if self.train and random.random() < 0.5:
            img, d = self.hflip(img, d)

        y_true = {k: torch.tensor(constants.SEVERITY2LABEL.get(d.get(k), -1)) for k in constants.CONDITION_LEVEL}
        return img, y_true


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def load_df(coor_path: Path = constants.COOR_PATH, train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    coor = pd.read_csv(coor_path)
    coor = coor.drop(columns=["x", "y"])
    norm_cond = coor.pop("condition").str.lower().str.replace(" ", "_")
    norm_level = coor.pop("level").str.lower().str.replace("/", "_")
    coor["condition_level"] = norm_cond + "_" + norm_level

    train = utils.load_train(train_path)

    result = train.merge(coor, how="inner", on=["study_id", "condition_level"])

    return result.set_index(["study_id", "series_id", "instance_number"]).sort_index()


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            coor_path: Path = constants.COOR_PATH,
            train_path: Path = constants.TRAIN_PATH,
            img_dir: Path = constants.TRAIN_IMG_DIR,
            n_splits: int = 5,
            this_split: int = 0,
            img_size: int = 256,
            batch_size: int = 64,
            num_workers: int = 8,
        ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.df = load_df(coor_path, train_path)

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        strats = self.df["condition_level"] + "_" + self.df["severity"]
        groups = self.df.reset_index()["study_id"]
        skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
            if i == self.this_split:
                break
        return self.df.iloc[train_ids], self.df.iloc[val_ids]

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.img_dir, train=True, transforms=get_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            _, val_df = self.split()
            self.predict_ds = Dataset(val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size))

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

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dm = DataModule()
    dm.setup("fit")
    for x, y in dm.val_dataloader():
        print(x.shape)
        print({k: v.shape for k, v in y.items()})
        break
