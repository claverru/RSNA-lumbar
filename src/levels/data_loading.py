from pathlib import Path
from typing import Optional, Tuple

import cv2
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from src import constants, utils


LEVEL2ID = {lvl: i for i, lvl in enumerate(constants.LEVELS)}


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        train: bool = False,
        transforms: Optional[A.Compose] = None,
    ):
        self.df = df
        self.img_dir = img_dir
        self.train = train
        self.transforms = transforms
        self.index2id = {i: idx for i, idx in enumerate(self.df.index.unique())}

    def __len__(self):
        return len(self.index2id)

    def get_target(self, idx):
        return torch.tensor(self.df.loc[idx, "level"])

    def __getitem__(self, index):
        idx = self.index2id[index]
        study_id, series_id, instance_number = idx
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        x = self.transforms(image=img)["image"]
        target = self.get_target(idx)
        return x, target


class PredictDataset(torch.utils.data.Dataset):
    def __init__(
        self, df: pd.DataFrame, img_dir: Path = constants.TRAIN_IMG_DIR, transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        study_id, series_id, instance_number = self.df.iloc[index].to_list()
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)[..., None]
        img = self.transforms(image=img)["image"]
        return img


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


def get_aug_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-30, 30), shear=(-25, 25), translate_percent=(-0.25, 0.25), scale=(0.8, 1.2), p=0.3),
            A.Perspective(0.1),
            A.GaussNoise(var_limit=(0, 0.02), mean=0, p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.2),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


def load_df(coor_path: Path = constants.COOR_PATH) -> pd.DataFrame:
    df = utils.load_coor(coor_path)
    df = df[df["type"].isin(["left", "right"])]
    df = df.drop(columns=["x", "y", "condition", "condition_level", "type"])
    df = df.set_index(["study_id", "series_id", "instance_number"])
    one_nunique = df.groupby(df.index).transform("nunique") == 1
    df = df[one_nunique.values]
    df = df.groupby(level=df.index.names).first().map(lambda x: LEVEL2ID[x])
    return df.sort_index()


def load_predict_df(img_dir: Path = constants.TRAIN_IMG_DIR, desc_path: Path = constants.DESC_PATH):
    imgs_df = utils.get_images_df(img_dir)
    desc = utils.load_desc(desc_path)
    df = imgs_df.merge(desc, how="inner", on=["study_id", "series_id"])
    df = df[df["series_description"] == "Axial T2"].drop(columns="series_description")
    return df.sort_values(list(df.columns)).reset_index(drop=True)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        coor_path: Path = constants.COOR_PATH,
        train_path: Path = constants.TRAIN_PATH,
        desc_path: Path = constants.DESC_PATH,
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
        self.df = load_df(coor_path)
        self.desc_path = desc_path
        self.train_path = train_path

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return utils.split(self.df, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.img_dir, transforms=get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            # train_df, _ = self.split()
            imgs_df = load_predict_df(self.img_dir, self.desc_path)
            # imgs_df = imgs_df[~imgs_df.set_index(train_df.index.names).index.isin(train_df.index)].reset_index(drop=True)
            self.predict_ds = PredictDataset(imgs_df, self.img_dir, transforms=get_transforms(self.img_size))

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
    import yaml

    config = yaml.load(open("configs/levels.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for x, y in dm.train_dataloader():
        print(x.shape, x.dtype)
        print(y.shape, y.dtype)
        break
