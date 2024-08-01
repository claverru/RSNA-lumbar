from typing import List, Tuple

from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
import lightning as L

from src import constants, utils
from src.d3.constants import PLANE2SIZE


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: pd.DataFrame,
        df: pd.DataFrame,
        img_size: int,
        transforms: A.Compose
    ):
        self.df = df
        self.df_index = set(df.index)
        self.train_index = train.index
        self.train = train
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.train_index)

    def get_target(self, idx):
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_imgs(self, study_id, plane) -> torch.Tensor:
        idx = study_id, plane
        C = PLANE2SIZE[plane]
        if idx in self.df_index:
            chunk: pd.DataFrame = self.df.loc[idx]
            imgs = chunk["img_path"].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))
            max_h, max_w = imgs.apply(lambda x: x.shape).max()
            pad = A.PadIfNeeded(max_h, max_w)
            imgs = imgs.apply(lambda x: pad(image=x)["image"])
            imgs = np.stack(imgs.to_list(), -1)
            X = self.transforms(image=imgs)["image"]
            _, H, W = X.shape
            X = torch.nn.functional.interpolate(X[None, None, ...], (C, H, W))[0, 0]
        else:
            X = torch.zeros((C, self.img_size, self.img_size), dtype=torch.float)
        return X

    def __getitem__(self, index):
        study_id = self.train_index[index]
        target = self.get_target(study_id)
        X = {}
        for plane in constants.DESCRIPTIONS:
            X[plane] = self.get_imgs(study_id, plane)
        return X, target


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ]
    )


def get_aug_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Affine(rotate=(-30, 30), translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), p=0.3),
            A.GaussNoise(var_limit=(0, 0.01), mean=0, p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.2),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ]
    )

def load_df(img_dir: Path = constants.TRAIN_IMG_DIR, desc_path: Path = constants.DESC_PATH):
    df = utils.get_images_df(img_dir)
    desc = utils.load_desc(desc_path)
    df = df.merge(desc, how="inner", on=constants.BASIC_COLS[:2])
    df["img_path"] = df[constants.BASIC_COLS].apply(lambda x: str(utils.get_image_path(*x, img_dir=img_dir, suffix=".png")), axis=1)
    return df.drop(columns=constants.BASIC_COLS[1:]).set_index(["study_id", "series_description"]).sort_index()


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            train_path: Path = constants.TRAIN_PATH,
            desc_path: Path = constants.DESC_PATH,
            img_size: int = 224,
            size_ratio: int = 10,
            img_dir: Path = constants.TRAIN_IMG_DIR,
            n_splits: int = 5,
            this_split: int = 0,
            batch_size: int = 64,
            num_workers: int = 8,
        ):
        super().__init__()
        self.img_size = img_size
        self.size_ratio = size_ratio
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = Path(img_dir)
        self.df = load_df(self.img_dir, desc_path)
        self.train = utils.load_train_flatten(train_path)
        self.train_path = train_path

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split, self.train_path)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.df, self.img_size, get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.df, self.img_size, get_transforms(self.img_size))

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


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/d3.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, target in dm.train_dataloader():
        print()
        print({k: (v.shape, v.dtype) for k, v in X.items()})
        print({k: (v.shape, v.dtype) for k, v in target.items()})
        print()
        input()
