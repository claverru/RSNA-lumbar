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


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: pd.DataFrame,
        df: pd.DataFrame,
        img_size: int,
        transforms: A.Compose
    ):
        self.df = df
        self.train_index = train.index
        self.train = train
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.train_index)

    def get_target(self, idx):
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_imgs(self, study_id) -> torch.Tensor:
        chunk = self.df.loc[study_id]
        imgs = chunk["img_path"].apply(
            lambda x: self.transforms(image=cv2.imread(str(x), cv2.IMREAD_GRAYSCALE))["image"]
        )
        X = torch.stack(imgs.to_list(), 0)
        return X

    def __getitem__(self, index):
        study_id = self.train_index[index]
        target = self.get_target(study_id)
        X = self.get_imgs(study_id)
        mask = torch.tensor([False] * len(X))
        return X, mask, target


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
            A.Perspective(scale=(0.05, 0.2), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=1, min_width=1, fill_value=0, p=0.3),
            A.GaussNoise(var_limit=(0, 0.01), mean=0, p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.2),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ]
    )


def collate_fn(data):
    X, mask, labels = zip(*data)
    X = utils.pad_sequences(X, padding_value=0)
    mask = utils.pad_sequences(mask, padding_value=True)
    labels = utils.cat_dict_tensor(labels, torch.stack)
    return X, mask, labels


def load_df(img_dir: Path = constants.TRAIN_IMG_DIR, desc_path: Path = constants.DESC_PATH):
    df = utils.get_images_df(img_dir)
    desc = utils.load_desc(desc_path)
    df = df.merge(desc, how="inner", on=constants.BASIC_COLS[:2]).drop(columns="series_description")
    df["img_path"] = df.apply(lambda x: utils.get_image_path(*x, img_dir=img_dir, suffix=".png"), axis=1)
    return df.drop(columns=constants.BASIC_COLS[1:]).set_index("study_id").sort_index()


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

    config = yaml.load(open("configs/e2e.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, mask, labels in dm.train_dataloader():
        print()
        print(X.shape, X.dtype)
        print(mask.shape, mask.dtype)
        print({k: (v.shape, v.dtype) for k, v in labels.items()})
        print()
        input()
