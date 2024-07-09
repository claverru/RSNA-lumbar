from pathlib import Path
import random
from typing import Dict, Optional, Tuple

import cv2
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
import torchvision

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

    def get_labels(self, chunk: pd.DataFrame):
        d = chunk.set_index("condition_level")["severity"].to_dict()
        return {k: torch.tensor(constants.SEVERITY2LABEL.get(d.get(k), 3)) for k in constants.CONDITION_LEVEL}

    def get_coors(self, chunk: pd.DataFrame, shape: Tuple[int, int]):
        d = chunk.set_index("condition_level")[["x", "y"]].to_dict("index")
        result = {}
        for k in constants.CONDITION_LEVEL:
            if k in d:
                result[k] = torch.tensor([d[k]["x"] / shape[1], d[k]["y"] / shape[0]])
            else:
                result[k] = torch.tensor([-1, -1], dtype=torch.float32)
        return result

    def get_plane(self, chunk: pd.DataFrame):
        desc = chunk["series_description"].iloc[0]
        return torch.tensor(constants.DESCRIPTIONS.index(desc))

    def __getitem__(self, index):
        idx = self.index2id[index]
        chunk: pd.DataFrame = self.df.loc[idx]
        img_path = utils.get_image_path(idx[0], idx[1], idx[2], self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)[..., None]
        x = self.transforms(image=img)["image"]
        labels = self.get_labels(chunk)
        coors = self.get_coors(chunk, img.shape)
        plane = self.get_plane(chunk)
        return x, labels, coors, plane


class PredictDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        transforms: Optional[A.Compose] = None
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
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ]
    )


def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomBrightnessContrast(p=0.1),
        A.MotionBlur(p=0.1),
        A.GaussNoise(p=0.1),
        A.Normalize((0.485, ), (0.229, )),
        ToTensorV2()
    ])


def load_df(
    coor_path: Path = constants.COOR_PATH,
    train_path: Path = constants.TRAIN_PATH,
    desc_path: Path = constants.DESC_PATH
) -> pd.DataFrame:
    coor = pd.read_csv(coor_path)
    norm_cond = coor.pop("condition").str.lower().str.replace(" ", "_")
    norm_level = coor.pop("level").str.lower().str.replace("/", "_")
    coor["condition_level"] = norm_cond + "_" + norm_level

    train = utils.load_train(train_path)
    df = train.merge(coor, how="inner", on=["study_id", "condition_level"])

    desc = pd.read_csv(desc_path)
    df = df.merge(desc, on=["study_id", "series_id"])

    return df.set_index(["study_id", "series_id", "instance_number"]).sort_index()


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
        self.df = load_df(coor_path, train_path, desc_path)

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
            self.train_ds = Dataset(train_df, self.img_dir, train=True, transforms=get_train_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            train_df, _ = self.split()
            imgs_df = utils.get_images_df(self.img_dir)
            imgs_df = imgs_df[~imgs_df.set_index(train_df.index.names).index.isin(train_df.index)].reset_index(drop=True)
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

    config = yaml.load(open("configs/image.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for x, y, coors, plane in dm.val_dataloader():
        print(x.shape, x.dtype)
        print(plane.shape, plane.dtype)
        print({k: (v.shape, v.dtype) for k, v in y.items()})
        print({k: (v.shape, v.dtype) for k, v in coors.items()})
        break
