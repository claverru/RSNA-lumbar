from pathlib import Path
import random
from typing import Optional, Tuple

import cv2
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from src import constants, utils


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.index2id = {i: idx for i, idx in enumerate(self.df.index.unique())}
        self.series_ids = np.array([v[1] for v in self.index2id.values()])

    def __len__(self):
        return len(self.index2id)

    def load_image(self,study_id, series_id, instance_number ):
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)[..., None]
        img = self.transforms(image=img)["image"]
        return img

    def get_random_idx(self, series_id):
        if random.random() < 0.5:
            return np.random.choice(np.where(self.series_ids == series_id)[0]), 1
        else:
            return np.random.choice(np.where(self.series_ids != series_id)[0]), 0

    def __getitem__(self, index):
        idx = self.index2id[index]
        img1 = self.load_image(idx[0], idx[1], idx[2])
        random_index, target = self.get_random_idx(idx[1])
        random_idx = self.index2id[random_index]
        img2 = self.load_image(random_idx[0], random_idx[1], random_idx[2])
        target = torch.tensor(target)
        return img1, img2, target


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
            self.train_ds = Dataset(train_df, self.img_dir, transforms=get_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, transforms=get_transforms(self.img_size))

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
    for x1, x2, y in dm.val_dataloader():
        print(x1.shape)
        print(x2.shape)
        print(y.shape)
        break
