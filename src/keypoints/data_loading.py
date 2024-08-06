from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import cv2
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

    def get_keypoints(self, chunk: pd.DataFrame):
        return chunk[["x", "y"]].values

    def keypoints2xy(self, keypoints, shape):
        C, H, W = shape
        denominator = torch.tensor([[W, H]])
        return torch.tensor(keypoints) / denominator

    def __getitem__(self, index):
        idx = self.index2id[index]
        study_id, series_id, instance_number, plane = idx
        chunk: pd.DataFrame = self.df.loc[idx]
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        keypoints = self.get_keypoints(chunk)

        transformed = self.transforms(image=img, keypoints=keypoints)
        x = transformed["image"]
        keypoints = transformed["keypoints"]  # xy

        if plane == "Axial T2":
            keypoints_axial = sorted(keypoints, key=lambda x: x[0])  # left to right
            xy_axial = self.keypoints2xy(keypoints_axial, x.shape)
            xy_saggital = -torch.ones(5, 2)
        else:
            keypoints_saggital = sorted(keypoints, key=lambda x: x[1])  # top to botton
            xy_saggital = self.keypoints2xy(keypoints_saggital, x.shape)
            xy_axial = -torch.ones(2, 2)

        return x, xy_saggital, xy_axial


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
        chunk = self.df.loc[index]
        img_path = utils.get_image_path(
            chunk["study_id"], chunk["series_id"], chunk["instance_number"], self.img_dir, suffix=".png"
        )
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        transformed = self.transforms(image=img)
        x = transformed["image"]
        return x


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_train_transforms(img_size):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=(-0.3, 0.3),
                rotate_limit=(-10, 10),
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.MotionBlur(p=0.3),
            A.GaussNoise(p=0.3),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_predict_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


def load_df(coor_path: Path = constants.COOR_PATH, desc_path: Path = constants.DESC_PATH) -> pd.DataFrame:
    df = utils.load_coor(coor_path)
    desc = utils.load_desc(desc_path)
    df = df[constants.BASIC_COLS + ["x", "y", "level"]]
    df = df.merge(desc, how="inner", on=constants.BASIC_COLS[:2])
    is_axial = df["series_description"] == "Axial T2"
    is_sagittal = ~is_axial
    size = df.groupby(constants.BASIC_COLS).transform("size")
    nunique = df.groupby(constants.BASIC_COLS)["level"].transform("nunique")
    df = df[(is_axial & size.eq(2) & nunique.eq(1)) | (is_sagittal & size.eq(5) & nunique.eq(5))]
    df = df.drop(columns="level")
    df = df.set_index(constants.BASIC_COLS + ["series_description"]).sort_index()
    return df


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        coor_path: Path = constants.COOR_PATH,
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
        self.df = load_df(coor_path, desc_path)

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return utils.split(self.df, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.img_dir, train=True, transforms=get_train_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            img_df = utils.get_images_df(self.img_dir)
            self.predict_ds = PredictDataset(img_df, self.img_dir, get_predict_transforms(self.img_size))

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
    for x, xy_saggital, xy_axial, level in dm.train_dataloader():
        print(x.shape, x.dtype)
        print(xy_saggital.shape, xy_saggital.dtype)
        print(xy_axial.shape, xy_axial.dtype)
        print(level.shape, level.dtype)
        input()
